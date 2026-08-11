#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common.h"
#include "cuda_utils.h"

// Dispatch helper that binds groups and determines the optimal BC_PER_THREAD at compile time
template <typename Functor>
inline void dispatch_groups(int groups_val, Functor&& launch) {
    switch (groups_val) {
        case 1:   launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 8>{}); break;
        case 2:   launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{}); break;
        case 4:   launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{}); break;
        case 8:   launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{}); break;
        default:  TORCH_CHECK(false, "Unsupported groups: ", groups_val);
    }
}

/* Forward pass
// input: [B, C, H_in, W_in]
// weight: [B, C, K, H_weight, W_weight]
// output: [B, C, H_out, W_out]
// get down-scaled weight map, upsampling simulataneously! "on the flying"
// 1. get input data pixel values
// 2. get weight map + upsample as target resolution (utilize TMU)
// 3. element wise multiplication & sum along kernel index axis (weight adaptive convolution)
*/
namespace {
template <typename scalar_t, int GROUPS, int BC_PER_THREAD>
__global__ void flying_conv2d_fwd_kernel(scalar_t* __restrict__ output, const scalar_t* __restrict__ input,
                                         cudaTextureObject_t texObj, cudaTextureObject_t texBiasObj, int B, int C,
                                         int C_weight, int H_in, int W_in, int H_weight, int W_weight, int H_out,
                                         int W_out, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
                                         int pad_w, int dil_h, int dil_w, float inv_scale_h, float inv_scale_w,
                                         bool has_bias, int bc_start, int bc_end) {
    int w_out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (w_out_idx >= W_out || h_out_idx >= H_out) return;

    int base_bc = bc_start + blockIdx.z * BC_PER_THREAD;
    if (base_bc >= bc_end) return;

    float x_coord = (float)(w_out_idx + 0.5f) * inv_scale_w;
    float y_coord = (float)(h_out_idx + 0.5f) * inv_scale_h;

    int Kernel_total = kernel_h * kernel_w;
    int c_stride = H_in * W_in;
    int groups_c_stride = GROUPS * c_stride;
    int out_c_stride = H_out * W_out;
    int groups_out_c_stride = GROUPS * out_c_stride;
    int base_layer = (base_bc - bc_start) * Kernel_total;

    scalar_t acc[BC_PER_THREAD][GROUPS] = {};
    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_kernel_idx = kh * dil_h;
        int h_in_idx = h_out_idx * stride_h - pad_h + h_kernel_idx;
        if (h_in_idx < 0 || h_in_idx >= H_in) continue;

        for (int kw = 0; kw < kernel_w; ++kw) {
            int k_idx = kh * kernel_w + kw;
            int w_kernel_idx = kw * dil_w;
            int w_in_idx = w_out_idx * stride_w - pad_w + w_kernel_idx;
            if (w_in_idx < 0 || w_in_idx >= W_in) continue;

            int spatial_in_idx = h_in_idx * W_in + w_in_idx;
            const scalar_t* base_in_ptr = input + (base_bc * groups_c_stride) + spatial_in_idx;
            int base_layer_k = base_layer + k_idx;

#pragma unroll
            for (int bc_offs = 0; bc_offs < BC_PER_THREAD; ++bc_offs) {
                if (base_bc + bc_offs >= bc_end) continue;
                int layer = base_layer_k + bc_offs * Kernel_total;
                scalar_t weight_val = tex2DLayered<float>(texObj, x_coord, y_coord, layer);

                const scalar_t* input_ptr = base_in_ptr + bc_offs * groups_c_stride;
#pragma unroll
                for (int g = 0; g < GROUPS; ++g) {
                    acc[bc_offs][g] += input_ptr[g * c_stride] * weight_val;
                }
            }
        }
    }

    if (has_bias) {
        int base_layer_bias = base_bc - bc_start;
#pragma unroll
        for (int bc_offs = 0; bc_offs < BC_PER_THREAD; ++bc_offs) {
            if (base_bc + bc_offs >= bc_end) continue;
            int layer_bias = base_layer_bias + bc_offs;
            scalar_t bias_val = tex2DLayered<float>(texBiasObj, x_coord, y_coord, layer_bias);
#pragma unroll
            for (int g = 0; g < GROUPS; ++g) {
                acc[bc_offs][g] += bias_val;
            }
        }
    }

    scalar_t* out_ptr = output + base_bc * groups_out_c_stride + h_out_idx * W_out + w_out_idx;
#pragma unroll
    for (int bc_offs = 0; bc_offs < BC_PER_THREAD; ++bc_offs) {
        if (base_bc + bc_offs < bc_end) {
            scalar_t* g_ptr = out_ptr;
#pragma unroll
            for (int g = 0; g < GROUPS; ++g) {
                *g_ptr = acc[bc_offs][g];
                g_ptr += out_c_stride;
            }
        }
        out_ptr += groups_out_c_stride;
    }
}

/* Backward pass for input data
// grad_input: [B, C, H_in, W_in]
// grad_output: [B, C, H_out, W_out]
// weight: [B, C, K, H_weight, W_weight]
// similar to the deconvolution (but on-the-flying calculation)
// 1. im2col grad_output data to the same resolution as output (for transposed convolution)
// 2. get upsampled weight for propagation & element wise multiplication
// 3. sum along kernel axis
*/
template <typename scalar_t, int GROUPS, int BC_PER_THREAD>
__global__ void flying_conv2d_input_bwd_kernel(scalar_t* __restrict__ grad_input,
                                               const scalar_t* __restrict__ grad_output, cudaTextureObject_t texObj,
                                               int B, int C, int C_weight, int H_in, int W_in, int H_weight,
                                               int W_weight, int H_out, int W_out, int kernel_h, int kernel_w,
                                               int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w,
                                               float inv_scale_h, float inv_scale_w, FastDivmod stride_h_div,
                                               FastDivmod stride_w_div, FastDivmod C_weight_div, int bc_start,
                                               int bc_end) {
    int w_in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_in_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_in_idx >= W_in || h_in_idx >= H_in) return;

    int Kernel_total = kernel_h * kernel_w;
    int c_out_stride = H_out * W_out;
    int c_in_stride = H_in * W_in;

    int base_bc = bc_start + blockIdx.z * BC_PER_THREAD;
    if (base_bc >= bc_end) return;

    int grad_out_bases[BC_PER_THREAD];
    int grad_inp_bases[BC_PER_THREAD];
#pragma unroll
    for (int bc_offs = 0; bc_offs < BC_PER_THREAD; ++bc_offs) {
        int bc_idx = base_bc + bc_offs;
        if (bc_idx >= bc_end) continue;
        unsigned int b_idx, c_idx;
        C_weight_div.divmod(bc_idx, b_idx, c_idx);
        int c_base = c_idx * GROUPS;
        grad_out_bases[bc_offs] = (b_idx * C + c_base) * c_out_stride;
        grad_inp_bases[bc_offs] = (b_idx * C + c_base) * c_in_stride;
    }

    scalar_t acc[BC_PER_THREAD][GROUPS] = {};
    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_kernel_idx = kh * dil_h;
        int h_num = (int)h_in_idx + pad_h - h_kernel_idx;
        if (h_num < 0) continue;

        unsigned int h_out_idx = h_num, h_rem = 0;
        if (stride_h != 1) stride_h_div.divmod((unsigned int)h_num, h_out_idx, h_rem);
        if (h_rem != 0 || h_out_idx >= H_out) continue;

        float h_weight_coord = (h_out_idx + 0.5f) * inv_scale_h;

        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_kernel_idx = kw * dil_w;
            int w_num = (int)w_in_idx + pad_w - w_kernel_idx;
            if (w_num < 0) continue;

            unsigned int w_out_idx = w_num, w_rem = 0;
            if (stride_w != 1) stride_w_div.divmod((unsigned int)w_num, w_out_idx, w_rem);
            if (w_rem != 0 || w_out_idx >= W_out) continue;

            float w_weight_coord = (w_out_idx + 0.5f) * inv_scale_w;
            int kernel_idx = kh * kernel_w + kw;
            int spatial_out_idx = h_out_idx * W_out + w_out_idx;

#pragma unroll
            for (int bc_offs = 0; bc_offs < BC_PER_THREAD; ++bc_offs) {
                if (base_bc + bc_offs >= bc_end) continue;
                int layer = (base_bc + bc_offs - bc_start) * Kernel_total + kernel_idx;
                scalar_t weight_val = tex2DLayered<float>(texObj, w_weight_coord, h_weight_coord, layer);
                const scalar_t* grad_out_ptr = grad_output + grad_out_bases[bc_offs] + spatial_out_idx;
#pragma unroll
                for (int g = 0; g < GROUPS; ++g) {
                    acc[bc_offs][g] += grad_out_ptr[g * c_out_stride] * weight_val;
                }
            }
        }
    }
    int spatial_in_idx = h_in_idx * W_in + w_in_idx;
    scalar_t* out_ptr = grad_input + spatial_in_idx;

#pragma unroll
    for (int bc_offs = 0; bc_offs < BC_PER_THREAD; ++bc_offs) {
        if (base_bc + bc_offs >= bc_end) continue;
        scalar_t* g_ptr = out_ptr + grad_inp_bases[bc_offs];
#pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            *g_ptr = acc[bc_offs][g];
            g_ptr += c_in_stride;
        }
    }
}

/* Backward pass for weight data: Gather version
// aL/aW = aL/aY * aY/aW = grad_output * input data (chain rule)

// grad_weight (to be calculated): [B, C, H_out // scale_h, W_out // scale_w, K]
// grad_output: [B, C, H_out, W_out]
// input: [B, C, H_in, W_in]
// 1. im2col for grad_output to the same resolution as input data
// 2. element wise multiplication (convolution)
// 3. sum along neighborhood axis for inverse bilinear interpolation
 */
template <typename scalar_t, int GROUPS>
__global__ void flying_conv2d_weight_bwd_kernel(
    scalar_t* __restrict__ grad_weight, const scalar_t* __restrict__ grad_output, const scalar_t* __restrict__ input,
    int B, int C, int C_weight, int H_in, int W_in, int H_weight, int W_weight, int H_out, int W_out, int kernel_h,
    int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int scale_h, int scale_w,
    float inv_scale_h, float inv_scale_w, FastDivmod C_weight_div, FastDivmod K_div) {
    int wk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int bc_idx = blockIdx.z;
    int Kernel_total = kernel_h * kernel_w;
    if (wk_idx >= W_weight * Kernel_total || h_idx >= H_weight || bc_idx >= B * C_weight) return;

    unsigned int w_idx, k;
    K_div.divmod(wk_idx, w_idx, k);

    int c_in_stride = H_in * W_in;
    int c_out_stride = H_out * W_out;
    int weight_spatial_stride = H_weight * W_weight;

    unsigned int b_idx, c_w_idx;
    C_weight_div.divmod(bc_idx, b_idx, c_w_idx);
    int c_base = c_w_idx * GROUPS;
    int in_base = (b_idx * C + c_base) * c_in_stride;
    int grad_out_base = (b_idx * C + c_base) * c_out_stride;

    int kh = k / kernel_w;
    int kw = k - kh * kernel_w;
    int h_in_offs = kh * dil_h - pad_h;
    int w_in_offs = kw * dil_w - pad_w;

    int h_out_start = (int)ceilf(((float)h_idx - 0.5f) * scale_h - 0.5f);
    int h_out_end = (int)floorf(((float)h_idx + 1.5f) * scale_h - 0.5f) + 1;
    if (h_idx == (unsigned int)(H_weight - 1)) h_out_end = H_out;

    int h_in_min_out = (h_in_offs >= 0) ? 0 : (-h_in_offs + stride_h - 1) / stride_h;
    int h_in_max_out = (H_in - h_in_offs + stride_h - 1) / stride_h;
    h_out_start = max(max(0, h_out_start), h_in_min_out);
    h_out_end = min(min(H_out, h_out_end), h_in_max_out);

    int w_out_start = (int)ceilf(((float)w_idx - 0.5f) * scale_w - 0.5f);
    int w_out_end = (int)floorf(((float)w_idx + 1.5f) * scale_w - 0.5f) + 1;
    if (w_idx == (unsigned int)(W_weight - 1)) w_out_end = W_out;

    int w_in_min_out = (w_in_offs >= 0) ? 0 : (-w_in_offs + stride_w - 1) / stride_w;
    int w_in_max_out = (W_in - w_in_offs + stride_w - 1) / stride_w;
    w_out_start = max(max(0, w_out_start), w_in_min_out);
    w_out_end = min(min(W_out, w_out_end), w_in_max_out);

    int spatial_in_h = (h_out_start * stride_h + h_in_offs) * W_in;
    int spatial_out_h = h_out_start * W_out;
    int stride_h_W_in = stride_h * W_in;
    float w_curr_start = (w_out_start + 0.5f) * inv_scale_w - 0.5f;
    int w_in_idx_start = w_out_start * stride_w + w_in_offs;
    bool is_last_h = (h_idx == (unsigned int)(H_weight - 1));
    bool is_last_w = (w_idx == (unsigned int)(W_weight - 1));
    float fh = (float)h_idx;
    float fw = (float)w_idx;

    scalar_t acc = 0;
    scalar_t* grad_weight_ptr = grad_weight + bc_idx * Kernel_total * weight_spatial_stride +
                                k * weight_spatial_stride + h_idx * W_weight + w_idx;
    float h_curr_iter = (h_out_start + 0.5f) * inv_scale_h - 0.5f;
    for (int h_out_curr = h_out_start; h_out_curr < h_out_end;
         ++h_out_curr, h_curr_iter += inv_scale_h, spatial_in_h += stride_h_W_in, spatial_out_h += W_out) {
        float h_curr = fmaxf(h_curr_iter, 0.0f);
        float coeff_h = 1.0f - fabsf(h_curr - fh);
        if (is_last_h && h_curr >= fh) coeff_h = 1.0f;
        if (coeff_h <= 0.0f) continue;

        float w_curr_iter = w_curr_start;
        int w_in_idx = w_in_idx_start;
        for (int w_out_curr = w_out_start; w_out_curr < w_out_end;
             ++w_out_curr, w_curr_iter += inv_scale_w, w_in_idx += stride_w) {
            float w_curr = fmaxf(w_curr_iter, 0.0f);
            float coeff_w = 1.0f - fabsf(w_curr - fw);
            if (is_last_w && w_curr >= fw) coeff_w = 1.0f;
            if (coeff_w <= 0.0f) continue;

            float coeff = coeff_h * coeff_w;
            const scalar_t* in_ptr = input + in_base + spatial_in_h + w_in_idx;
            const scalar_t* grad_out_ptr = grad_output + grad_out_base + spatial_out_h + w_out_curr;

            scalar_t sum = 0;
#pragma unroll
            for (int g = 0; g < GROUPS; ++g) {
                sum += in_ptr[g * c_in_stride] * grad_out_ptr[g * c_out_stride];
            }
            acc += sum * (scalar_t)coeff;
        }
    }
    *grad_weight_ptr = acc;
}

/* Backward pass for bias data: gather version
// aL/aW = aL/aY * aY/aW = grad_output * input data (chain rule)
// grad_bias (to be calculated): [B, C, H_out // scale_h, W_out // scale_w]
// grad_output: [B, C, H_out, W_out]
// 1. get four neighborhood pixels from grad_output data
// 2. sum along neighborhood axis for inverse bilinear interpolation
*/
template <typename scalar_t, int GROUPS>
__global__ void flying_conv2d_bias_bwd_kernel(scalar_t* __restrict__ grad_bias,
                                              const scalar_t* __restrict__ grad_output, int B, int C, int C_weight,
                                              int H_weight, int W_weight, int H_out, int W_out, int scale_h,
                                              int scale_w, float inv_scale_h, float inv_scale_w,
                                              FastDivmod C_weight_div) {
    // --- Thread index decomposition ---
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int bc_idx = blockIdx.z;
    if (w_idx >= W_weight || h_idx >= H_weight || bc_idx >= B * C_weight) return;

    unsigned int b_idx, c_dw_idx;
    C_weight_div.divmod(bc_idx, b_idx, c_dw_idx);
    int c_base = c_dw_idx * GROUPS;
    int chan_stride = H_out * W_out;
    int go_base = (b_idx * C + c_base) * chan_stride;
    scalar_t* grad_bias_ptr = grad_bias + bc_idx * H_weight * W_weight + h_idx * W_weight + w_idx;

    int h_out_start = (int)ceilf(((float)h_idx - 0.5f) * scale_h - 0.5f);
    int h_out_end = (int)floorf(((float)h_idx + 1.5f) * scale_h - 0.5f) + 1;
    if (h_idx == H_weight - 1) h_out_end = H_out;
    h_out_start = max(0, h_out_start);
    h_out_end = min(H_out, h_out_end);

    int w_out_start = (int)ceilf(((float)w_idx - 0.5f) * scale_w - 0.5f);
    int w_out_end = (int)floorf(((float)w_idx + 1.5f) * scale_w - 0.5f) + 1;
    if (w_idx == W_weight - 1) w_out_end = W_out;
    w_out_start = max(0, w_out_start);
    w_out_end = min(W_out, w_out_end);

    int spatial_out_h = h_out_start * W_out;
    float w_curr_start = (w_out_start + 0.5f) * inv_scale_w - 0.5f;
    bool is_last_h = (h_idx == H_weight - 1);
    bool is_last_w = (w_idx == W_weight - 1);
    float fh = (float)h_idx;
    float fw = (float)w_idx;

    scalar_t acc = 0;
    float h_curr_iter = (h_out_start + 0.5f) * inv_scale_h - 0.5f;
    for (int h_out_curr = h_out_start; h_out_curr < h_out_end;
         ++h_out_curr, h_curr_iter += inv_scale_h, spatial_out_h += W_out) {
        float h_curr = fmaxf(h_curr_iter, 0.0f);
        float coeff_h = 1.0f - fabsf(h_curr - fh);
        if (is_last_h && h_curr >= fh) coeff_h = 1.0f;
        if (coeff_h <= 0.0f) continue;

        float w_curr_iter = w_curr_start;
        const scalar_t* grad_out_ptr = grad_output + go_base + spatial_out_h + w_out_start;
        for (int w_out_curr = w_out_start; w_out_curr < w_out_end;
             ++w_out_curr, w_curr_iter += inv_scale_w, ++grad_out_ptr) {
            float w_curr = fmaxf(w_curr_iter, 0.0f);
            float coeff_w = 1.0f - fabsf(w_curr - fw);
            if (is_last_w && w_curr >= fw) coeff_w = 1.0f;
            if (coeff_w <= 0.0f) continue;

            float coeff = coeff_h * coeff_w;
#pragma unroll
            for (int g = 0; g < GROUPS; ++g) {
                acc += grad_out_ptr[g * chan_stride] * (scalar_t)coeff;
            }
        }
    }
    *grad_bias_ptr = acc;
}
}  // namespace

namespace cuda_ver {
torch::Tensor flying_conv2d_fwd(torch::Tensor input, torch::Tensor weight, at::optional<torch::Tensor> bias,
                                int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h,
                                int dil_w, int scale_h, int scale_w, int groups) {
    int B = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int C_weight = weight.size(1);
    int K = weight.size(2);
    int H_weight = weight.size(3);
    int W_weight = weight.size(4);

    int H_out = H_weight * scale_h;
    int W_out = W_weight * scale_w;

    float inv_scale_h = 1.0f / (float)scale_h;
    float inv_scale_w = 1.0f / (float)scale_w;

    FastDivmod C_weight_div(C_weight);

    int BC_total = B * C_weight;
    int bc_per_chunk = (K > 0) ? (2048 / K) : 2048;
    if (bc_per_chunk == 0) bc_per_chunk = 1;

    auto& weightManager = TexturePool::get(W_weight, H_weight, 2048, 0 /* weight tag */);
    auto& biasManager = TexturePool::get(W_weight, H_weight, 2048, 1 /* bias tag */);

    auto output = torch::zeros({B, C, H_out, W_out}, input.options());
    auto contig_weight = weight.contiguous();
    bool has_bias = bias.has_value();
    auto contig_bias = has_bias ? bias.value().contiguous() : torch::Tensor();

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    for (int bc_start = 0; bc_start < BC_total; bc_start += bc_per_chunk) {
        int bc_end = std::min(bc_start + bc_per_chunk, BC_total);
        int bc_range = bc_end - bc_start;

        weightManager.copyFrom(contig_weight.data_ptr<float>() + (size_t)bc_start * K * H_weight * W_weight,
                               W_weight * sizeof(float), W_weight, H_weight, bc_range * K,
                               at::cuda::getCurrentCUDAStream());

        cudaTextureObject_t texObj = weightManager.getTexture(texDesc);
        cudaTextureObject_t texBiasObj = 0;

        if (has_bias) {
            biasManager.copyFrom(contig_bias.data_ptr<float>() + (size_t)bc_start * H_weight * W_weight,
                                 W_weight * sizeof(float), W_weight, H_weight, bc_range,
                                 at::cuda::getCurrentCUDAStream());
            texBiasObj = biasManager.getTexture(texDesc);
        }

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "flying_conv2d_fwd", ([&] {
                dispatch_groups(groups, [&](auto G_const, auto BC_const) {
                    constexpr int G = decltype(G_const)::value;
                    constexpr int BC = decltype(BC_const)::value;
                    dim3 threads(32, 8);
                    dim3 blocks((W_out + threads.x - 1) / threads.x,
                                (H_out + threads.y - 1) / threads.y,
                                (bc_range + BC - 1) / BC);
                    flying_conv2d_fwd_kernel<scalar_t, G, BC><<<blocks, threads>>>(
                        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), texObj, texBiasObj, B, C,
                        C_weight, H_in, W_in, H_weight, W_weight, H_out, W_out, kernel_h, kernel_w, stride_h,
                        stride_w, pad_h, pad_w, dil_h, dil_w, inv_scale_h, inv_scale_w, has_bias, bc_start, bc_end
                    );
                });
            }));

        // Texture objects are managed by the Pool
    }

    return output;
}

torch::Tensor flying_conv2d_input_bwd(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                      int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
                                      int dil_h, int dil_w, int scale_h, int scale_w, int groups) {
    int B = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int C_weight = weight.size(1);
    int K = weight.size(2);
    int H_weight = weight.size(3);
    int W_weight = weight.size(4);

    int BC_total = B * C_weight;
    int H_out = H_weight * scale_h;
    int W_out = W_weight * scale_w;

    float inv_scale_h = 1.0f / (float)scale_h;
    float inv_scale_w = 1.0f / (float)scale_w;

    FastDivmod stride_h_div(stride_h);
    FastDivmod stride_w_div(stride_w);
    FastDivmod C_weight_div(C_weight);

    int bc_per_chunk = (K > 0) ? (2048 / K) : 2048;
    if (bc_per_chunk == 0) bc_per_chunk = 1;

    auto& weightManager = TexturePool::get(W_weight, H_weight, 2048, 0 /* weight tag */);

    auto grad_input = torch::zeros_like(input);
    auto contig_weight = weight.contiguous();

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    for (int bc_start = 0; bc_start < BC_total; bc_start += bc_per_chunk) {
        int bc_end = std::min(bc_start + bc_per_chunk, BC_total);
        int bc_range = bc_end - bc_start;

        weightManager.copyFrom(contig_weight.data_ptr<float>() + (size_t)bc_start * K * H_weight * W_weight,
                               W_weight * sizeof(float), W_weight, H_weight, bc_range * K,
                               at::cuda::getCurrentCUDAStream());

        cudaTextureObject_t texObj = weightManager.getTexture(texDesc);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "flying_conv2d_input_bwd", ([&] {
                dispatch_groups(groups, [&](auto G_const, auto BC_const) {
                    constexpr int G = decltype(G_const)::value;
                    constexpr int BC = decltype(BC_const)::value;
                    dim3 threads(32, 8);
                    dim3 blocks((W_in + threads.x - 1) / threads.x,
                                (H_in + threads.y - 1) / threads.y,
                                (bc_range + BC - 1) / BC);
                    flying_conv2d_input_bwd_kernel<scalar_t, G, BC><<<blocks, threads>>>(
                        grad_input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), texObj, B, C,
                        C_weight, H_in, W_in, H_weight, W_weight, H_out, W_out, kernel_h, kernel_w, stride_h,
                        stride_w, pad_h, pad_w, dil_h, dil_w, inv_scale_h, inv_scale_w, stride_h_div,
                        stride_w_div, C_weight_div, bc_start, bc_end
                    );
                });
            }));

        // Texture object is managed by the Pool
    }
    return grad_input;
}

torch::Tensor flying_conv2d_weight_bwd(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                       int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
                                       int dil_h, int dil_w, int scale_h, int scale_w, int groups) {
    int B = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int C_weight = weight.size(1);
    int K_weight = weight.size(2);
    int H_weight = weight.size(3);
    int W_weight = weight.size(4);
    int BC_total = B * C_weight;
    int H_out = H_weight * scale_h;
    int W_out = W_weight * scale_w;

    float inv_scale_h = 1.0f / (float)scale_h;
    float inv_scale_w = 1.0f / (float)scale_w;

    FastDivmod C_weight_div(C_weight);
    FastDivmod K_div(kernel_h * kernel_w);

    auto grad_weight = torch::zeros_like(weight);

    int K = kernel_h * kernel_w;
    int WK = W_weight * K;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "flying_conv2d_weight_bwd", ([&] {
            dispatch_groups(groups, [&](auto G_const, auto BC_const) {
                constexpr int G = decltype(G_const)::value;
                dim3 threads(32, 8);
                dim3 blocks((WK + threads.x - 1) / threads.x,
                            (H_weight + threads.y - 1) / threads.y,
                            BC_total);
                flying_conv2d_weight_bwd_kernel<scalar_t, G><<<blocks, threads>>>(
                    grad_weight.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(), B, C, C_weight, H_in, W_in, H_weight, W_weight, H_out, W_out,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, scale_h, scale_w,
                    inv_scale_h, inv_scale_w, C_weight_div, K_div
                );
            });
        }));
    return grad_weight;
}

torch::Tensor flying_conv2d_bias_bwd(torch::Tensor grad_output, torch::Tensor bias, int scale_h, int scale_w,
                                     int groups) {
    int B = grad_output.size(0);
    int C = grad_output.size(1);

    int C_weight = bias.size(1);
    int H_weight = bias.size(2);
    int W_weight = bias.size(3);

    int BC_total = B * C_weight;
    int H_out = H_weight * scale_h;
    int W_out = W_weight * scale_w;

    float inv_scale_h = 1.0f / (float)scale_h;
    float inv_scale_w = 1.0f / (float)scale_w;

    FastDivmod C_weight_div(C_weight);

    auto grad_bias = torch::zeros_like(bias);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.scalar_type(), "flying_conv2d_bias_bwd", ([&] {
            dispatch_groups(groups, [&](auto G_const, auto BC_const) {
                constexpr int G = decltype(G_const)::value;
                dim3 threads(32, 8);
                dim3 blocks((W_weight + threads.x - 1) / threads.x,
                            (H_weight + threads.y - 1) / threads.y,
                            BC_total);
                flying_conv2d_bias_bwd_kernel<scalar_t, G><<<blocks, threads>>>(
                    grad_bias.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), B, C, C_weight, H_weight,
                    W_weight, H_out, W_out, scale_h, scale_w, inv_scale_h, inv_scale_w, C_weight_div
                );
            });
        }));
    return grad_bias;
}
}  // namespace cuda_ver
