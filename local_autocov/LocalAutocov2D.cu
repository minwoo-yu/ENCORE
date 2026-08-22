#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>

#include "common.h"

#define TILE_DIM 16

template <int PH, int PW, bool do_accum>
__global__ void local_autocov2d(const float* __restrict__ input, float* __restrict__ output, const int B,
                                 const int C_total, const int H, const int W, const int H_out, const int W_out,
                                 const int kH, const int kW, const int dH, const int dW, const int padH, const int padW,
                                 const int dilH, const int dilW, const int dil_pH, const int dil_pW, const int tile_H,
                                 const int tile_W, const int c_curr, const float inv_norm, const FastDivmod fd_tileW) {
    extern __shared__ float s_tile[];
    float* s_array = &s_tile[c_curr * tile_H * tile_W];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;

    const int w_out_start = bx * TILE_DIM, h_out_start = by * TILE_DIM;
    const int w_out = w_out_start + tx, h_out = h_out_start + ty;
    const int P_total = PH * PW;
    const int tile_size = tile_H * tile_W;

    const int in_h_start = h_out_start * dH - padH - (PH / 2) * dil_pH;
    const int in_w_start = w_out_start * dW - padW - (PW / 2) * dil_pW;
    float accum[PH * PW] = {0.0f};

    // Load current chunk of channels into shared memory
    for (int c = 0; c < c_curr; ++c) {
        float* s_ptr = &s_tile[c * tile_size];
        const float* inp_ptr = input + (b_idx * C_total + c) * H * W;
        for (int i = ty * TILE_DIM + tx; i < tile_size; i += TILE_DIM * TILE_DIM) {
            unsigned int r, c_idx;
            fd_tileW.divmod(i, r, c_idx);
            int ch = in_h_start + (int)r, cw = in_w_start + (int)c_idx;
            s_ptr[i] = (ch >= 0 && ch < H && cw >= 0 && cw < W) ? inp_ptr[ch * W + cw] : 0.0f;
        }
    }
    __syncthreads();

#pragma unroll
    for (int p_idx = 0; p_idx < PH * PW; ++p_idx) {
        const int ph_offs = (p_idx / PW - PH / 2) * dil_pH;
        const int pw_offs = (p_idx % PW - PW / 2) * dil_pW;

        // Step 1: Collaborative Pointwise Channel Covariance
        for (int i = ty * TILE_DIM + tx; i < tile_size; i += TILE_DIM * TILE_DIM) {
            unsigned int r, c_idx;
            fd_tileW.divmod(i, r, c_idx);
            int neighb_H = (int)r + ph_offs, neighb_W = (int)c_idx + pw_offs;
            float local_sum = 0.0f;
            if (neighb_H >= 0 && neighb_H < tile_H && neighb_W >= 0 && neighb_W < tile_W) {
                const int neighbor_pixel_offs = neighb_H * tile_W + neighb_W;
#pragma unroll 4
                for (int ch = 0; ch < c_curr; ++ch) {
                    local_sum += s_tile[ch * tile_size + i] * s_tile[ch * tile_size + neighbor_pixel_offs];
                }
            }
            s_array[i] = local_sum;
        }
        __syncthreads();

        // Step 2: Spatial Window Sum
        if (h_out < H_out && w_out < W_out) {
            const int s_h = ty * dH + (PH / 2) * dil_pH;
            const int s_w = tx * dW + (PW / 2) * dil_pW;
            float current_window_sum = 0.0f;
            for (int u = 0; u < kH; ++u) {
                const int s_h1 = s_h + u * dilH;
                for (int v = 0; v < kW; ++v) {
                    const int s_w1 = s_w + v * dilW;
                    current_window_sum += s_array[s_h1 * tile_W + s_w1];
                }
            }
            accum[p_idx] += current_window_sum;
        }
        __syncthreads();
    }

    if (h_out < H_out && w_out < W_out) {
        float* out_ptr = output + (b_idx * P_total * H_out * W_out + h_out * W_out + w_out);
        const int spatial_size = H_out * W_out;
#pragma unroll
        for (int i = 0; i < P_total; ++i) {
            float val = accum[i] * inv_norm;
            if (do_accum)
                *out_ptr += val;
            else
                *out_ptr = val;
            out_ptr += spatial_size;
        }
    }
}

at::Tensor local_autocov2d(const at::Tensor& input, int kH, int kW, int pH, int pW, int strH, int strW, int padH,
                            int padW, int dilH, int dilW, int dil_pH, int dil_pW) {
    const int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int H_out = (H + 2 * padH - dilH * (kH - 1) - 1) / strH + 1,
              W_out = (W + 2 * padW - dilW * (kW - 1) - 1) / strW + 1;
    auto output = at::empty({B, pH * pW, H_out, W_out}, input.options());

    const dim3 threads(TILE_DIM, TILE_DIM, 1),
        blocks((W_out + TILE_DIM - 1) / TILE_DIM, (H_out + TILE_DIM - 1) / TILE_DIM, B);
    const int tile_H = (TILE_DIM - 1) * strH + (kH - 1) * dilH + (pH - 1) * dil_pH + 1,
              tile_W = (TILE_DIM - 1) * strW + (kW - 1) * dilW + (pW - 1) * dil_pW + 1;

    int c_chunk = std::min((int)C, (int)(32 * 1024 / ((tile_H * tile_W + 1) * sizeof(float))));
    c_chunk = std::max(1, c_chunk);
    const size_t shm = (size_t)(c_chunk + 1) * tile_H * tile_W * sizeof(float);
    const FastDivmod fd_tileW(tile_W);

    if (pH != pW) throw std::runtime_error("pH must be equal to pW");
#define CALL_FWD(PH, PW)                                                                                               \
    if (do_accum)                                                                                                      \
        local_autocov2d<PH, PW, true>                                                                                 \
            <<<blocks, threads, shm>>>(input.data_ptr<float>() + c_base * H * W, output.data_ptr<float>(), B, C, H, W, \
                                       H_out, W_out, kH, kW, strH, strW, padH, padW, dilH, dilW, dil_pH, dil_pW,       \
                                       tile_H, tile_W, c_curr, 1.0f / (C * kH * kW), fd_tileW);                        \
    else                                                                                                               \
        local_autocov2d<PH, PW, false>                                                                                \
            <<<blocks, threads, shm>>>(input.data_ptr<float>() + c_base * H * W, output.data_ptr<float>(), B, C, H, W, \
                                       H_out, W_out, kH, kW, strH, strW, padH, padW, dilH, dilW, dil_pH, dil_pW,       \
                                       tile_H, tile_W, c_curr, 1.0f / (C * kH * kW), fd_tileW);

    for (int c_base = 0; c_base < C; c_base += c_chunk) {
        int c_curr = std::min(c_chunk, C - c_base);
        bool do_accum = (c_base > 0);
        switch (pH) {
            case 3:
                CALL_FWD(3, 3);
                break;
            case 5:
                CALL_FWD(5, 5);
                break;
            case 7:
                CALL_FWD(7, 7);
                break;
            default:
                throw std::runtime_error("Unsupported patch size (pH)");
        }
    }
#undef CALL_FWD
    return output;
}
