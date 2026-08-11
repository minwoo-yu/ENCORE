#include <torch/extension.h>

// CUDA forward declarations (in cuda_impl namespace to avoid name collisions)
namespace cuda_ver {
torch::Tensor flying_conv2d_fwd(torch::Tensor input, torch::Tensor weight, at::optional<torch::Tensor> bias,
                                int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h,
                                int dil_w, int scale_h, int scale_w, int groups);

torch::Tensor flying_conv2d_input_bwd(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                      int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
                                      int dil_h, int dil_w, int scale_h, int scale_w, int groups);

torch::Tensor flying_conv2d_weight_bwd(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                       int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
                                       int dil_h, int dil_w, int scale_h, int scale_w, int groups);

torch::Tensor flying_conv2d_bias_bwd(torch::Tensor grad_output, torch::Tensor bias, int scale_h, int scale_w,
                                     int groups);
}  // namespace cuda_ver

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor flying_conv2d_fwd(torch::Tensor input, torch::Tensor weight, at::optional<torch::Tensor> bias,
                                int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h,
                                int dil_w, int scale_h, int scale_w, int groups) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    return cuda_ver::flying_conv2d_fwd(input, weight, bias, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h,
                                       dil_w, scale_h, scale_w, groups);
}

torch::Tensor flying_conv2d_input_bwd(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                      int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
                                      int dil_h, int dil_w, int scale_h, int scale_w, int groups) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return cuda_ver::flying_conv2d_input_bwd(grad_output, input, weight, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                                             pad_w, dil_h, dil_w, scale_h, scale_w, groups);
}

torch::Tensor flying_conv2d_weight_bwd(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                       int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
                                       int dil_h, int dil_w, int scale_h, int scale_w, int groups) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return cuda_ver::flying_conv2d_weight_bwd(grad_output, input, weight, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                                              pad_w, dil_h, dil_w, scale_h, scale_w, groups);
}

torch::Tensor flying_conv2d_bias_bwd(torch::Tensor grad_output, torch::Tensor bias, int scale_h, int scale_w,
                                     int groups) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(bias);
    return cuda_ver::flying_conv2d_bias_bwd(grad_output, bias, scale_h, scale_w, groups);
}

PYBIND11_MODULE(flying_conv_backend, m) {
    m.def("forward2d", &flying_conv2d_fwd, "FlyingConv2D forward (CUDA)");
    m.def("backward_input2d", &flying_conv2d_input_bwd, "FlyingConv2D backward input (CUDA)");
    m.def("backward_weight2d", &flying_conv2d_weight_bwd, "FlyingConv2D backward weight (CUDA)");
    m.def("backward_bias2d", &flying_conv2d_bias_bwd, "FlyingConv2D backward bias (CUDA)");
}
