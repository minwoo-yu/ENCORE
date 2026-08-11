#include <torch/extension.h>

torch::Tensor local_autocov2d(const torch::Tensor& input, int kH, int kW, int pH, int pW, int dH, int dW, int padH,
                               int padW, int dilH, int dilW, int dil_pH, int dil_pW);

PYBIND11_MODULE(local_autocov_backend, m) {
    m.def("autocov_2d", &local_autocov2d, "Local Autocovariance for 2D input (CUDA)");
}
