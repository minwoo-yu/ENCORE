import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from tqdm import tqdm

# pyrefly: ignore [missing-import]
# try:
#     import spatial_correlation_sampler
# except ImportError:
#     spatial_correlation_sampler = None

# Add parent dir to path to import local package
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curr_dir))

print("Compiling CUDA extension...")
torch.utils.cpp_extension.load(
    name='local_autocov_backend',
    sources=[
        os.path.join(curr_dir, 'local_autocov.cpp'),
        os.path.join(curr_dir, 'LocalAutocov2D.cu')
    ],
    extra_cuda_cflags=['-O3', '-use_fast_math'],
    verbose=True
)

from local_autocov import _local_autocov

def _local_autocov_torch(input, kernel_size, patch_size, stride=1, padding=0, dilation=1, dilation_patch=1):
    def _pair(x):
        if isinstance(x, (list, tuple)):
            return x
        return (x, x)

    kH, kW = _pair(kernel_size)
    pH, pW = _pair(patch_size)
    strH, strW = _pair(stride)
    padH, padW = _pair(padding)
    dilH, dilW = _pair(dilation)
    dil_pH, dil_pW = _pair(dilation_patch)

    B, C, H, W = input.shape
    P = pH * pW

    half_pH = (pH - 1) // 2 * dil_pH
    half_pW = (pW - 1) // 2 * dil_pW

    # Vectorized unfold to get all patch pixels simultaneously
    unfolded = F.unfold(input, kernel_size=(pH, pW), padding=(half_pH, half_pW), dilation=(dil_pH, dil_pW))
    unfolded = unfolded.view(B, C, P, H, W)

    # Product I(x) * I(x + offset) and sum over channel dimension
    elementwise_mul = input.unsqueeze(2) * unfolded
    summed_channels = elementwise_mul.sum(dim=1) # (B, P, H, W)

    # Run grouped convolution to perform spatial window sum for all patches in parallel
    weight = torch.ones((P, 1, kH, kW), dtype=input.dtype, device=input.device)
    summed_map = F.conv2d(
        summed_channels,
        weight,
        stride=(strH, strW),
        padding=(padH, padW),
        dilation=(dilH, dilW),
        groups=P
    ) # (B, P, H_out, W_out)

    output = summed_map / (C * kH * kW)
    return output

def benchmark_model(name, model_fn, input_tensor, iters=500, warmup=100):
    print(f"\n--- Benchmarking {name} ---")
    
    # Define events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 1. Warmup
    print(f"Warmup ({warmup} iters)...")
    for _ in range(warmup):
        output = model_fn(input_tensor)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 2. Forward Benchmark
    print(f"Forward ({iters} iters)...")
    start_event.record()
    for _ in tqdm(range(iters)):
        output = model_fn(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    fwd_time = start_event.elapsed_time(end_event) / iters
    fwd_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n{name} Results:")
    print(f"  Forward:  {fwd_time:.3f} ms/iter, Peak Mem: {fwd_mem:.2f} MB")
    
    return fwd_time, fwd_mem


def test_forward(accuracy=False, benchmark=True):
    B, C = 8, 1
    H, W = 512, 512
    kernel_size = (11, 11)
    patch_size = (5, 5)
    stride, padding = (1, 1), (5, 5)
    dilation, dilation_patch = (1, 1), (1, 1)

    # conventional_sampler = spatial_correlation_sampler.SpatialCorrelationSampler(
    #     kernel_size=kernel_size, patch_size=patch_size, stride=stride,
    #     padding=padding, dilation=dilation, dilation_patch=dilation_patch,
    # ).cuda()

    torch.manual_seed(0)

    input_0 = torch.randn((B, C, H, W)).cuda().requires_grad_(True)
    input_1 = input_0.detach().clone().requires_grad_(True)
    input_2 = input_0.detach().clone().requires_grad_(True)

    if accuracy:
        # Conventional
        # output_0 = conventional_sampler(input_0, input_0) / C / (kernel_size[0] * kernel_size[1])
        
        # CUDA
        output_1 = _local_autocov.apply(input_1, kernel_size, patch_size, stride, padding, dilation, dilation_patch)
        
        # Torch
        output_2 = _local_autocov_torch(input_2, kernel_size, patch_size, stride, padding, dilation, dilation_patch) / C / (kernel_size[0] * kernel_size[1])

        diff_cuda_fwd = (output_2 - output_1.view(B, -1, H, W)).abs().max().item()
        # diff_cuda_fwd = (output_1 - output_0.view(B, -1, H, W)).abs().max().item()
        
        print(f"Forward diff max:  CUDA: {diff_cuda_fwd:.2e}")

    if benchmark:
        warmup = 20
        iters = 1000
        
        # Define functions for benchmarking
        def run_conventional(x):
            return _local_autocov_torch(x, kernel_size, patch_size, stride, padding, dilation, dilation_patch) / C / (kernel_size[0] * kernel_size[1])

        def run_cuda(x):
            return _local_autocov.apply(x, kernel_size, patch_size, stride, padding, dilation, dilation_patch)

        # Start benchmarks
        print("\n" + "="*50)
        print(f"Benchmark Config: B={B}, C={C}, H={H}, W={W}, iters={iters}")
        
        benchmark_model("CUDA", run_cuda, input_2, iters, warmup)


if __name__ == "__main__":
    test_forward(accuracy=True, benchmark=True)
