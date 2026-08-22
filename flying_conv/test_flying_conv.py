import torch
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm

# Add parent dir to path to import original python code
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curr_dir))

# Compile and load CUDA extension using JIT
import torch.utils.cpp_extension
import time

print("Compiling CUDA extension...")
# JIT compile the backend so that flying_conv/__init__.py can import it
torch.utils.cpp_extension.load(
    name='flying_conv_backend',
    sources=[
        os.path.join(curr_dir, 'flying_conv.cpp'),
        os.path.join(curr_dir, 'FlyingConv2D.cu'),
        os.path.join(curr_dir, 'cuda_utils.cu')
    ],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)

from flying_conv import _flying_conv2d_torch
from flying_conv import _flying_conv2d

def benchmark_breakdown(name, backbone_kernel, input_tensor, weight_tensor, bias_tensor, out_cuda, kernel_size, stride, padding, dilation, scale, groups, iters=1000):
    print(f"\n--- {name} Kernels Breakdown ---")
    
    # Forward
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    
    t_fwd = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in tqdm(range(iters), desc=f"{name} Forward"):
        start_event.record()
        out = backbone_kernel.forward2d(
            input_tensor, weight_tensor, bias_tensor, kernel_size[0], kernel_size[1],
            stride[0], stride[1], padding[0], padding[1], 
            dilation[0], dilation[1], scale[0], scale[1], groups
        )
        end_event.record()
        torch.cuda.synchronize()
        t_fwd += start_event.elapsed_time(end_event)
    
    peak_mem = torch.cuda.max_memory_allocated()
    mem_fwd = (peak_mem - base_mem) / (1024 ** 2)
    print(f" - Forward: {t_fwd/iters:.4f} ms, Peak Mem: {mem_fwd:.2f} MB")

    # Backward Input
    grad_output = torch.ones_like(out_cuda)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    
    t_bwd_inp = 0
    for _ in tqdm(range(iters), desc=f"{name} Backward Input"):
        start_event.record()
        grad_input = backbone_kernel.backward_input2d(
            grad_output, input_tensor, weight_tensor, kernel_size[0], kernel_size[1],
            stride[0], stride[1], padding[0], padding[1], 
            dilation[0], dilation[1], scale[0], scale[1], groups
        )
        end_event.record()
        torch.cuda.synchronize()
        t_bwd_inp += start_event.elapsed_time(end_event)
        
    peak_mem = torch.cuda.max_memory_allocated()
    mem_bwd_inp = (peak_mem - base_mem) / (1024 ** 2)
    print(f" - Backward Input: {t_bwd_inp/iters:.4f} ms, Peak Mem: {mem_bwd_inp:.2f} MB")

    # Backward Weight
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    
    t_bwd_weight = 0
    for _ in tqdm(range(iters), desc=f"{name} Backward Weight"):
        start_event.record()
        grad_weight = backbone_kernel.backward_weight2d(
            grad_output, input_tensor, weight_tensor, kernel_size[0], kernel_size[1],
            stride[0], stride[1], padding[0], padding[1], 
            dilation[0], dilation[1], scale[0], scale[1], groups
        )
        end_event.record()
        torch.cuda.synchronize()
        t_bwd_weight += start_event.elapsed_time(end_event)
        
    peak_mem = torch.cuda.max_memory_allocated()
    mem_bwd_weight = (peak_mem - base_mem) / (1024 ** 2)
    print(f" - Backward Weight: {t_bwd_weight/iters:.4f} ms, Peak Mem: {mem_bwd_weight:.2f} MB")

    # Backward Bias
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    
    t_bwd_bias = 0
    for _ in tqdm(range(iters), desc=f"{name} Backward Bias"):
        start_event.record()
        grad_bias = backbone_kernel.backward_bias2d(
            grad_output, bias_tensor, scale[0], scale[1], groups
        )
        end_event.record()
        torch.cuda.synchronize()
        t_bwd_bias += start_event.elapsed_time(end_event)
        
    peak_mem = torch.cuda.max_memory_allocated()
    mem_bwd_bias = (peak_mem - base_mem) / (1024 ** 2)
    print(f" - Backward Bias: {t_bwd_bias/iters:.4f} ms, Peak Mem: {mem_bwd_bias:.2f} MB")

def test_forward(accuracy=False, benchmark=True):
    B = 8
    C = 32
    H_in, W_in = 512, 512
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)  
    scale = (8, 8)
    groups = 2

    K = kernel_size[0] * kernel_size[1]

    torch.manual_seed(0)
    ## for torch
    input_0 = torch.randn(B, C, H_in, W_in, device="cuda", dtype=torch.float32, requires_grad=True)
    weight_0 = torch.randn(B, C // groups, K, H_in // scale[0], W_in // scale[1], device="cuda", dtype=torch.float32, requires_grad=True)
    bias_0 = torch.randn(B, C // groups, H_in // scale[0], W_in // scale[1], device="cuda", dtype=torch.float32, requires_grad=True)

    ## for triton
    input_1 = input_0.detach().clone().requires_grad_(True)
    weight_1 = torch.randn(B, C, H_in // scale[0], W_in // scale[1], K, device="cuda", dtype=torch.float32, requires_grad=True)
    bias_1 = torch.randn(B, C, H_in // scale[0], W_in // scale[1], device="cuda", dtype=torch.float32, requires_grad=True)

    ## for cuda
    input_2 = input_0.detach().clone().requires_grad_(True)
    weight_2 = torch.randn(B, C // groups, K, H_in // scale[0], W_in // scale[1], device="cuda", dtype=torch.float32, requires_grad=True)
    bias_2 = torch.randn(B, C // groups, H_in // scale[0], W_in // scale[1], device="cuda", dtype=torch.float32, requires_grad=True)

    with torch.no_grad():
        weight_0.copy_(weight_2)
        bias_0.copy_(bias_2)
        weight_1.copy_(weight_2.permute(0, 1, 3, 4, 2).repeat_interleave(groups, dim=1).contiguous())
        bias_1.copy_(bias_2.repeat_interleave(groups, dim=1))

    if accuracy:
        # # Triton
        # out_triton = _triton_flying_conv2d.apply(input_1, weight_1, bias_1, kernel_size, stride, padding, dilation, scale)
        # out_triton.mean().backward()

        # CUDA
        out_cuda = _flying_conv2d.apply(input_2, weight_2, bias_2, kernel_size, stride, padding, dilation, scale, groups)
        out_cuda.mean().backward()

        # Torch
        out_torch = _flying_conv2d_torch(input_0, weight_0, bias_0, kernel_size, stride, padding, dilation, scale, groups)
        out_torch.mean().backward()
        
        diff_1 = (out_torch - out_cuda).abs()
        # diff_2 = (out_torch - out_triton).abs()
        w0_grad_grouped = weight_0.grad
        w1_grad_grouped = None

        print("Forward diff max: Torch vs CUDA {}".format(diff_1.max().item()))
        print("Backward input diff max: Torch vs CUDA {}".format((input_0.grad - input_2.grad).abs().max().item()))
        print("Backward weight diff max: Torch vs CUDA {}".format((w0_grad_grouped - weight_2.grad).abs().max().item()))
        
        b0_grad_grouped = bias_0.grad
        print("Backward bias diff max: Torch vs CUDA {}".format((b0_grad_grouped - bias_2.grad).abs().max().item()))

    if benchmark:
        # Benchmark
        print("\n--- Benchmarking ---")
        iters = 1000
        # --- Kernels Breakdown ---
        # pyrefly: ignore [missing-import]
        import flying_conv_backend
        # import flying_conv_triton.flying_conv as flying_triton
        
        # triton_backend = type('obj', (object,), {
        #     'forward': lambda *args: flying_triton.flying_conv2d_fwd(*args[:-1]),
        #     'backward_input': lambda *args: flying_triton.flying_conv2d_input_bwd(*args[:-1]),
        #     'backward_weight': lambda *args: flying_triton.flying_conv2d_weight_bwd(*args[:-1]),
        #     'backward_bias': lambda *args: flying_triton.flying_conv2d_bias_bwd(*args[:-1])
        # })

        out_cuda = _flying_conv2d.apply(input_2, weight_2, bias_2, kernel_size, stride, padding, dilation, scale, groups)
        # out_cuda.mean().backward()
        benchmark_breakdown("CUDA", flying_conv_backend, input_2, weight_2, bias_2, out_cuda, kernel_size, stride, padding, dilation, scale, groups, iters)
        # benchmark_breakdown("Triton", triton_backend, input_1, weight_1, bias_1, out_cuda, kernel_size, stride, padding, dilation, scale, groups, iters)

if __name__ == "__main__":
    test_forward(accuracy=True, benchmark=True)
