from torch.autograd import Function
import torch
import torch.nn as nn

try:
    import local_autocov_backend as backend
except ImportError:
    backend = None


class _local_autocov(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, patch_size, stride, padding, dilation, dilation_patch):
        if backend is None:
            raise ImportError("local_autocov_backend not found. Please run 'pip install . --no-build-isolation' in local_autocov directory.")
        
        if input.dim() == 4:                
            if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
            if isinstance(patch_size, int): patch_size = (patch_size, patch_size)
            if isinstance(stride, int): stride = (stride, stride)
            if isinstance(padding, int): padding = (padding, padding)
            if isinstance(dilation, int): dilation = (dilation, dilation)
            if isinstance(dilation_patch, int): dilation_patch = (dilation_patch, dilation_patch)
        elif input.dim() == 5:
            if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
            if isinstance(patch_size, int): patch_size = (patch_size, patch_size, patch_size)
            if isinstance(stride, int): stride = (stride, stride, stride)
            if isinstance(padding, int): padding = (padding, padding, padding)
            if isinstance(dilation, int): dilation = (dilation, dilation, dilation)
            if isinstance(dilation_patch, int): dilation_patch = (dilation_patch, dilation_patch, dilation_patch)
        else:
            raise ValueError("Input must be 4D or 5D tensor.")
            
        ctx.save_for_backward(input)
        ctx.params = (kernel_size, patch_size, stride, padding, dilation, dilation_patch)

        output = backend.autocov_2d(
            input,
            kernel_size[0],
            kernel_size[1],
            patch_size[0],
            patch_size[1],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            dilation_patch[0],
            dilation_patch[1],
        )
        return output