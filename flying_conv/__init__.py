from torch.autograd import Function
import torch
import torch.nn.functional as F
try:
    import flying_conv_backend as backend
except ImportError:
    backend = None

def _flying_conv2d_torch(input, weight, bias=None, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), scale=4, groups=1):
    B, C, H, W = input.shape
    C_weight = C // groups
    _, _, K, H_down, W_down = weight.shape
    scale_h, scale_w = scale if isinstance(scale, (list, tuple)) else (scale, scale)
    H_out, W_out = H_down * scale_h, W_down * scale_w
    
    input_unfold = F.unfold(input, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride).view(B, C, K, H_out, W_out)
    weight_up = F.interpolate(weight.view(B, C_weight * K, H_down, W_down), scale_factor=scale, mode="bilinear").view(B, C_weight, K, H_out, W_out)
    
    if groups > 1:
        weight_up = weight_up.repeat_interleave(groups, dim=1)
        
    output = (input_unfold * weight_up).sum(dim=2)
    
    if bias is not None:
        bias_up = F.interpolate(bias, scale_factor=scale, mode="bilinear")
        if groups > 1:
            bias_up = bias_up.repeat_interleave(groups, dim=1)
        output = output + bias_up
        
    return output


def _torch_flying_conv3d(input, weight, bias=None, stride=1, padding=1, dilation=1, scale=4):
    B, C, D, H, W = input.shape
    _, _, D_down, H_down, W_down, K = weight.shape

    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    k_size = int(round(K ** (1.0 / 3.0)))
    k_d = k_h = k_w = k_size

    weight = weight.permute(0, 1, 5, 2, 3, 4).contiguous() # (B, C, K, D_down, H_down, W_down)

    input_pad = F.pad(input, (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]))
    eff_k_d = (k_d - 1) * dilation[0] + 1
    eff_k_h = (k_h - 1) * dilation[1] + 1
    eff_k_w = (k_w - 1) * dilation[2] + 1

    input_unfold = input_pad.unfold(2, eff_k_d, stride[0]).unfold(3, eff_k_h, stride[1]).unfold(4, eff_k_w, stride[2])
    input_unfold = input_unfold[..., :: dilation[0], :: dilation[1], :: dilation[2]]

    B, C, D_out, H_out, W_out, _, _, _ = input_unfold.shape
    input_unfold = input_unfold.contiguous().view(B, C, D_out, H_out, W_out, K)
    input_unfold = input_unfold.permute(0, 1, 5, 2, 3, 4)

    weight_up = F.interpolate(weight.view(B, C * K, D_down, H_down, W_down), scale_factor=scale, mode="trilinear").view(B, C, K, D_out, H_out, W_out)

    output = (input_unfold * weight_up).sum(dim=2)
    if bias is not None:
        bias_up = F.interpolate(bias, scale_factor=scale, mode="trilinear")
        output = output + bias_up
    return output


class _flying_conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, kernel_size, stride, padding, dilation, scale, groups):
        if backend is None:
            raise ImportError("flying_conv_backend not found. Please run 'pip install . --no-build-isolation' in flying_conv directory.")
        
        if input.dim() == 4:
            if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int): stride = (stride, stride)
            if isinstance(padding, int): padding = (padding, padding)
            if isinstance(dilation, int): dilation = (dilation, dilation)
            if isinstance(scale, int): scale = (scale, scale)
        elif input.dim() == 5:
            if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
            if isinstance(stride, int): stride = (stride, stride, stride)
            if isinstance(padding, int): padding = (padding, padding, padding)
            if isinstance(dilation, int): dilation = (dilation, dilation, dilation)
            if isinstance(scale, int): scale = (scale, scale, scale)
        else:
            raise ValueError("Input must be 4D or 5D tensor.")

        ctx.save_for_backward(input, weight, bias)
        ctx.params = (kernel_size, stride, padding, dilation, scale, groups)

        output = backend.forward2d(
            input,
            weight,
            bias,
            kernel_size[0],
            kernel_size[1],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            scale[0],
            scale[1],
            groups,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output): # grad_output: [B, C, H_out, W_out]
        if backend is None:
             raise ImportError("flying_conv_backend not found.")
        input, weight, bias = ctx.saved_tensors
        kernel_size, stride, padding, dilation, scale, groups = ctx.params

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = backend.backward_input2d(
                grad_output,
                input,
                weight,
                kernel_size[0],
                kernel_size[1],
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                scale[0],
                scale[1],
                groups,
            )

        if ctx.needs_input_grad[1]:
            grad_weight = backend.backward_weight2d(
                grad_output,
                input,
                weight, #weight.permute(0, 4, 1, 2, 3).contiguous() [B, K, C, H_weight, W_weight]
                kernel_size[0],
                kernel_size[1],
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                scale[0],
                scale[1],
                groups,
            )

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = backend.backward_bias2d(grad_output, bias, scale[0], scale[1], groups)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None