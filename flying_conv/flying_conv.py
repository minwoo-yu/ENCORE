import torch
import time
from tqdm import tqdm
import torch.nn.functional as F

def _torch_flying_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, scale=4):
    B, C, H, W = input.shape
    _, _, K, H_down, W_down = weight.shape
    weight = weight
    input_unfold = F.unfold(input, kernel_size=3, dilation=dilation, padding=padding, stride=stride).view(B, C, K, H, W)
    weight_up = F.interpolate(weight.view(B, C * K, H_down, W_down), scale_factor=scale, mode="bilinear").view(B, C, K, H, W)
    output = (input_unfold * weight_up).sum(dim=2)
    if bias is not None:
        bias_up = F.interpolate(bias, scale_factor=scale, mode="bilinear")
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