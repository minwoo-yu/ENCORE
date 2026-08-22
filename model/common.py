import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fleet
from einops import rearrange
from flying_conv import _flying_conv2d
from local_autocov import _local_autocov

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


def _local_autocov_torch(input, kernel_size, patch_size, stride=1, padding=0, dilation=1, dilation_patch=1):
    def _p(x):
        return x if isinstance(x, (list, tuple)) else (x, x)

    pH, pW = _p(patch_size)
    dpH, dpW = _p(dilation_patch)
    
    B, C, H, W = input.shape
    P = pH * pW
    pad_patch = ((pH - 1) // 2 * dpH, (pW - 1) // 2 * dpW)

    # Vectorized unfold to get all patch pixels simultaneously
    unfolded = F.unfold(input, kernel_size=patch_size, padding=pad_patch, dilation=dilation_patch)
    unfolded = unfolded.view(B, C, P, H, W)

    # Product I(x) * I(x + offset) and sum over channel dimension
    summed_channels = (input.unsqueeze(2) * unfolded).sum(dim=1) # (B, P, H, W)

    # Run grouped convolution to perform spatial window sum for all patches in parallel
    weight = torch.ones((P, 1) + _p(kernel_size), dtype=input.dtype, device=input.device)
    summed_map = F.conv2d(summed_channels, weight, stride=stride, padding=padding, dilation=dilation, groups=P)

    return summed_map / (C * _p(kernel_size)[0] * _p(kernel_size)[1])


class FLEETRecon(nn.Module):
    def __init__(self, config, device, mode="VD"):
        super(FLEETRecon, self).__init__()
        self.config = config
        self.device = device
        self.mode = mode
        self.projector = fleet.Projector()
        self.geometry_setting(self.config["dataset"]["batch_size"] * (1 + self.config["N_augment"]["N_ch"]))
        phis = np.linspace(0, 2 * np.pi, self.config["dataset"]["geometry"]["num_view"], endpoint=False, dtype="float32")
        self.projector.set_angles(torch.from_numpy(phis - np.pi / 2).cuda())
        self.projector.set_projector(self.mode)
        self.last_voxelWidth = 0
        self.last_num_rows = 0

    def geometry_setting(self, num_rows):
        self.projector.set_fanbeam(
            self.config["dataset"]["geometry"]["num_view"],
            num_rows,
            self.config["dataset"]["geometry"]["num_det"],
            1,
            self.config["dataset"]["geometry"]["det_interval"],
            0,
            0.5 * (self.config["dataset"]["geometry"]["num_det"] - 1),
            self.config["dataset"]["geometry"]["sod"],
            self.config["dataset"]["geometry"]["sdd"],
            "flat",
        )

    def volume_setting(self, metadata, num_rows):
        if self.last_voxelWidth == metadata[0, 3] and self.last_num_rows == num_rows:
            return
        self.geometry_setting(num_rows)
        self.projector.set_volume(
            num_rows,
            512,
            512,
            1,
            metadata[0, 3],
        )
        self.last_voxelWidth, self.last_num_rows = metadata[0, 3], num_rows

    def fbp(self, projection, metadata):
        projection = rearrange(projection, "b v r c -> v (b r) c").contiguous()
        self.volume_setting(metadata, projection.shape[1])
        recon = self.projector.fbp(projection)
        return recon

    def flops(self, num_view, num_rows, num_dets, Nx=512, Ny=512):
        # 1. Fan-beam cosine weight
        flops_pre = num_view * num_dets * num_rows

        # 2. Filtering (FFT-based Ram-Lak filter)
        N_fft = 2 ** math.ceil(math.log2(num_dets * 2))
        # 1D FFT complexity
        flops_fft = 5 * N_fft * math.log2(N_fft)
        # FFT + Filter Multiplication (complex) + IFFT
        flops_filter_1d = 2 * flops_fft + 6 * N_fft
        flops_filtering = num_view * num_rows * flops_filter_1d

        # 3. Back-projection
        if self.mode == "VD":
            # Voxel-driven ()
            ops_per_voxel = 28  # 15 (backprojection weight) + 13 (interpolation)
            flops_bp = Nx * Ny * num_rows * num_view * ops_per_voxel
        elif self.mode == "SF":
            # Separable Footprint
            ops_per_voxel = 58  # 32 (backprojection weight) + 13 * 2 (interpolation)
            flops_bp = Nx * Ny * num_rows * num_view * ops_per_voxel
        else:
            flops_bp = 0

        return flops_pre + flops_filtering + flops_bp


class NoiseRecon(nn.Module):
    def __init__(self, config):
        super(NoiseRecon, self).__init__()
        self.config = config
        self.mode = config["dataset"]["mode"]
        self.N_in = float(config["dataset"]["N_in"])
        self.sigma_e = config["dataset"]["sigma_e"]
        self.generation = config["dataset"]["generation"]
        self.device = torch.device(0)
        if isinstance(config["dataset"]["dose_level"], list):
            self.dose_level = torch.Tensor(config["dataset"]["dose_level"]).to(self.device)
        else:
            self.dose_level = torch.Tensor([config["dataset"]["dose_level"]]).to(self.device)
        self.fleet = FLEETRecon(config, self.device)

    def generate_noise(self, base_proj, dose_level, shape, mode="poisson"):
        batch, view, row, col = shape
        if mode == "poisson":
            q_noise = torch.poisson(base_proj) - base_proj
            e_noise = torch.randn(shape, device=self.device).mul_(self.sigma_e)
        elif mode in ["gaussian", "cornish"]:
            noise = torch.randn((batch, view, 2 * row, col), device=self.device)
            e_noise = noise[:, :, row:, :].mul_(self.sigma_e)
            sqrt_base = torch.sqrt(base_proj)
            if mode == "gaussian":
                q_noise = noise[:, :, :row, :].mul_(sqrt_base)
            else:  # cornish
                N = noise[:, :, :row, :]
                if dose_level is None:  # for noise augmentation
                    beta = 1.0 / (6.0 * sqrt_base + 1e-6)
                else:
                    dose_factor = torch.sqrt(dose_level * (1.0 - dose_level))
                    sigma_add = dose_factor * sqrt_base
                    beta = ((1.0 + dose_level) / (6.0 * sigma_add.clamp(min=1e-6))).clamp(max=math.sqrt(0.5))
                alpha = torch.sqrt((1.0 - 2.0 * beta**2).clamp_(min=0))
                W = alpha * N + beta * (N**2 - 1)
                q_noise = W.mul_(sqrt_base)
        else:
            ValueError("Invalid mode")

        # if self.kernel_qe is not None:
        #     combined_in = torch.stack([q_input, e_input], dim=2).view(batch * view, 2, row, col)
        #     out = F.conv2d(combined_in, self.kernel_qe, groups=2, padding=(0, 2))
        #     out = out.view(batch, view, 2, row, col)
        #     q_noise, e_noise = out[:, :, 0, :, :], out[:, :, 1, :, :]
        # else:
        #     q_noise, e_noise = q_input, e_input
        return q_noise, e_noise

    def forward(self, metadata, inp_proj, ld_proj=None, N_ch=0, target_dose=None):
        # Check if metadata values vary across the batch
        if metadata.shape[0] > 1 and torch.any(metadata[1:] != metadata[0]):
            print("Warning: Metadata values vary across the batch. Using the first value for all samples.")
        batch, _, view, col = inp_proj.shape

        u_water = 0.0192867 if int(metadata[0, 0]) == 120 else 0.0205888
        inp_proj = rearrange(inp_proj, "b r v c -> b v r c")  # [batch, view, row, col]
        
        with torch.no_grad():
            if ld_proj is None:
                dose_level = self.dose_level[torch.randint(0, len(self.dose_level), (batch,))].view(batch, 1, 1, 1)
                if self.mode == "N2C":
                    scaled_proj = inp_proj * dose_level
                    q_noise, e_noise = self.generate_noise(scaled_proj, dose_level, scaled_proj.shape, mode="poisson")
                    ld_proj = (scaled_proj + q_noise + e_noise).clamp_(min=1)
                else:
                    a = torch.sqrt(1.0 / dose_level - 1.0)
                    b = torch.sqrt(1.0 / dose_level + 1.0)
                    q_noise, e_noise = self.generate_noise(inp_proj, dose_level, inp_proj.shape, mode=self.generation)
                    ld_proj = (dose_level * (inp_proj + a * q_noise + a * b * e_noise)).clamp_(min=1)
            else:
                dose_level = self.dose_level.view(-1, 1, 1, 1)
                ld_proj = rearrange(ld_proj, "b r v c -> b v r c")

            if N_ch != 0:
                if target_dose is not None:
                    ratio = dose_level / target_dose
                    q_lower, e_lower = self.generate_noise(ld_proj * (1 - ratio), None, shape=(batch, view, N_ch, col), mode=self.generation)
                    e_lower *= torch.sqrt(1 - ratio**2)
                else:
                    q_lower, e_lower = self.generate_noise(ld_proj, None, shape=(batch, view, N_ch, col), mode=self.generation)
                ld_proj = torch.cat((ld_proj / self.N_in / dose_level, (ld_proj + q_lower + e_lower).clamp_(min=1) / ld_proj), 2)
            else:
                ld_proj = ld_proj / self.N_in / dose_level

            ## reconstruction
            if self.training and self.mode == "N2N":  ## Noise2Noise, input is ND data
                in_proj = (inp_proj - q_noise / a - e_noise / (a * b)).clamp_(min=1)
                data_proj = torch.cat((-torch.log(ld_proj), -torch.log(in_proj / self.N_in)), 2)
                recon = self.fleet.fbp(data_proj, metadata)
            elif self.training and self.mode == "N2F":  ## Noise2Full, input is ND data
                data_proj = torch.cat((-torch.log(ld_proj), -torch.log(inp_proj / self.N_in)), 2)
                recon = self.fleet.fbp(data_proj, metadata)
            else:  # Noise2Clean or validation / test mode, input is Clean data
                data_proj = torch.cat((-torch.log(ld_proj), -torch.log(inp_proj / self.N_in)), 2)
                recon = self.fleet.fbp(data_proj, metadata)
            recon = recon.view(batch, -1, 512, 512)
        return (
            recon[:, :1] * 1000 / u_water - 1000,  # LDCT
            None if N_ch == 0 else recon[:, 1:-1] * 1000 / u_water,  # noise
            recon[:, -1:] * 1000 / u_water - 1000,  # GT
        )


class NoiseCovariance(nn.Module):
    def __init__(self, kernel_size, patch_size, stride=1, padding=0, dilation=1, dilation_patch=1):
        super(NoiseCovariance, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, noise):
        B, C, H, W = noise.shape
        with torch.no_grad():
            noise_cov = _local_autocov.apply(noise, self.kernel_size, self.patch_size, self.stride, self.padding, self.dilation, self.dilation_patch)
            # noise_cov = _local_autocov_torch(noise, self.kernel_size, self.patch_size, self.stride, self.padding, self.dilation, self.dilation_patch)
        # noise_cov = noise_cov.view(B, -1, H, W)
        return noise_cov

    def flops(self, C, H, W):
        kh, kw = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
        ph, pw = (self.patch_size, self.patch_size) if isinstance(self.patch_size, int) else self.patch_size
        sh, sw = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        padh, padw = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        dh, dw = (self.dilation, self.dilation) if isinstance(self.dilation, int) else self.dilation

        h_out = (H + 2 * padh - dh * (kh - 1) - 1) // sh + 1
        w_out = (W + 2 * padw - dw * (kw - 1) - 1) // sw + 1

        return h_out * w_out * kh * kw * ph * pw * C


def get_recon_patch(input, target, patch_coord, position, noise=None):
    B, H, W, _ = patch_coord.shape
    patch_coord = patch_coord / (512 / H) + position.flip(-1).view(B, 1, 1, 2)
    input = F.grid_sample(input, patch_coord, mode="nearest", align_corners=False)
    target = F.grid_sample(target, patch_coord, mode="nearest", align_corners=False)
    if noise is not None:
        noise = F.grid_sample(noise, patch_coord, mode="nearest", align_corners=False)
    return input, noise, target


class Predictor(nn.Module):
    def __init__(self, channel, arch="pconv"):
        super(Predictor, self).__init__()
        self.arch = arch
        self.channel = channel
        self.expand, self.ratio = 1, 1 / 4

        if arch == "pconv":
            self.split_ch = int(channel * self.ratio)
            self.pconv = nn.Conv2d(self.split_ch, self.split_ch, kernel_size=3, padding=1, bias=False)
            self.pwconv1 = nn.Conv2d(channel, channel * self.expand, kernel_size=1, bias=False)
            self.norm = nn.GroupNorm(num_groups=1, num_channels=channel * self.expand)
            self.pwconv2 = nn.Conv2d(channel * self.expand, channel, kernel_size=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm = nn.GroupNorm(num_groups=1, num_channels=channel)
            self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, inp):
        B, C, H, W = inp.shape
        if self.arch == "pconv":
            x1, x2 = torch.split(inp, [self.split_ch, C - self.split_ch], dim=1)
            x1 = self.pconv(x1)
            x = torch.cat([x1, x2], dim=1)
            x = self.pwconv1(x)
            x = self.norm(x)
            x = self.act(x)
            x = self.pwconv2(x)
        else:
            x = self.conv1(inp)
            x = self.norm(x)
            x = self.act(x)
            x = self.conv2(x)
        return inp + x

    def flops(self, H, W):
        if self.arch == "pconv":
            flops = H * W * self.split_ch * self.split_ch * 9
            flops += H * W * self.channel * (self.channel * self.expand)
            flops += H * W * (self.channel * self.expand) * self.channel
        else:
            flops = H * W * self.channel * self.channel * 9
            flops += H * W * self.channel * self.channel * 9
        return flops


class FlyingConv(nn.Module):
    def __init__(self, in_channel, out_channel, n_ch=0, bias=False, pooling=True, num_blocks=4, groups=1, scale=4, dropout=0.0):
        super(FlyingConv, self).__init__()
        self.kernel_size = (3, 3)
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.scale = (scale * 2, scale * 2) if pooling and num_blocks > 1 else (scale, scale)
        self.groups = groups

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_ch = n_ch
        self.bias = bias

        self.pool = nn.AvgPool2d(scale, scale)
        self.pooling = pooling
        self.num_blocks = num_blocks
        self.partitions = in_channel // groups

        self.dropout = nn.Dropout2d(dropout)
        self.predictor = []
        self.predictor.append(nn.Conv2d(in_channel + n_ch, in_channel, kernel_size=1, padding=0))
        for i in range(self.num_blocks):
            self.predictor.append(Predictor(in_channel, arch="pconv"))
            if i == self.num_blocks // 2 - 1 and pooling:
                self.predictor.append(nn.AvgPool2d(2, 2))
        self.predictor = nn.Sequential(*self.predictor)

        self.expander = nn.Conv2d(
            in_channel,
            self.partitions * (self.kernel_size[0] * self.kernel_size[1] + self.bias),
            kernel_size=1,
            padding=0,
        )
        target_std = math.sqrt(1.0 / (self.kernel_size[0] * self.kernel_size[1]))
        nn.init.normal_(self.expander.bias, 0, target_std)
        nn.init.normal_(self.expander.weight, 0, target_std * 0.1)

        self.pwconv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.clipper = nn.Hardtanh(min_val=-2, max_val=2)

    def forward(self, x, noise=None):
        B, C, H, W = x.shape
        x_pool = self.pool(x)
        if self.n_ch > 0:
            noise = self.dropout(noise)
            feat = torch.cat((x_pool, noise), dim=1)
        else:
            feat = x_pool
        feat = self.predictor(feat)

        weights = self.expander(feat).view(B, self.partitions, -1, H // self.scale[0], W // self.scale[1])
        weights = self.clipper(weights)
        if not self.bias:
            weight, bias = weights.contiguous(), None
        else:
            weight, bias = weights[..., :-1].contiguous(), weights[..., -1].contiguous()
        self.weight = weight
        weight = self.clipper(weight)
        out = _flying_conv2d.apply(x, weight, bias, self.kernel_size, self.stride, self.padding, self.dilation, self.scale, self.groups)
        # out = _flying_conv2d_torch(x, weight, bias, self.kernel_size, self.stride, self.padding, self.dilation, self.scale, self.groups)
        out = self.pwconv(out)
        return out

    def flops(self, H, W):
        flops = 0
        scale_initial = self.pool.kernel_size
        if isinstance(scale_initial, int):
            scale_initial = (scale_initial, scale_initial)
        H_p, W_p = H // scale_initial[0], W // scale_initial[1]

        for layer in self.predictor:
            if isinstance(layer, nn.Conv2d):
                flops += H_p * W_p * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
            elif hasattr(layer, "flops"):
                flops += layer.flops(H_p, W_p)
            elif isinstance(layer, nn.AvgPool2d):
                H_p //= 2
                W_p //= 2

        # expander
        flops += H_p * W_p * self.in_channel * self.partitions * (self.kernel_size[0] * self.kernel_size[1] + self.bias)
        # flying conv
        flops += H * W * self.in_channel * (self.kernel_size[0] * self.kernel_size[1] + self.bias) * (1 + 4 / self.groups)
        # pwconv
        flops += H * W * self.out_channel * self.in_channel
        return flops
