import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from model import common


@torch.jit.script
def _norm_fused(n: torch.Tensor) -> torch.Tensor:
    return torch.sign(n) * torch.log1p(torch.abs(n))


def make_model(config):  # S: Small, B: Base
    if config["model"]["level"] == "S":
        dim = 16
    elif config["model"]["level"] == "B":
        dim = 32
    print("model loaded!: UNet-" + config["model"]["level"] + "+" + config["model"]["backbone"])
    return UNet(
        noise_spec=config["N_augment"],
        backbone=config["model"]["backbone"],
        dim=dim,
        num_blocks=config["model"]["num_blocks"],
        pooling=config["model"]["pooling"],
        groups=config["model"]["groups"],
        scale=config["model"]["scale"],
    )


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        noise_ch=0,
        strides=1,
        backbone="conv",
        num_blocks=4,
        pooling=True,
        groups=1,
        scale=4,
        dropout=0.0,
    ):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.noise_ch = noise_ch
        self.backbone = backbone
        self.pooling = pooling
        self.groups = groups

        self.block = []
        if self.backbone == "conv":
            self.block.append(nn.Conv2d(in_channel + noise_ch, out_channel, kernel_size=3, stride=strides, padding=1))
            self.block.append(nn.LeakyReLU(inplace=True))
            self.block.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1))
        else:
            self.block.append(
                common.FlyingConv(
                    in_channel,
                    out_channel,
                    n_ch=noise_ch,
                    num_blocks=num_blocks,
                    pooling=pooling,
                    groups=groups,
                    scale=scale,
                    dropout=dropout,
                )
            )
            self.block.append(nn.LeakyReLU(inplace=True))
            self.block.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1))

        self.block.append(nn.LeakyReLU(inplace=True))
        self.block = nn.ModuleList(self.block)
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x, noise=None):
        out1 = x
        if noise is not None and self.noise_ch > 0 and self.backbone == "conv":
            out1 = torch.cat((out1, noise), dim=1)

        for layer in self.block:
            if isinstance(layer, common.FlyingConv):
                out1 = layer(out1, noise=noise)
            else:
                out1 = layer(out1)

        return out1 + self.conv11(x)

    def flops(self, H, W):
        flops = 0
        for layer in self.block:
            if isinstance(layer, nn.Conv2d):
                flops += H * W * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
            elif hasattr(layer, "flops"):
                flops += layer.flops(H, W)

        flops += (
            H
            * W
            * self.conv11.in_channels
            * self.conv11.out_channels
            * self.conv11.kernel_size[0]
            * self.conv11.kernel_size[1]
        )
        return flops


class UNet(nn.Module):
    def __init__(
        self, noise_spec, backbone, dim=32, num_blocks=[2, 2, 2, 2, 2], pooling=True, groups=[2, 2, 2, 2, 2], scale=4
    ):
        super(UNet, self).__init__()
        self.noise_spec = noise_spec
        self.backbone = backbone
        self.dim = dim
        self.num_blocks = num_blocks
        self.pooling = pooling
        self.groups = groups
        self.scale = scale
        if self.noise_spec["N_cov"] and self.noise_spec["N_ch"] != 0:
            patch_size = self.noise_spec["patch_size"]
            kernel_size = self.noise_spec["kernel_size"]
            self.Covariance = common.NoiseCovariance(kernel_size, patch_size, stride=1, padding=kernel_size // 2)
            cat_ch = patch_size**2
            print("Covariance Configs: ", self.Covariance.kernel_size, self.Covariance.patch_size)
        else:
            cat_ch = self.noise_spec["N_ch"]

        growth = 0.0
        enc_drop = [x * growth for x in range(4)]
        conv_drop = 4 * growth
        dec_drop = enc_drop[::-1]
        print(enc_drop, conv_drop, dec_drop)

        self.conv0 = nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1)
        self.ConvBlock1 = ConvBlock(
            dim,
            dim,
            noise_ch=cat_ch,
            strides=1,
            backbone=backbone,
            num_blocks=num_blocks[0],
            pooling=pooling,
            groups=groups[0],
            scale=scale,
            dropout=enc_drop[0],
        )
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = ConvBlock(
            dim,
            dim * 2,
            noise_ch=cat_ch,
            strides=1,
            backbone=backbone,
            num_blocks=num_blocks[1],
            pooling=pooling,
            groups=groups[1],
            scale=scale,
            dropout=enc_drop[1],
        )
        # print(self.ConvBlock2)
        self.pool2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = ConvBlock(
            dim * 2,
            dim * 4,
            noise_ch=cat_ch,
            strides=1,
            backbone=backbone,
            num_blocks=num_blocks[2],
            pooling=pooling,
            groups=groups[2],
            scale=scale,
            dropout=enc_drop[2],
        )
        self.pool3 = nn.Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = ConvBlock(
            dim * 4,
            dim * 8,
            noise_ch=cat_ch,
            strides=1,
            backbone=backbone,
            num_blocks=num_blocks[3],
            pooling=pooling,
            groups=groups[3],
            scale=scale,
            dropout=enc_drop[3],
        )
        self.pool4 = nn.Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = ConvBlock(
            dim * 8,
            dim * 16,
            noise_ch=cat_ch,
            strides=1,
            backbone=backbone,
            num_blocks=num_blocks[4],
            pooling=pooling,
            groups=groups[4],
            scale=scale,
            dropout=conv_drop,
        )

        self.upsample6 = nn.ConvTranspose2d(dim * 16, dim * 8, 2, stride=2)
        self.ConvBlock6 = ConvBlock(
            dim * 16,
            dim * 8,
            strides=1,
            noise_ch=cat_ch if self.backbone != "conv" else 0,
            backbone=backbone,
            num_blocks=num_blocks[3],
            pooling=pooling,
            groups=groups[3],
            scale=scale,
            dropout=dec_drop[0],
        )

        self.upsample7 = nn.ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.ConvBlock7 = ConvBlock(
            dim * 8,
            dim * 4,
            strides=1,
            noise_ch=cat_ch if self.backbone != "conv" else 0,
            backbone=backbone,
            num_blocks=num_blocks[2],
            pooling=pooling,
            groups=groups[2],
            scale=scale,
            dropout=dec_drop[1],
        )

        self.upsample8 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.ConvBlock8 = ConvBlock(
            dim * 4,
            dim * 2,
            strides=1,
            noise_ch=cat_ch if self.backbone != "conv" else 0,
            backbone=backbone,
            num_blocks=num_blocks[1],
            pooling=pooling,
            groups=groups[1],
            scale=scale,
            dropout=dec_drop[2],
        )

        self.upsample9 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.ConvBlock9 = ConvBlock(
            dim * 2,
            dim,
            strides=1,
            noise_ch=cat_ch if self.backbone != "conv" else 0,
            backbone=backbone,
            num_blocks=num_blocks[0],
            pooling=pooling,
            groups=groups[0],
            scale=scale,
            dropout=dec_drop[3],
        )

        self.conv10 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)
        self.norm = _norm_fused if self.noise_spec["N_cov"] else (lambda n: n)

    def noise_preprocessing(self, noise):
        if noise is None:
            return None
        if self.noise_spec["N_cov"]:
            noise = self.Covariance(noise)
        return noise

    def forward(self, inp, noise=None, benchmark=False):
        if not benchmark:
            noise = self.noise_preprocessing(noise)

        if noise is None:
            noise_list = [None] * 5
        else:
            if self.backbone != "conv":
                noise = F.avg_pool2d(noise, self.scale, self.scale)
            noise_list = [noise]
            for _ in range(4):
                noise_list.append(F.avg_pool2d(noise_list[-1], 2, 2))
            if self.norm is not None:
                noise_list = [self.norm(n) for n in noise_list]

        conv0 = self.conv0(inp)
        conv1 = self.ConvBlock1(conv0, noise=noise_list[0])
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1, noise=noise_list[1])
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2, noise=noise_list[2])
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3, noise=noise_list[3])
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4, noise=noise_list[4])

        up6 = self.upsample6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6, noise=noise_list[3])

        up7 = self.upsample7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7, noise=noise_list[2])

        up8 = self.upsample8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8, noise=noise_list[1])

        up9 = self.upsample9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9, noise=noise_list[0])

        conv10 = self.conv10(conv9)
        return conv10 + inp

    def flops(self, H=512, W=512):  # its MACs!
        if self.noise_spec["N_cov"] and self.noise_spec["N_ch"] != 0:
            cov_flops = self.Covariance.flops(self.noise_spec["N_ch"], H, W)
        else:
            cov_flops = 0

        model_flops = 0
        model_flops += H * W * self.conv0.in_channels * self.conv0.out_channels * 3 * 3
        model_flops += self.ConvBlock1.flops(H, W)
        # pool1
        model_flops += (H / 2) * (W / 2) * self.pool1.in_channels * self.pool1.out_channels * 4 * 4
        model_flops += self.ConvBlock2.flops(H / 2, W / 2)
        # pool2
        model_flops += (H / 4) * (W / 4) * self.pool2.in_channels * self.pool2.out_channels * 4 * 4
        model_flops += self.ConvBlock3.flops(H / 4, W / 4)
        # pool3
        model_flops += (H / 8) * (W / 8) * self.pool3.in_channels * self.pool3.out_channels * 4 * 4
        model_flops += self.ConvBlock4.flops(H / 8, W / 8)
        # pool4
        model_flops += (H / 16) * (W / 16) * self.pool4.in_channels * self.pool4.out_channels * 4 * 4
        model_flops += self.ConvBlock5.flops(H / 16, W / 16)
        # upsample6
        model_flops += (H / 16) * (W / 16) * self.upsample6.in_channels * self.upsample6.out_channels * 2 * 2
        model_flops += self.ConvBlock6.flops(H / 8, W / 8)
        # upsample7
        model_flops += (H / 8) * (W / 8) * self.upsample7.in_channels * self.upsample7.out_channels * 2 * 2
        model_flops += self.ConvBlock7.flops(H / 4, W / 4)
        # upsample8
        model_flops += (H / 4) * (W / 4) * self.upsample8.in_channels * self.upsample8.out_channels * 2 * 2
        model_flops += self.ConvBlock8.flops(H / 2, W / 2)
        # upsample9
        model_flops += (H / 2) * (W / 2) * self.upsample9.in_channels * self.upsample9.out_channels * 2 * 2
        model_flops += self.ConvBlock9.flops(H, W)
        model_flops += H * W * self.conv10.in_channels * self.conv10.out_channels * 3 * 3
        return cov_flops, model_flops
