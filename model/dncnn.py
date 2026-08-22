import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


@torch.jit.script
def _norm_fused(n: torch.Tensor) -> torch.Tensor:
    return torch.sign(n) * torch.log1p(torch.abs(n))


def make_model(config):  # T: tiny, S: small, B: base
    if config["model"]["level"] == "S":
        n_layers = 3
    elif config["model"]["level"] == "B":
        n_layers = 8
    print("model loaded!: DNCNN-" + config["model"]["level"] + "+" + config["model"]["backbone"])
    return DnCNN(
        noise_spec=config["N_augment"],
        backbone=config["model"]["backbone"],
        num_layers=n_layers,
        num_blocks=config["model"]["num_blocks"],
        pooling=config["model"]["pooling"],
        groups=config["model"]["groups"],
        scale=config["model"]["scale"],
    )


class DnCNN(nn.Module):
    def __init__(self, noise_spec, backbone, num_layers, num_blocks, pooling, groups, scale):
        super(DnCNN, self).__init__()
        self.noise_spec = noise_spec
        self.backbone = backbone
        self.num_layers = num_layers
        self.channel = 64
        self.num_blocks = num_blocks[0] if isinstance(num_blocks, list) else num_blocks
        self.pooling = pooling
        self.groups = groups[0] if isinstance(groups, list) else groups
        self.scale = scale
        self.norm = _norm_fused if self.noise_spec["N_cov"] else (lambda n: n)
        if self.noise_spec["N_cov"] and self.noise_spec["N_ch"] != 0:
            patch_size = self.noise_spec["patch_size"]
            kernel_size = self.noise_spec["kernel_size"]
            self.Covariance = common.NoiseCovariance(kernel_size=kernel_size, patch_size=patch_size, stride=1, padding=kernel_size // 2)
            cat_ch = patch_size**2
            print("Covariance Configs: ", self.Covariance.kernel_size, self.Covariance.patch_size)
        else:
            cat_ch = self.noise_spec["N_ch"]

        layers = []
        if self.backbone == "conv":
            layers.append(nn.Conv2d(in_channels=1 + cat_ch, out_channels=self.channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(self.num_layers - 2):
                layers.append(nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(self.channel))
                layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            for i in range(self.num_layers - 2):
                if i % 2 == 0:
                    layers.append(
                        common.FlyingConv(
                            self.channel,
                            self.channel,
                            n_ch=cat_ch,
                            num_blocks=self.num_blocks,
                            pooling=self.pooling,
                            groups=self.groups,
                            scale=self.scale,
                            dropout=0,
                        )
                    )
                else:
                    layers.append(nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(self.channel))
                layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=3, padding=1))
        self.dncnn = nn.ModuleList(layers)

    def noise_preprocessing(self, noise):
        if noise is None:
            return None
        if self.noise_spec["N_cov"]:
            noise = self.Covariance(noise)
        return noise

    def forward(self, x, noise=None, benchmark=False):
        if not benchmark:
            noise = self.noise_preprocessing(noise)

        inp = x
        if noise is not None:
            if self.backbone != "conv":
                noise = F.avg_pool2d(noise, self.scale, self.scale)
                noise = self.norm(noise)
            else:
                noise = self.norm(noise)
        x = torch.cat((x, noise), dim=1) if self.noise_spec["N_ch"] > 0 and self.backbone == "conv" else x
        for layer in self.dncnn:
            if isinstance(layer, common.FlyingConv):
                x = layer(x, noise=noise)
            else:
                x = layer(x)
        return x + inp

    def flops(self, H=512, W=512):
        if self.noise_spec["N_cov"] and self.noise_spec["N_ch"] != 0:
            cov_flops = self.Covariance.flops(self.noise_spec["N_ch"], H, W)
        else:
            cov_flops = 0

        model_flops = 0
        for layer in self.dncnn:
            if isinstance(layer, nn.Conv2d):
                model_flops += layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels * layer.out_channels * H * W
            elif isinstance(layer, nn.BatchNorm2d):
                model_flops += 2 * layer.num_features * H * W
            elif isinstance(layer, common.FlyingConv):
                model_flops += layer.flops(H, W)
        return cov_flops, model_flops