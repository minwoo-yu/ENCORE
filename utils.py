import os
import torch
import torchvision.transforms as transforms
import datetime
import time
import torch.nn.functional as F
import pytorch_ssim
_ssim = pytorch_ssim.SSIM()

class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


class checkpoint:
    def __init__(self, save_dir, save):
        self.ok = True
        self.train_log = torch.Tensor()
        self.val_log = torch.Tensor()
        self.top3_val_losses = []

        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if not save:
            save = now
        self.dir = os.path.join(save_dir, save)

        print("experiment directory is {}".format(self.dir))
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path("model"), exist_ok=True)
        os.makedirs(self.get_path("results"), exist_ok=True)

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def add_train_log(self, log):
        self.train_log = torch.cat([self.train_log, log])

    def add_val_log(self, log):
        self.val_log = torch.cat([self.val_log, log])

    def save(self, model, is_best=False):
        state_dict = getattr(model, "_orig_mod", model).state_dict()
        if is_best:
            torch.save(state_dict, self.get_path("model", "model_best.pt"))
        torch.save(state_dict, self.get_path("model", "model_latest.pt"))


# get optimizer by name
def get_optimizer(optim_name, net_parameters, lr):
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(net_parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(net_parameters, lr=lr, alpha=0.9)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(net_parameters, lr=lr)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(net_parameters, lr=lr)
    elif optim_name == "sparseadam":
        optimizer = torch.optim.SparseAdam(net_parameters, lr=lr)

    return optimizer


# Averager for loss computation
class Averager:
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


def normalize(tensor, slope=500, intercept=500):
    """
    Normalize the tensor
    """
    tensor = (tensor + intercept) / slope

    return tensor


def denormalize(tensor, slope=500, intercept=500):
    """
    Denormalize the tensor
    """
    tensor = tensor * slope - intercept

    return tensor

def make_coord(shape, device):
    y_seqs = -1 + 1 / shape[0] + (2 / shape[0]) * (torch.arange(shape[0], device=device).float())
    x_seqs = -1 + 1 / shape[1] + (2 / shape[1]) * (torch.arange(shape[1], device=device).float())
    coord = torch.stack(torch.meshgrid((x_seqs, y_seqs), indexing="xy"), dim=-1).unsqueeze(0)
    return coord

def get_transforms():
    return transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

def cal_psnr(output, target, window=[-160, 240]):
    output_clip = output.clamp(window[0], window[1])
    target_clip = target.clamp(window[0], window[1])
    mse = torch.mean((output_clip - target_clip) ** 2, dim=(1, 2, 3))
    max_val = target_clip.amax((1, 2, 3))
    return 10 * torch.log10(max_val**2 / mse)


def cal_ssim(output, target, window=[-160, 240]):
    output_clip = output.clamp(window[0], window[1])
    target_clip = target.clamp(window[0], window[1])
    output_norm = (output_clip - window[0]) / (window[1] - window[0])
    target_norm = (target_clip - window[0]) / (window[1] - window[0])
    return _ssim(output_norm, target_norm)

def patching_images(input_, ref_, patch_size=64):
    """Extracts non-overlapping patches from divisible full-size image pairs."""
    B, _, H, W = input_.shape
    nH, nW = H // patch_size, W // patch_size

    def to_patches(img):
        return img[:, 0].view(B, nH, patch_size, nW, patch_size).permute(0, 1, 3, 2, 4).reshape(B, -1, patch_size, patch_size)

    return to_patches(input_), to_patches(ref_)


def air_thresholding(sub_label, air_thresh=-800, patch_size=64):
    """Filters out background/air patches with 4-connectivity flood-fill."""
    B, num_patches, _, _ = sub_label.shape
    means = sub_label.mean(dim=(2, 3))
    N = int(num_patches ** 0.5)
    mask = (means > air_thresh).reshape(B, N, N).float()

    bg = torch.zeros_like(mask)
    bg[:, [0, -1], :] = 1.0 - mask[:, [0, -1], :]
    bg[:, :, [0, -1]] = 1.0 - mask[:, :, [0, -1]]

    inv_mask = 1.0 - mask
    for _ in range(2 * N):
        bg_left = F.pad(bg[:, :, :-1], (1, 0))
        bg_right = F.pad(bg[:, :, 1:], (0, 1))
        bg_up = F.pad(bg[:, :-1, :], (0, 0, 1, 0))
        bg_down = F.pad(bg[:, 1:, :], (0, 0, 0, 1))
        bg_next = torch.max(torch.max(torch.max(bg, bg_left), bg_right), torch.max(bg_up, bg_down)) * inv_mask

        if torch.equal(bg_next, bg):
            break
        bg = bg_next

    return [torch.where(f.ravel())[0] for f in (bg == 0.0)]
