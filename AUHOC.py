import torch
import torch.nn.functional as F
import utils


class AUHOC:
    def __init__(self, patch_size=64, air_thresh=-750, return_curve=False, device="cuda"):
        self.patch_size = patch_size
        self.device = device
        self.maxindex = patch_size // 2
        self.return_curve = return_curve
        self.air_thresh = air_thresh

        # Create coordinate grid and precompute radial averaging matrix (spinavej)
        r = torch.arange(patch_size, device=device) - self.maxindex
        R, C = torch.meshgrid(r, r, indexing="ij")
        dist = torch.sqrt((R**2 + C**2).float())

        indices_f = [(torch.floor(dist).long() == i).float().view(-1) for i in range(self.maxindex)]
        indices_C = [(torch.ceil(dist).long() == i).float().view(-1) for i in range(self.maxindex)]
        self.ring_matrix = (torch.stack(indices_f) + torch.stack(indices_C)).t() / 2.0

        hann_1d = torch.hann_window(patch_size, periodic=False, device=device)
        self.hann_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)

    def apply_hanning_2d(self, img):
        """Applies Hanning filter to minimize boundary effects."""
        return img * self.hann_2d

    def spinavej(self, x):
        """Extracts rings and performs radial average of real parts."""
        return torch.matmul(x.reshape(x.shape[0], -1), self.ring_matrix.to(x.dtype))

    def FRC(self, i1, i2):
        """Calculates Fourier Ring Correlation for the two input arrays."""
        I1 = torch.fft.fftshift(torch.fft.fft2(i1, dim=(-2, -1)), dim=(-2, -1))
        I2 = torch.fft.fftshift(torch.fft.fft2(i2, dim=(-2, -1)), dim=(-2, -1))

        C = self.spinavej(I1 * torch.conj(I2)).real.float()
        C1 = self.spinavej(torch.abs(I1) ** 2).real.float()
        C2 = self.spinavej(torch.abs(I2) ** 2).real.float()

        FSC = torch.abs(C) / (torch.sqrt(C1 * C2) + 1e-9)
        x_fsc = torch.arange(self.maxindex, device=self.device).float() / (self.patch_size / 2.0)
        return x_fsc, FSC

    def forward(self, input_image, ref_image, apply_hann=True, freq_band=[0, 1], normalize=True, air_thresh=-750, window=[-160, 240]):
        """Calculates AU-HOC."""
        input_image = input_image.to(self.device).float()
        ref_image = ref_image.to(self.device).float()
        B, _, H, W = input_image.shape

        inp_patches, ref_patches = utils.patching_images(input_image, ref_image, self.patch_size)
        candidate_indices = utils.air_thresholding(ref_patches, self.air_thresh, self.patch_size)
        num_patches = ref_patches.shape[1]

        ref_patches = ref_patches.clamp_(window[0], window[1])
        inp_patches = inp_patches.clamp_(window[0], window[1])
        if normalize:
            min_val = ref_patches.amin(dim=(-2, -1), keepdim=True)
            max_val = ref_patches.amax(dim=(-2, -1), keepdim=True)
            inp_patches = (inp_patches - min_val) / (max_val - min_val).clamp(min=1e-8)
            ref_patches = (ref_patches - min_val) / (max_val - min_val).clamp(min=1e-8)

        inp_flat = inp_patches.reshape(-1, self.patch_size, self.patch_size)
        ref_flat = ref_patches.reshape(-1, self.patch_size, self.patch_size)

        if apply_hann:
            inp_flat = self.apply_hanning_2d(inp_flat)
            ref_flat = self.apply_hanning_2d(ref_flat)

        xc, FSC_flat = self.FRC(inp_flat, ref_flat)

        # Vectorized intersection frequency calculation
        total_patches = B * num_patches
        diff = FSC_flat - 0.5
        is_below = diff <= 0
        has_intersection = is_below.any(dim=1)
        first_idx = is_below.float().argmax(dim=1)

        batch_ids = torch.arange(total_patches, device=self.device)
        idx_prev = torch.clamp(first_idx - 1, min=0)

        y_curr = diff[batch_ids, first_idx]
        y_prev = diff[batch_ids, idx_prev]
        denom = torch.where((y_prev - y_curr) == 0, torch.ones_like(y_curr), y_prev - y_curr)

        interp_val = xc[idx_prev] + (y_prev / denom) * (xc[first_idx] - xc[idx_prev])
        result = torch.where(first_idx == 0, xc[0], interp_val)
        intersection_freqs = torch.where(has_intersection, result, torch.tensor(float("inf"), device=self.device)).reshape(B, num_patches)

        hts = torch.linspace(freq_band[0], freq_band[1], int((freq_band[1] - freq_band[0]) * 100 + 1), device=self.device)

        au_hocs, curves = [], []
        for b in range(B):
            patch_idx = candidate_indices[b]
            if len(patch_idx) == 0:
                au_hocs.append(torch.tensor(0.0, device=self.device))
                curves.append(torch.zeros(len(hts), device=self.device))
            else:
                rates = (intersection_freqs[b, patch_idx].unsqueeze(1) <= hts).float().mean(dim=0)
                au_hocs.append(rates.mean())
                curves.append(rates)

        if self.return_curve:
            return torch.stack(au_hocs).squeeze(), torch.stack(curves).squeeze()
        return torch.stack(au_hocs).squeeze()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


sfrc = AUHOC(patch_size=64, device="cuda")


def calc_auhoc(input_image, reference_image, apply_hann=True, freq_band=[0, 1], normalize=True, window=[-160, 240]):
    """Calculates threshold-free sFRC (AU-HOC). Wrapper for backward compatibility."""
    return sfrc(input_image, reference_image, apply_hann=apply_hann, normalize=normalize, freq_band=freq_band, window=[-160, 240])
