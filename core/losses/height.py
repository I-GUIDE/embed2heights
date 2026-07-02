"""Height-specific loss helpers."""

import torch
import torch.nn.functional as F

from core.data.datasets import HEIGHT_NORM_CONSTANT


def height_error(pred, target):
    """Elementwise L1 height error (the regression loss used throughout)."""
    return torch.abs(pred - target)


def masked_height(pred, target, mask):
    """Mean L1 height error over the valid pixels in ``mask``."""
    err = height_error(pred, target)
    return torch.sum(err * mask) / (torch.sum(mask) + 1e-6)


def height_bin_ce(logits, target_norm, mask, log_centers, sigma_bins):
    """Soft-target cross-entropy in log-height space."""
    if mask is None or torch.sum(mask) <= 0 or logits is None or log_centers is None:
        return torch.zeros(
            (),
            device=logits.device if logits is not None else target_norm.device,
            dtype=logits.dtype if logits is not None else target_norm.dtype,
        )

    log_target = torch.log1p(target_norm * HEIGHT_NORM_CONSTANT)
    centers = log_centers.to(logits.device, logits.dtype).view(1, -1, 1, 1)
    if log_centers.numel() >= 2:
        bin_width = float((log_centers[1] - log_centers[0]).item())
    else:
        bin_width = 1.0
    sigma = max(float(sigma_bins) * bin_width, 1e-3)

    diff = centers - log_target
    soft_target = torch.exp(-(diff * diff) / (2.0 * sigma * sigma))
    soft_target = soft_target / (soft_target.sum(dim=1, keepdim=True) + 1e-8)

    log_p = F.log_softmax(logits, dim=1)
    ce = -(soft_target * log_p).sum(dim=1, keepdim=True)
    return torch.sum(ce * mask) / (torch.sum(mask) + 1e-6)
