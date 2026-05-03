"""Height-specific loss helpers."""

import torch
import torch.nn.functional as F


def height_error(pred, target, *, kind, huber_delta):
    """Elementwise height error under the configured regression loss."""
    diff = pred - target
    if kind == "l1":
        return torch.abs(diff)
    if kind == "mse":
        return diff * diff

    abs_diff = torch.abs(diff)
    delta = float(huber_delta)
    quadratic = 0.5 * diff * diff / delta
    linear = abs_diff - 0.5 * delta
    return torch.where(abs_diff <= delta, quadratic, linear)


def height_bin_ce(logits, target_norm, mask, log_centers, sigma_bins):
    """Soft-target cross-entropy in log-height space."""
    if mask is None or torch.sum(mask) <= 0 or logits is None or log_centers is None:
        return torch.zeros(
            (),
            device=logits.device if logits is not None else target_norm.device,
            dtype=logits.dtype if logits is not None else target_norm.dtype,
        )

    log_target = torch.log1p(target_norm * 30.0)
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
