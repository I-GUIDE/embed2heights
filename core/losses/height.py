"""Height-specific loss helpers."""

import torch
import torch.nn.functional as F


def height_error(pred, target, *, kind, huber_delta, pinball_tau=0.5):
    """Elementwise height error under the configured regression loss.

    'pinball' is the quantile (tilted-L1) loss. With diff = pred - target and
    quantile tau, the loss is max(-tau * diff, (1 - tau) * diff): tau > 0.5
    penalizes UNDER-prediction more than over-prediction. This matches the DSM
    target (which records the highest surface point), where under-estimating
    building/vegetation height is the costlier error. tau = 0.5 reduces to
    0.5 * |diff|, i.e. half of L1.
    """
    diff = pred - target
    if kind == "l1":
        return torch.abs(diff)
    if kind == "mse":
        return diff * diff
    if kind == "pinball":
        tau = float(pinball_tau)
        return torch.maximum(-tau * diff, (1.0 - tau) * diff)

    abs_diff = torch.abs(diff)
    delta = float(huber_delta)
    quadratic = 0.5 * diff * diff / delta
    linear = abs_diff - 0.5 * delta
    return torch.where(abs_diff <= delta, quadratic, linear)


def height_bin_ce(logits, target_norm, mask, log_centers, sigma_bins, *,
                  height_norm_stats=None):
    """Soft-target cross-entropy in log-height space."""
    if mask is None or torch.sum(mask) <= 0 or logits is None or log_centers is None:
        return torch.zeros(
            (),
            device=logits.device if logits is not None else target_norm.device,
            dtype=logits.dtype if logits is not None else target_norm.dtype,
        )

    from ..data.height_stats import denormalize_height_torch
    from ..data.datasets import HEIGHT_NORM_CONSTANT

    if height_norm_stats is not None:
        meters = denormalize_height_torch(target_norm, height_norm_stats)
    else:
        meters = target_norm * HEIGHT_NORM_CONSTANT
    log_target = torch.log1p(meters)
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
