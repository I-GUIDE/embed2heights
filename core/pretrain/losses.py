"""Self-supervised reconstruction losses."""

import torch
import torch.nn.functional as F


def channel_standardize(x, eps=1e-5):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    var = (x - mean).pow(2).mean(dim=(-2, -1), keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps)


def masked_reconstruction_loss(pred, target, mask, cosine_weight=0.05):
    """MSE on masked pixels, with a small per-pixel channel-cosine term."""
    target = channel_standardize(target)
    mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp_min(1.0)
    mse = ((pred - target).pow(2) * mask).sum() / denom

    cosine = pred.new_tensor(0.0)
    if cosine_weight > 0:
        pred_pix = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
        tgt_pix = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
        mask_pix = mask[:, :1].permute(0, 2, 3, 1).reshape(-1) > 0.5
        if mask_pix.any():
            cosine = (1.0 - F.cosine_similarity(
                pred_pix[mask_pix], tgt_pix[mask_pix], dim=1, eps=1e-6
            )).mean()
    return mse + cosine_weight * cosine, {"mse": mse.detach(), "cosine": cosine.detach()}
