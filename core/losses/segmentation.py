"""Segmentation / presence loss primitives.

These operate on the class-fraction and presence channels (building, vegetation,
water); composite.py combines and weights them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation.

    alpha penalizes false positives; beta penalizes false negatives. Setting
    beta > alpha forces the model to capture minority classes such as sparse
    buildings.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets, valid_mask=None):
        """
        Args:
            preds, targets: (B, H, W) or (B, N)
            valid_mask: optional bool/float mask, same shape as preds.
        """
        batch_size = preds.size(0)
        p = preds.reshape(batch_size, -1)
        t = targets.reshape(batch_size, -1)

        if valid_mask is not None:
            m = valid_mask.reshape(batch_size, -1).bool()
            p = torch.where(m, p, torch.zeros_like(p))
            t = torch.where(m, t, torch.zeros_like(t))
            one_minus_p = torch.where(m, 1.0 - p, torch.zeros_like(p))
            one_minus_t = torch.where(m, 1.0 - t, torch.zeros_like(t))
        else:
            one_minus_p = 1.0 - p
            one_minus_t = 1.0 - t

        tp = torch.sum(p * t, dim=1)
        fp = torch.sum(p * one_minus_t, dim=1)
        fn = torch.sum(one_minus_p * t, dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.mean(1.0 - tversky)


def split_fg_bg_mae(pred, target, mask, bg_weight):
    """Per-channel foreground/background-split MAE, summed over channels.
    Foreground = pixels with target > 0; background is down-weighted by bg_weight."""
    abs_err = torch.abs(pred - target)
    fg_mask = (target > 0).float() * mask
    bg_mask = (1.0 - (target > 0).float()) * mask
    fg_sum = torch.sum(fg_mask, dim=(0, 2, 3)) + 1e-6
    bg_sum = torch.sum(bg_mask, dim=(0, 2, 3)) + 1e-6
    mae_fg = torch.sum(abs_err * fg_mask, dim=(0, 2, 3)) / fg_sum
    mae_bg = torch.sum(abs_err * bg_mask, dim=(0, 2, 3)) / bg_sum
    return torch.sum(mae_fg + bg_weight * mae_bg)


def building_ring(mask01, kernel):
    """~(kernel//2)-px ring around a binary mask: dilation minus erosion."""
    pad = kernel // 2
    dil = F.max_pool2d(mask01, kernel, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask01, kernel, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def masked_presence_bce(logits, target, valid_mask, ring=None, ring_alpha=0.0):
    """Masked BCE over the 3 presence channels. When ``ring`` is given, the
    building channel's boundary-ring pixels are upweighted by (1 + ring_alpha*ring)."""
    safe_logits = torch.where(valid_mask.bool(), logits, torch.zeros_like(logits))
    safe_target = torch.where(valid_mask.bool(), target, torch.zeros_like(target))
    loss = F.binary_cross_entropy_with_logits(safe_logits, safe_target, reduction="none")
    weight = valid_mask
    if ring is not None:
        weight = valid_mask.clone()
        weight[:, 0:1, :, :] = weight[:, 0:1, :, :] * (1.0 + ring_alpha * ring)
    return torch.sum(loss * weight) / (torch.sum(weight) + 1e-6)


def empty_water_topk_penalty(water_logit, water_target, global_mask, k):
    """On tiles with NO water, penalise the k highest water probabilities (push
    down spurious water). Returns 0 when every tile has some water."""
    wt = water_target * global_mask
    empty = torch.sum(wt, dim=(1, 2, 3)) <= 0
    if not torch.any(empty):
        return water_logit.new_zeros(())
    water_prob = torch.sigmoid(water_logit) * global_mask
    flat = water_prob[empty].flatten(1)
    if flat.numel() == 0:
        return water_logit.new_zeros(())
    kk = min(int(k), flat.shape[1])
    return torch.topk(flat, kk, dim=1).values.mean()


def _soft_erode(img):
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def _soft_open(img):
    return F.max_pool2d(_soft_erode(img), kernel_size=3, stride=1, padding=1)


def soft_skeleton(img, iters):
    """Differentiable morphological skeleton (Shit et al., clDice)."""
    opened = _soft_open(img)
    skel = F.relu(img - opened)
    for _ in range(iters):
        img = _soft_erode(img)
        opened = _soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice_loss(prob, gt, mask, iters, smooth=1.0):
    """1 - clDice on a single presence channel. prob/gt/mask: (B,1,H,W) in [0,1].

    Topology precision = skeleton(pred) covered by gt; sensitivity = skeleton(gt)
    covered by pred. Penalises merged/broken thin structures (inter-building
    bridges) that area losses are blind to."""
    prob = (prob * mask).clamp(0.0, 1.0)
    gt = (gt * mask).clamp(0.0, 1.0)
    sk_p = soft_skeleton(prob, iters)
    sk_t = soft_skeleton(gt, iters)
    tprec = (torch.sum(sk_p * gt) + smooth) / (torch.sum(sk_p) + smooth)
    tsens = (torch.sum(sk_t * prob) + smooth) / (torch.sum(sk_t) + smooth)
    cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
    return 1.0 - cl_dice


def building_boundary_loss(logits, fractions, mask):
    """Auxiliary edge supervision on the building boundary ring.

    ``fractions`` is (B,3,H,W) GT coverage. The hard building mask is the
    dominant (argmax) class where any class is present — matching the presence
    target convention. Its 3x3 dilation-minus-erosion gives a ~1px ring; BCE
    handles per-pixel error and soft Dice counters the boundary class imbalance.
    Both terms are restricted to valid (``mask``) pixels."""
    any_present = (fractions > 0).any(dim=1, keepdim=True).float()
    argmax_idx = fractions.argmax(dim=1, keepdim=True)
    m = (argmax_idx == 0).float() * any_present            # building = channel 0
    dil = F.max_pool2d(m, 3, stride=1, padding=1)
    ero = 1.0 - F.max_pool2d(1.0 - m, 3, stride=1, padding=1)
    boundary = (dil - ero).clamp(0.0, 1.0)

    bce_map = F.binary_cross_entropy_with_logits(logits, boundary, reduction="none")
    bce = torch.sum(bce_map * mask) / (torch.sum(mask) + 1e-6)

    pred = torch.sigmoid(logits) * mask
    tgt = boundary * mask
    inter = torch.sum(pred * tgt)
    dice = 1.0 - (2.0 * inter + 1.0) / (torch.sum(pred) + torch.sum(tgt) + 1.0)
    return bce + dice
