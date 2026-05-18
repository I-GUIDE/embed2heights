"""Segmentation-oriented loss primitives."""

import torch
import torch.nn as nn


def _lovasz_grad(gt_sorted):
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-8)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels, valid_mask=None):
    """Lovász hinge for binary segmentation (Berman, Triki, Blaschko CVPR 2018).
    logits: arbitrary-shape float tensor (raw, pre-sigmoid).
    labels: same shape, in {0,1} (will be converted to {-1,+1}).
    valid_mask: optional bool/float mask same shape as logits; False/0 pixels are skipped.
    Run in float32 for stability under AMP (caller's responsibility — typically wrap in autocast(enabled=False)).
    """
    logits = logits.reshape(-1)
    labels = labels.reshape(-1)
    if valid_mask is not None:
        m = valid_mask.reshape(-1).bool()
        logits = logits[m]
        labels = labels[m]
    if logits.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = (1.0 - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(torch.relu(errors_sorted), grad)


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
