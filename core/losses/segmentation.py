"""Segmentation-oriented loss primitives."""

import torch
import torch.nn as nn


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
