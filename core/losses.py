import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation.
    alpha: penalizes False Positives.
    beta: penalizes False Negatives.
    Setting beta > alpha forces the model to capture minority classes (like sparse buildings).
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets, valid_mask=None):
        """
        Args:
            preds, targets: (B, H, W) or (B, N)
            valid_mask: optional bool/float mask, same shape as preds.
                        Only valid pixels contribute to TP/FP/FN (no dilution
                        from nodata zeros).
        """
        batch_size = preds.size(0)
        p = preds.reshape(batch_size, -1)
        t = targets.reshape(batch_size, -1)

        if valid_mask is not None:
            m = valid_mask.reshape(batch_size, -1).bool()
            # Compute per-sample on valid pixels only; pad invalid with 0 so
            # they contribute nothing to sums (and mask-zero denominators too).
            p = torch.where(m, p, torch.zeros_like(p))
            t = torch.where(m, t, torch.zeros_like(t))
            # (1 - p) and (1 - t) terms must also be zeroed on invalid pixels
            one_minus_p = torch.where(m, 1.0 - p, torch.zeros_like(p))
            one_minus_t = torch.where(m, 1.0 - t, torch.zeros_like(t))
        else:
            one_minus_p = 1.0 - p
            one_minus_t = 1.0 - t

        TP = torch.sum(p * t, dim=1)
        FP = torch.sum(p * one_minus_t, dim=1)
        FN = torch.sum(one_minus_p * t, dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return torch.mean(1.0 - tversky)


class GradientDifferenceLoss(nn.Module):
    """Penalizes differences in image gradients (edges/sharpness)."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        loss_x = torch.mean(torch.abs(pred_dx - target_dx))
        loss_y = torch.mean(torch.abs(pred_dy - target_dy))

        return loss_x + loss_y


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) Loss using a Gaussian window."""

    def __init__(self, window_size=5, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            elif img1.device.type == 'mps':
                window = window.to('mps')

            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class ImprovedCompositeLoss(nn.Module):
    """
    Combined Recipe:
    1. Split Foreground/Background MAE (L1).
    2. SSIM and Gradient Loss on RGB channels.
    3. Tversky Loss specifically to boost Building recall.
    4. Structure-Boosted Height MAE.
    """

    def __init__(self, lambdas=[1.0, 0.5, 0.5, 2.0], bg_weight=0.05):
        super().__init__()
        self.ssim = SSIMLoss(window_size=11)
        self.gdl = GradientDifferenceLoss()

        # Using Tversky instead of Jaccard to heavily penalize missing buildings
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

        self.w_mae = lambdas[0]
        self.w_ssim = lambdas[1]
        self.w_grad = lambdas[2]
        self.w_structure = lambdas[3]  # Represents both Tversky and Building-Height weights

        self.bg_weight = bg_weight

        # Height is now normalized, so we weight all 4 channels equally in base MAE
        self.mae_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).float()

    def forward(self, preds, targets, valid_mask=None):
        """
        Args:
            preds:      (B, 4, H, W) model predictions
            targets:    (B, 4, H, W) ground truth
            valid_mask: (B, 2, H, W) float mask. Channel 0 = global validity (for land cover),
                        Channel 1 = height validity (global + nDSM hole exclusion).
                        If None, all pixels are valid.
        """
        device = preds.device
        mae_weights = self.mae_weights.to(device)

        # Default: all pixels valid (backward compatible)
        if valid_mask is None:
            valid_mask = torch.ones(preds.shape[0], 2, preds.shape[2], preds.shape[3],
                                    device=device)

        global_mask = valid_mask[:, 0:1, :, :]   # (B, 1, H, W) — land cover validity
        height_mask = valid_mask[:, 1:2, :, :]   # (B, 1, H, W) — height validity (stricter)

        # Per-channel mask: global for bands 0-2, height-specific for band 3
        ch_mask = torch.cat([global_mask.expand(-1, 3, -1, -1), height_mask], dim=1)  # (B, 4, H, W)
        global_1ch = global_mask.squeeze(1)       # (B, H, W)
        height_1ch = height_mask.squeeze(1)       # (B, H, W)

        # --- 1. Base MAE (Foreground / Background Split) with nodata masking ---
        abs_err = torch.abs(preds - targets)
        fg_mask = (targets > 0).float() * ch_mask
        bg_mask = (1.0 - (targets > 0).float()) * ch_mask

        fg_sum = torch.sum(fg_mask, dim=(0, 2, 3)) + 1e-6
        bg_sum = torch.sum(bg_mask, dim=(0, 2, 3)) + 1e-6

        mae_fg = torch.sum(abs_err * fg_mask, dim=(0, 2, 3)) / fg_sum
        mae_bg = torch.sum(abs_err * bg_mask, dim=(0, 2, 3)) / bg_sum

        mae_per_channel = mae_fg + (self.bg_weight * mae_bg)
        loss_mae = torch.sum(mae_per_channel * mae_weights)

        # --- 2. Structural & Gradient Loss on Land Cover Only ---
        # Zero out predictions at nodata pixels so they match targets (no spurious loss)
        lc_mask = global_mask.expand(-1, 3, -1, -1)       # (B, 3, H, W)
        lc_pred = preds[:, :3, :, :] * lc_mask
        lc_target = targets[:, :3, :, :] * lc_mask

        loss_ssim = self.ssim(lc_pred, lc_target)
        loss_grad = self.gdl(lc_pred, lc_target)

        # --- 3. Multi-Class Tversky Loss with nodata masking (global mask for land cover) ---
        # Pass the mask explicitly so invalid pixels are excluded from TP/FP/FN
        # (no dilution from zero-filled nodata regions).
        gm_bool = global_1ch.bool()
        t_build = self.tversky(torch.relu(preds[:, 0, :, :]),
                               targets[:, 0, :, :], valid_mask=gm_bool)
        t_veg = self.tversky(torch.relu(preds[:, 1, :, :]),
                             targets[:, 1, :, :], valid_mask=gm_bool)
        t_water = self.tversky(torch.relu(preds[:, 2, :, :]),
                               targets[:, 2, :, :], valid_mask=gm_bool)

        loss_tversky = (t_build + t_veg + t_water) / 3.0

        # --- 4. Height-Aware Building Masking with nDSM-specific mask ---
        # Use height_mask: excludes both global nodata AND nDSM holes
        build_presence_mask = (targets[:, 0, :, :] > 0.1).float() * height_1ch
        height_err = torch.abs(preds[:, 3, :, :] - targets[:, 3, :, :]) * height_1ch

        height_valid_count = torch.sum(height_1ch) + 1e-6
        loss_height_boost = torch.sum(height_err * (1.0 + 5.0 * build_presence_mask)) / height_valid_count

        # --- Combine Total Loss ---
        total_loss = (self.w_mae * loss_mae) + \
                     (self.w_ssim * loss_ssim) + \
                     (self.w_grad * loss_grad) + \
                     (self.w_structure * loss_tversky) + \
                     (self.w_structure * loss_height_boost)

        return total_loss, loss_mae, loss_ssim, loss_grad, loss_tversky