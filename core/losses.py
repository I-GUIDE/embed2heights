import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla 2018).

    Wraps an inner ImprovedCompositeLoss and replaces its hand-tuned
    aux/presence/structure scalars with learned log-variances per task group.
    The total per Kendall et al. is

        L = sum_i 0.5 * exp(-s_i) * L_i + 0.5 * s_i

    where s_i = log(sigma_i^2) is a learned scalar per task. The log-term
    prevents the trivial "push all sigmas to infinity" minimum.

    Task groupings:
        presence : presence_bce + presence_tversky    (classification-like)
        fraction : fraction_mae + tversky             (soft regression)
        height   : height_boost + aux_height          (regression)

    The inner loss still computes raw component values; this module only
    re-weights them. So the existing component logging stays meaningful.
    """

    def __init__(self, inner_loss):
        super().__init__()
        self.inner = inner_loss
        # Initialize at log_var=0 → variance=1 → effective weight=0.5 for all
        # tasks. Letting the optimizer see all three from the same starting
        # weight means we don't bake in the previous hand-tuned ordering.
        self.log_var_presence = nn.Parameter(torch.zeros(()))
        self.log_var_fraction = nn.Parameter(torch.zeros(()))
        self.log_var_height = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _uw(log_var, loss):
        # 0.5 * exp(-s) * L + 0.5 * s
        return 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var

    def forward(self, preds, targets, valid_mask=None):
        # Run the inner loss to get raw per-component values. We will discard
        # its hand-weighted total and reassemble using learned uncertainties.
        _ignored_total, components = self.inner(preds, targets, valid_mask)

        zero = torch.zeros((), device=self.log_var_presence.device,
                           dtype=self.log_var_presence.dtype)

        L_presence = (components.get("presence_bce", zero)
                      + components.get("presence_tversky", zero))
        L_fraction = (components.get("fraction_mae", zero)
                      + components.get("tversky", zero))
        L_height = (components.get("height_boost", zero)
                    + components.get("aux_height", zero))

        total = (self._uw(self.log_var_presence, L_presence)
                 + self._uw(self.log_var_fraction, L_fraction)
                 + self._uw(self.log_var_height, L_height))

        # Annotate components with the learned log-vars for logging/debug.
        components = dict(components)
        components["uw_log_var_presence"] = self.log_var_presence.detach()
        components["uw_log_var_fraction"] = self.log_var_fraction.detach()
        components["uw_log_var_height"] = self.log_var_height.detach()
        components["uw_L_presence"] = L_presence.detach()
        components["uw_L_fraction"] = L_fraction.detach()
        components["uw_L_height"] = L_height.detach()
        return total, components


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
        # Registered as buffer so .to(device) / autocast follow the module.
        self.register_buffer("window", self.create_window(window_size, self.channel),
                             persistent=False)

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

        if channel != self.channel or self.window.device != img1.device:
            window = self.create_window(self.window_size, channel).to(
                device=img1.device, dtype=self.window.dtype)
            self.window = window
            self.channel = channel

        window = self.window.to(dtype=img1.dtype)
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class ImprovedCompositeLoss(nn.Module):
    """
    Combined Recipe:
    1. Split Foreground/Background MAE (L1).
    2. SSIM and Gradient Loss on RGB channels.
    3. Tversky Loss specifically to boost Building recall.
    4. Structure-Boosted Height MAE.
    """

    def __init__(self, lambdas=[1.0, 0.5, 0.5, 2.0], bg_weight=0.05,
                 aux_weight=0.25, loss_preset="current",
                 presence_tversky_weight=1.0, fraction_mae_weight=0.1,
                 height_loss_kind="l1", huber_delta=1.0,
                 build_height_boost=5.0,
                 veg_height_boost=0.0,
                 aux_veg_weight=1.0,
                 iou_loss_kind="tversky",
                 focal_gamma=2.0, focal_alpha=0.25,
                 aux_tversky_weight=0.0):
        super().__init__()
        if loss_preset not in {"current", "no_ssim_grad", "presence_centered"}:
            raise ValueError(
                "loss_preset must be one of: current, no_ssim_grad, presence_centered"
            )
        if height_loss_kind not in {"l1", "huber", "mse"}:
            raise ValueError("height_loss_kind must be one of: l1, huber, mse")
        if iou_loss_kind not in {"tversky", "focal"}:
            raise ValueError("iou_loss_kind must be one of: tversky, focal")
        self.loss_preset = loss_preset
        self.ssim = SSIMLoss(window_size=11)
        self.gdl = GradientDifferenceLoss()

        # Using Tversky instead of Jaccard to heavily penalize missing buildings
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

        self.w_mae = lambdas[0]
        self.w_ssim = 0.0 if loss_preset in {"no_ssim_grad", "presence_centered"} else lambdas[1]
        self.w_grad = 0.0 if loss_preset in {"no_ssim_grad", "presence_centered"} else lambdas[2]
        self.w_structure = lambdas[3]  # Represents both Tversky and Building-Height weights

        self.bg_weight = bg_weight
        self.aux_weight = aux_weight
        self.presence_tversky_weight = presence_tversky_weight
        self.fraction_mae_weight = fraction_mae_weight

        self.height_loss_kind = height_loss_kind
        self.huber_delta = float(huber_delta)
        self.build_height_boost = float(build_height_boost)
        self.veg_height_boost = float(veg_height_boost)
        self.aux_veg_weight = float(aux_veg_weight)
        self.iou_loss_kind = iou_loss_kind
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        # Under presence_centered the main Tversky on fractions is off by
        # default (the presence head owns the IoU objective). Setting this
        # > 0 re-enables it on the aux fraction map as an auxiliary term —
        # a direct IoU surrogate on the main output channels.
        self.aux_tversky_weight = float(aux_tversky_weight)

        # Height is now normalized, so we weight all 4 channels equally in base MAE
        self.mae_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).float()
        self.fraction_mae_weights = torch.tensor([1.0, 1.0, 1.0]).float()

    def _height_err(self, pred, target):
        """Elementwise height error under the configured regression loss.

        Returns a tensor with the same shape as pred/target containing per-pixel
        loss values (no reduction, no masking — callers handle those).
        """
        diff = pred - target
        if self.height_loss_kind == "l1":
            return torch.abs(diff)
        if self.height_loss_kind == "mse":
            return diff * diff
        abs_diff = torch.abs(diff)
        delta = self.huber_delta
        quadratic = 0.5 * diff * diff / delta
        linear = abs_diff - 0.5 * delta
        return torch.where(abs_diff <= delta, quadratic, linear)

    def _focal_bce_loss(self, logits, target, mask):
        """Sigmoid-focal BCE, masked; scalar mean over valid pixels."""
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = target * p + (1.0 - target) * (1.0 - p)
        alpha_t = target * self.focal_alpha + (1.0 - target) * (1.0 - self.focal_alpha)
        focal_weight = alpha_t * (1.0 - p_t).clamp_min(1e-6).pow(self.focal_gamma)
        loss = focal_weight * bce
        denom = torch.sum(mask) + 1e-6
        return torch.sum(loss * mask) / denom

    def forward(self, preds, targets, valid_mask=None):
        """
        Args:
            preds:      (B, 4, H, W) model predictions, or a dict with
                        preds["out"] plus optional auxiliary heads
            targets:    (B, 4, H, W) ground truth
            valid_mask: (B, 2, H, W) float mask. Channel 0 = global validity (for land cover),
                        Channel 1 = height validity (global + nDSM hole exclusion).
                        If None, all pixels are valid.
        """
        aux_outputs = preds if isinstance(preds, dict) else None
        if aux_outputs is not None:
            preds = aux_outputs["out"]

        device = preds.device
        mae_weights = self.mae_weights.to(device)
        fraction_mae_weights = self.fraction_mae_weights.to(device)

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

        # With the dual-head design (v3), submission channels 0-2 of `preds`
        # are presence_prob (binary-aligned), so the soft-regression losses
        # (MAE/SSIM/Gradient/Tversky on land cover) must target the auxiliary
        # `fractions` instead, which is still a soft [0,1] coverage regressor.
        # Height (channel 3) continues to come from `preds[:, 3]`.
        if aux_outputs is not None and "fractions" in aux_outputs:
            class_pred = aux_outputs["fractions"]  # (B, 3, H, W) — soft fractions
        else:
            class_pred = preds[:, :3, :, :]        # fallback for models without a split head

        reg_pred = torch.cat([class_pred, preds[:, 3:4, :, :]], dim=1)  # (B, 4, H, W) for MAE

        def split_fg_bg_mae(pred, target, mask, weights):
            abs_err = torch.abs(pred - target)
            fg_mask = (target > 0).float() * mask
            bg_mask = (1.0 - (target > 0).float()) * mask

            fg_sum = torch.sum(fg_mask, dim=(0, 2, 3)) + 1e-6
            bg_sum = torch.sum(bg_mask, dim=(0, 2, 3)) + 1e-6

            mae_fg = torch.sum(abs_err * fg_mask, dim=(0, 2, 3)) / fg_sum
            mae_bg = torch.sum(abs_err * bg_mask, dim=(0, 2, 3)) / bg_sum
            mae_per_channel = mae_fg + (self.bg_weight * mae_bg)
            return torch.sum(mae_per_channel * weights)

        # --- 1. Base MAE (Foreground / Background Split) with nodata masking ---
        loss_mae = split_fg_bg_mae(reg_pred, targets, ch_mask, mae_weights)
        loss_fraction_mae = split_fg_bg_mae(
            class_pred,
            targets[:, :3, :, :],
            global_mask.expand(-1, 3, -1, -1),
            fraction_mae_weights,
        )

        # --- 2. Structural & Gradient Loss on Land Cover Only ---
        # Zero out predictions at nodata pixels so they match targets (no spurious loss)
        lc_mask = global_mask.expand(-1, 3, -1, -1)       # (B, 3, H, W)
        lc_pred = class_pred * lc_mask
        lc_target = targets[:, :3, :, :] * lc_mask

        zero = torch.zeros((), device=device, dtype=preds.dtype)
        loss_ssim = self.ssim(lc_pred, lc_target) if self.w_ssim != 0 else zero
        loss_grad = self.gdl(lc_pred, lc_target) if self.w_grad != 0 else zero

        # --- 3. Multi-Class Tversky Loss on the auxiliary soft fractions ---
        # Pass the mask explicitly so invalid pixels are excluded from TP/FP/FN
        # (no dilution from zero-filled nodata regions).
        gm_bool = global_1ch.bool()
        if self.loss_preset == "presence_centered" and self.aux_tversky_weight == 0.0:
            loss_tversky = zero
        else:
            t_build = self.tversky(torch.relu(class_pred[:, 0, :, :]),
                                   targets[:, 0, :, :], valid_mask=gm_bool)
            t_veg = self.tversky(torch.relu(class_pred[:, 1, :, :]),
                                 targets[:, 1, :, :], valid_mask=gm_bool)
            t_water = self.tversky(torch.relu(class_pred[:, 2, :, :]),
                                   targets[:, 2, :, :], valid_mask=gm_bool)
            loss_tversky = (t_build + t_veg + t_water) / 3.0

        # --- 4. Height-Aware Building Masking with nDSM-specific mask ---
        # Use height_mask: excludes both global nodata AND nDSM holes
        build_presence_mask = (targets[:, 0, :, :] > 0.1).float() * height_1ch
        veg_presence_mask = (targets[:, 1, :, :] > 0.1).float() * height_1ch
        height_err = self._height_err(preds[:, 3, :, :], targets[:, 3, :, :]) * height_1ch

        height_valid_count = torch.sum(height_1ch) + 1e-6
        per_pixel_weight = (1.0 + self.build_height_boost * build_presence_mask
                            + self.veg_height_boost * veg_presence_mask)
        loss_height_boost = torch.sum(height_err * per_pixel_weight) / height_valid_count

        # --- Combine Total Loss ---
        if self.loss_preset == "presence_centered":
            weighted_mae = self.fraction_mae_weight * loss_fraction_mae
        else:
            weighted_mae = self.w_mae * loss_mae
        weighted_ssim = self.w_ssim * loss_ssim
        weighted_grad = self.w_grad * loss_grad
        # Under presence_centered, w_structure is reserved for height_boost —
        # the main-fraction Tversky is off unless aux_tversky_weight is set,
        # in which case the explicit aux weight owns the term.
        if self.loss_preset == "presence_centered":
            weighted_tversky = self.aux_tversky_weight * loss_tversky
        else:
            weighted_tversky = self.w_structure * loss_tversky
        weighted_height_boost = self.w_structure * loss_height_boost
        total_loss = weighted_mae + weighted_ssim + weighted_grad + \
            weighted_tversky + weighted_height_boost

        # --- 5. Auxiliary supervision for the dual-head model ---
        # Presence target is `label > 0` (matches the leaderboard's IoU
        # binarization — see logs/METRIC_PROBE_REPORT.md). Previously 0.5.
        presence_loss = zero
        presence_tversky_loss = zero
        aux_height_building_loss = zero
        aux_height_vegetation_loss = zero
        if aux_outputs is not None and "presence_logits" in aux_outputs:
            presence_target = (targets[:, :3, :, :] > 0).float()

            def masked_presence_bce(logits):
                safe_logits = torch.where(
                    lc_mask.bool(),
                    logits,
                    torch.zeros_like(logits),
                )
                safe_target = torch.where(
                    lc_mask.bool(),
                    presence_target,
                    torch.zeros_like(presence_target),
                )
                loss = F.binary_cross_entropy_with_logits(
                    safe_logits, safe_target, reduction="none"
                )
                return torch.sum(loss * lc_mask) / (torch.sum(lc_mask) + 1e-6)

            presence_loss = masked_presence_bce(aux_outputs["presence_logits"])
            if self.loss_preset == "presence_centered":
                if self.iou_loss_kind == "focal":
                    # Reuses the presence_tversky slot: the same weight knob
                    # (`presence_tversky_weight`) applies — it is effectively
                    # the "IoU-oriented auxiliary term" weight.
                    presence_logits = aux_outputs["presence_logits"]
                    presence_tversky_loss = self._focal_bce_loss(
                        presence_logits, presence_target, lc_mask
                    )
                else:
                    presence_prob = torch.sigmoid(aux_outputs["presence_logits"])
                    p_build = self.tversky(
                        presence_prob[:, 0, :, :],
                        presence_target[:, 0, :, :],
                        valid_mask=gm_bool,
                    )
                    p_veg = self.tversky(
                        presence_prob[:, 1, :, :],
                        presence_target[:, 1, :, :],
                        valid_mask=gm_bool,
                    )
                    p_water = self.tversky(
                        presence_prob[:, 2, :, :],
                        presence_target[:, 2, :, :],
                        valid_mask=gm_bool,
                    )
                    presence_tversky_loss = (p_build + p_veg + p_water) / 3.0

            target_height = targets[:, 3:4, :, :]
            # Aux class-height supervision matches the leaderboard RMSE pixel
            # selection (label > 0).
            bld_height_mask = (targets[:, 0:1, :, :] > 0).float() * height_mask
            veg_height_mask = (targets[:, 1:2, :, :] > 0).float() * height_mask

            def masked_height(pred, target, mask):
                err = self._height_err(pred, target)
                return torch.sum(err * mask) / (torch.sum(mask) + 1e-6)

            if "height_building" in aux_outputs:
                aux_height_building_loss = masked_height(
                    aux_outputs["height_building"], target_height, bld_height_mask
                )
            if "height_vegetation" in aux_outputs:
                aux_height_vegetation_loss = masked_height(
                    aux_outputs["height_vegetation"], target_height, veg_height_mask
                )

            total_loss = total_loss + self.aux_weight * (
                presence_loss + aux_height_building_loss
                + self.aux_veg_weight * aux_height_vegetation_loss
            ) + self.presence_tversky_weight * presence_tversky_loss

        aux_height_loss = (aux_height_building_loss
                           + self.aux_veg_weight * aux_height_vegetation_loss)
        components = {
            "mae": loss_mae,
            "fraction_mae": loss_fraction_mae,
            "ssim": loss_ssim,
            "grad": loss_grad,
            "tversky": loss_tversky,
            "height_boost": loss_height_boost,
            "presence_bce": presence_loss,
            "presence_tversky": presence_tversky_loss,
            "aux_height_building": aux_height_building_loss,
            "aux_height_vegetation": aux_height_vegetation_loss,
            "aux_height": aux_height_loss,
            "weighted_mae": weighted_mae,
            "weighted_ssim": weighted_ssim,
            "weighted_grad": weighted_grad,
            "weighted_tversky": weighted_tversky,
            "weighted_height_boost": weighted_height_boost,
            "weighted_presence_bce": self.aux_weight * presence_loss,
            "weighted_presence_tversky": self.presence_tversky_weight * presence_tversky_loss,
            "weighted_aux_height": self.aux_weight * aux_height_loss,
        }
        return total_loss, components
