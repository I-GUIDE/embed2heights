"""Composite supervised training loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .height import height_bin_ce, height_error
from .segmentation import TverskyLoss
from .structure import GradientDifferenceLoss, SSIMLoss


class ImprovedCompositeLoss(nn.Module):
    """
    Combined Recipe:
    1. Split Foreground/Background MAE (L1).
    2. SSIM and Gradient Loss on RGB channels.
    3. Tversky Loss specifically to boost Building recall.
    4. Structure-Boosted Height MAE.
    """

    def __init__(self, weight_mae=1.0, weight_presence_tversky=1.0,
                 weight_fraction_mae=0.1, weight_height_boost=2.0,
                 bg_weight=0.05, aux_weight=0.25, loss_preset="presence_centered",
                 height_loss_kind="l1", huber_delta=1.0, pinball_tau=0.5,
                 build_height_boost=5.0, veg_height_boost=0.0,
                 aux_veg_weight=1.0, height_bin_aux_weight=0.0,
                 height_bin_sigma_bins=1.5, tversky_building_alpha=0.3,
                 tversky_water_alpha=0.3,
                 water_empty_topk=0, weight_water_empty_topk=0.0,
                 building_presence_pos_weight=1.0,
                 small_building_presence_weight=1.0,
                 small_building_max_pixels=0,
                 building_boundary_weight=0.0,
                 building_ring_presence_alpha=0.0,
                 building_ring_kernel=5,
                 presence_coverage_threshold=0.0):
        super().__init__()
        if loss_preset != "presence_centered":
            raise ValueError("loss_preset must be presence_centered")
        if height_loss_kind not in {"l1", "huber", "mse", "pinball"}:
            raise ValueError("height_loss_kind must be one of: l1, huber, mse, pinball")
        self.loss_preset = loss_preset
        self.ssim = SSIMLoss(window_size=11)
        self.gdl = GradientDifferenceLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        building_alpha = float(tversky_building_alpha)
        self.tversky_building = TverskyLoss(
            alpha=building_alpha,
            beta=1.0 - building_alpha,
        )
        water_alpha = float(tversky_water_alpha)
        self.tversky_water = TverskyLoss(alpha=water_alpha, beta=1.0 - water_alpha)

        self.w_mae = weight_mae
        self.w_ssim = 0.0
        self.w_grad = 0.0
        self.w_structure = weight_height_boost

        self.bg_weight = bg_weight
        self.aux_weight = aux_weight
        self.presence_tversky_weight = weight_presence_tversky
        self.fraction_mae_weight = weight_fraction_mae

        self.height_loss_kind = height_loss_kind
        self.huber_delta = float(huber_delta)
        self.pinball_tau = float(pinball_tau)
        self.build_height_boost = float(build_height_boost)
        self.veg_height_boost = float(veg_height_boost)
        self.aux_veg_weight = float(aux_veg_weight)
        self.height_bin_aux_weight = float(height_bin_aux_weight)
        self.height_bin_sigma_bins = float(height_bin_sigma_bins)
        self.water_empty_topk = max(0, int(water_empty_topk))
        self.weight_water_empty_topk = float(weight_water_empty_topk)
        self.building_presence_pos_weight = float(building_presence_pos_weight)
        self.small_building_presence_weight = float(small_building_presence_weight)
        self.small_building_max_pixels = max(0, int(small_building_max_pixels))
        self.building_boundary_weight = float(building_boundary_weight)
        self.building_ring_presence_alpha = float(building_ring_presence_alpha)
        ring_kernel = int(building_ring_kernel)
        if ring_kernel % 2 == 0:
            raise ValueError("building_ring_kernel must be odd")
        self.building_ring_kernel = ring_kernel
        # >0 aligns presence supervision with the official GT (coverage>thr,
        # ~0.10 reverse-engineered from the public board). 0.0 keeps the legacy
        # argmax+any-present target. See _build_presence_target in forward().
        self.presence_coverage_threshold = float(presence_coverage_threshold)

        self.mae_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).float()
        self.fraction_mae_weights = torch.tensor([1.0, 1.0, 1.0]).float()

    def _height_err(self, pred, target):
        return height_error(
            pred,
            target,
            kind=self.height_loss_kind,
            huber_delta=self.huber_delta,
            pinball_tau=self.pinball_tau,
        )

    def _height_bin_ce(self, logits, target_norm, mask, log_centers):
        return height_bin_ce(
            logits,
            target_norm,
            mask,
            log_centers,
            sigma_bins=self.height_bin_sigma_bins,
        )

    def _building_boundary_loss(self, logits, fractions, mask):
        """Auxiliary edge supervision on the building boundary ring.

        `fractions` is the (B, 3, H, W) GT coverage for {building, veg, water}.
        The hard building mask matches the dataset's binarization convention:
        each pixel is assigned to its single argmax class (and only where some
        class is present), so a pixel counts as building iff building is the
        dominant fraction — NOT merely `building_fraction > 0`. This is the
        same rule used to build `presence_target`.

        The boundary target is then derived on-the-fly: a morphological
        dilation minus erosion (3x3) of that hard mask yields a ~1px-wide ring.
        BCE handles per-pixel error; soft Dice counters the extreme class
        imbalance (boundary pixels are a tiny fraction of the patch). Both
        terms are restricted to valid (`mask`) pixels.
        """
        any_present = (fractions > 0).any(dim=1, keepdim=True).float()
        argmax_idx = fractions.argmax(dim=1, keepdim=True)
        # Building is channel 0; dominant-class assignment matches the labels.
        m = (argmax_idx == 0).float() * any_present
        dil = F.max_pool2d(m, 3, stride=1, padding=1)                 # dilation
        ero = 1.0 - F.max_pool2d(1.0 - m, 3, stride=1, padding=1)     # erosion
        boundary = (dil - ero).clamp(0.0, 1.0)                        # ring = dil - ero

        bce_map = F.binary_cross_entropy_with_logits(
            logits, boundary, reduction="none"
        )
        bce = torch.sum(bce_map * mask) / (torch.sum(mask) + 1e-6)

        pred = torch.sigmoid(logits) * mask
        tgt = boundary * mask
        inter = torch.sum(pred * tgt)
        dice = 1.0 - (2.0 * inter + 1.0) / (torch.sum(pred) + torch.sum(tgt) + 1.0)
        return bce + dice

    def forward(self, preds, targets, valid_mask=None):
        """
        Args:
            preds:      (B, 4, H, W) model predictions, or a dict with
                        preds["out"] plus optional auxiliary heads.
            targets:    (B, 4, H, W) ground truth.
            valid_mask: (B, 2, H, W), channel 0 = global validity,
                        channel 1 = height validity.
        """
        aux_outputs = preds if isinstance(preds, dict) else None
        if aux_outputs is not None:
            preds = aux_outputs["out"]

        device = preds.device
        mae_weights = self.mae_weights.to(device)
        fraction_mae_weights = self.fraction_mae_weights.to(device)

        if valid_mask is None:
            valid_mask = torch.ones(preds.shape[0], 2, preds.shape[2], preds.shape[3],
                                    device=device)

        global_mask = valid_mask[:, 0:1, :, :]
        height_mask = valid_mask[:, 1:2, :, :]
        ch_mask = torch.cat([global_mask.expand(-1, 3, -1, -1), height_mask], dim=1)
        global_1ch = global_mask.squeeze(1)
        height_1ch = height_mask.squeeze(1)

        if aux_outputs is not None and "fractions" in aux_outputs:
            class_pred = aux_outputs["fractions"]
        else:
            class_pred = preds[:, :3, :, :]
        reg_pred = torch.cat([class_pred, preds[:, 3:4, :, :]], dim=1)

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

        loss_mae = split_fg_bg_mae(reg_pred, targets, ch_mask, mae_weights)
        loss_fraction_mae = split_fg_bg_mae(
            class_pred,
            targets[:, :3, :, :],
            global_mask.expand(-1, 3, -1, -1),
            fraction_mae_weights,
        )

        lc_mask = global_mask.expand(-1, 3, -1, -1)
        lc_pred = class_pred * lc_mask
        lc_target = targets[:, :3, :, :] * lc_mask

        zero = torch.zeros((), device=device, dtype=preds.dtype)
        loss_ssim = self.ssim(lc_pred, lc_target) if self.w_ssim != 0 else zero
        loss_grad = self.gdl(lc_pred, lc_target) if self.w_grad != 0 else zero

        gm_bool = global_1ch.bool()
        if self.loss_preset == "presence_centered":
            loss_tversky = zero
        else:
            t_build = self.tversky(torch.relu(class_pred[:, 0, :, :]),
                                   targets[:, 0, :, :], valid_mask=gm_bool)
            t_veg = self.tversky(torch.relu(class_pred[:, 1, :, :]),
                                 targets[:, 1, :, :], valid_mask=gm_bool)
            t_water = self.tversky_water(torch.relu(class_pred[:, 2, :, :]),
                                         targets[:, 2, :, :], valid_mask=gm_bool)
            loss_tversky = (t_build + t_veg + t_water) / 3.0

        build_presence_mask = (targets[:, 0, :, :] > 0.1).float() * height_1ch
        veg_presence_mask = (targets[:, 1, :, :] > 0.1).float() * height_1ch
        height_err = self._height_err(preds[:, 3, :, :], targets[:, 3, :, :]) * height_1ch

        height_valid_count = torch.sum(height_1ch) + 1e-6
        per_pixel_weight = (1.0 + self.build_height_boost * build_presence_mask
                            + self.veg_height_boost * veg_presence_mask)
        loss_height_boost = torch.sum(height_err * per_pixel_weight) / height_valid_count

        if self.loss_preset == "presence_centered":
            weighted_mae = self.fraction_mae_weight * loss_fraction_mae
        else:
            weighted_mae = self.w_mae * loss_mae
        weighted_ssim = self.w_ssim * loss_ssim
        weighted_grad = self.w_grad * loss_grad
        weighted_tversky = self.w_structure * loss_tversky
        weighted_height_boost = self.w_structure * loss_height_boost

        total_loss = weighted_mae + weighted_ssim + weighted_grad + \
            weighted_tversky + weighted_height_boost

        presence_loss = zero
        presence_tversky_loss = zero
        water_empty_topk_loss = zero
        aux_height_building_loss = zero
        aux_height_vegetation_loss = zero
        if aux_outputs is not None and "presence_logits" in aux_outputs:
            fractions = targets[:, :3, :, :]
            if self.presence_coverage_threshold > 0.0:
                # Official GT marks a 10 m pixel positive for a class iff its
                # coverage exceeds a low threshold (~0.10, reverse-engineered
                # from the public board); the legacy argmax+any-present rule
                # below over-counts building ~3x. Per-class (multi-label) is
                # fine: the 3 coverages sum to <=1, so double-positives are rare.
                presence_target = (fractions > self.presence_coverage_threshold).float()
            else:
                any_present = (fractions > 0).any(dim=1, keepdim=True).float()
                argmax_idx = fractions.argmax(dim=1, keepdim=True)
                presence_target = torch.zeros_like(fractions).scatter_(1, argmax_idx, 1.0) * any_present

            # Building boundary ring for presence-BCE upweighting. Same hard
            # mask convention as `_building_boundary_loss` (argmax building),
            # but the ring here is wider (default 5x5 -> ~2px each side) to
            # tolerate GT registration error, and it reweights the MAIN
            # presence loss instead of supervising a separate aux head: the
            # extra pressure lands directly on the logits that decide IoU.
            building_ring = None
            if self.building_ring_presence_alpha > 0:
                m = presence_target[:, 0:1, :, :]
                pad = self.building_ring_kernel // 2
                dil = F.max_pool2d(m, self.building_ring_kernel, stride=1, padding=pad)
                ero = 1.0 - F.max_pool2d(1.0 - m, self.building_ring_kernel,
                                         stride=1, padding=pad)
                building_ring = (dil - ero).clamp(0.0, 1.0)

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
                bce_weight = lc_mask
                if (self.building_presence_pos_weight != 1.0
                        or self.small_building_presence_weight != 1.0):
                    bce_weight = lc_mask.clone()
                    b_pos = safe_target[:, 0:1, :, :] > 0
                    b_weight = torch.ones_like(bce_weight[:, 0:1, :, :])
                    b_weight = torch.where(
                        b_pos,
                        b_weight * self.building_presence_pos_weight,
                        b_weight,
                    )
                    if (self.small_building_max_pixels > 0
                            and self.small_building_presence_weight != 1.0):
                        b_area = torch.sum(
                            safe_target[:, 0:1, :, :] * global_mask,
                            dim=(1, 2, 3),
                        )
                        small = (
                            (b_area > 0)
                            & (b_area <= self.small_building_max_pixels)
                        ).view(-1, 1, 1, 1)
                        b_weight = torch.where(
                            small & b_pos,
                            b_weight * self.small_building_presence_weight,
                            b_weight,
                        )
                    bce_weight[:, 0:1, :, :] = bce_weight[:, 0:1, :, :] * b_weight
                if building_ring is not None:
                    if bce_weight is lc_mask:
                        bce_weight = lc_mask.clone()
                    bce_weight[:, 0:1, :, :] = bce_weight[:, 0:1, :, :] * (
                        1.0 + self.building_ring_presence_alpha * building_ring
                    )
                return torch.sum(loss * bce_weight) / (torch.sum(bce_weight) + 1e-6)

            presence_loss = masked_presence_bce(aux_outputs["presence_logits"])
            if self.loss_preset == "presence_centered":
                presence_prob = torch.sigmoid(aux_outputs["presence_logits"])
                p_build = self.tversky_building(
                    presence_prob[:, 0, :, :],
                    presence_target[:, 0, :, :],
                    valid_mask=gm_bool,
                )
                p_veg = self.tversky(
                    presence_prob[:, 1, :, :],
                    presence_target[:, 1, :, :],
                    valid_mask=gm_bool,
                )
                p_water = self.tversky_water(
                    presence_prob[:, 2, :, :],
                    presence_target[:, 2, :, :],
                    valid_mask=gm_bool,
                )
                presence_tversky_loss = (p_build + p_veg + p_water) / 3.0

            if self.water_empty_topk > 0 and self.weight_water_empty_topk > 0:
                water_target = presence_target[:, 2:3, :, :] * global_mask
                empty_water = torch.sum(water_target, dim=(1, 2, 3)) <= 0
                if torch.any(empty_water):
                    water_prob = torch.sigmoid(
                        aux_outputs["presence_logits"][:, 2:3, :, :]
                    ) * global_mask
                    flat = water_prob[empty_water].flatten(1)
                    if flat.numel() > 0:
                        k = min(self.water_empty_topk, flat.shape[1])
                        water_empty_topk_loss = torch.topk(flat, k, dim=1).values.mean()

            target_height = targets[:, 3:4, :, :]
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
            total_loss = total_loss + self.weight_water_empty_topk * water_empty_topk_loss

        height_bin_ce_loss = zero
        if (aux_outputs is not None
                and self.height_bin_aux_weight > 0
                and "height_log_bin_centers" in aux_outputs):
            log_centers = aux_outputs["height_log_bin_centers"]
            target_height = targets[:, 3:4, :, :]
            bld_height_mask = (targets[:, 0:1, :, :] > 0).float() * height_mask
            veg_height_mask = (targets[:, 1:2, :, :] > 0).float() * height_mask
            ce_base = self._height_bin_ce(
                aux_outputs.get("height_base_logits"),
                target_height, height_mask, log_centers,
            )
            ce_bld = self._height_bin_ce(
                aux_outputs.get("height_building_logits"),
                target_height, bld_height_mask, log_centers,
            )
            ce_veg = self._height_bin_ce(
                aux_outputs.get("height_vegetation_logits"),
                target_height, veg_height_mask, log_centers,
            )
            height_bin_ce_loss = ce_base + ce_bld + self.aux_veg_weight * ce_veg
            total_loss = total_loss + self.height_bin_aux_weight * height_bin_ce_loss

        building_boundary_loss = zero
        if (aux_outputs is not None
                and self.building_boundary_weight > 0
                and "building_boundary_logits" in aux_outputs):
            building_boundary_loss = self._building_boundary_loss(
                aux_outputs["building_boundary_logits"],
                targets[:, :3, :, :],   # land-cover fractions -> argmax building mask
                global_mask,
            )
            total_loss = total_loss + self.building_boundary_weight * building_boundary_loss

        aux_height_loss = (aux_height_building_loss
                           + self.aux_veg_weight * aux_height_vegetation_loss)
        w_pres = self.aux_weight
        w_ptv = self.presence_tversky_weight
        w_axh = self.aux_weight
        w_bince = self.height_bin_aux_weight
        w_wetopk = self.weight_water_empty_topk
        w_bbnd = self.building_boundary_weight

        components = {
            "mae": loss_mae,
            "fraction_mae": loss_fraction_mae,
            "ssim": loss_ssim,
            "grad": loss_grad,
            "tversky": loss_tversky,
            "height_boost": loss_height_boost,
            "presence_bce": presence_loss,
            "presence_tversky": presence_tversky_loss,
            "water_empty_topk": water_empty_topk_loss,
            "aux_height_building": aux_height_building_loss,
            "aux_height_vegetation": aux_height_vegetation_loss,
            "aux_height": aux_height_loss,
            "height_bin_ce": height_bin_ce_loss,
            "building_boundary": building_boundary_loss,
            "weighted_mae": weighted_mae,
            "weighted_ssim": weighted_ssim,
            "weighted_grad": weighted_grad,
            "weighted_tversky": weighted_tversky,
            "weighted_height_boost": weighted_height_boost,
            "weighted_presence_bce": w_pres * presence_loss,
            "weighted_presence_tversky": w_ptv * presence_tversky_loss,
            "weighted_water_empty_topk": w_wetopk * water_empty_topk_loss,
            "weighted_aux_height": w_axh * aux_height_loss,
            "weighted_height_bin_ce": w_bince * height_bin_ce_loss,
            "weighted_building_boundary": w_bbnd * building_boundary_loss,
        }
        return total_loss, components
