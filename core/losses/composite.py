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
                 height_loss_kind="l1", huber_delta=1.0,
                 build_height_boost=5.0, veg_height_boost=0.0,
                 aux_veg_weight=1.0, height_bin_aux_weight=0.0,
                 height_bin_sigma_bins=1.5,
                 dual_presence_consistency_weight=0.0,
                 ae_only_deep_sup_weight=0.0,
                 building_smooth_weight=0.0,
                 building_smooth_erode_px=1,
                 building_smooth_thr=0.0,
                 vegetation_smooth_weight=0.0,
                 vegetation_smooth_erode_px=1,
                 vegetation_smooth_thr=0.0,
                 building_presence_tversky_weight=1.0,
                 building_presence_bce_weight=1.0):
        super().__init__()
        if loss_preset != "presence_centered":
            raise ValueError("loss_preset must be presence_centered")
        if height_loss_kind not in {"l1", "huber", "mse", "berhu"}:
            raise ValueError("height_loss_kind must be one of: l1, huber, mse, berhu")
        self.loss_preset = loss_preset
        self.ssim = SSIMLoss(window_size=11)
        self.gdl = GradientDifferenceLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

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
        self.build_height_boost = float(build_height_boost)
        self.veg_height_boost = float(veg_height_boost)
        self.aux_veg_weight = float(aux_veg_weight)
        self.height_bin_aux_weight = float(height_bin_aux_weight)
        self.height_bin_sigma_bins = float(height_bin_sigma_bins)
        self.dual_presence_consistency_weight = float(dual_presence_consistency_weight)
        self.ae_only_deep_sup_weight = float(ae_only_deep_sup_weight)
        self.building_smooth_weight = float(building_smooth_weight)
        self.building_smooth_erode_px = int(building_smooth_erode_px)
        self.building_smooth_thr = float(building_smooth_thr)
        self.vegetation_smooth_weight = float(vegetation_smooth_weight)
        self.vegetation_smooth_erode_px = int(vegetation_smooth_erode_px)
        self.vegetation_smooth_thr = float(vegetation_smooth_thr)
        self.building_presence_tversky_weight = float(building_presence_tversky_weight)
        self.building_presence_bce_weight = float(building_presence_bce_weight)

        self.task = "both"
        self.train_presence = True
        self.train_height = True

        self.mae_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).float()
        self.fraction_mae_weights = torch.tensor([1.0, 1.0, 1.0]).float()

    def _height_err(self, pred, target, valid_mask=None):
        return height_error(
            pred,
            target,
            kind=self.height_loss_kind,
            huber_delta=self.huber_delta,
            valid_mask=valid_mask,
        )

    def _height_bin_ce(self, logits, target_norm, mask, log_centers):
        return height_bin_ce(
            logits,
            target_norm,
            mask,
            log_centers,
            sigma_bins=self.height_bin_sigma_bins,
        )

    @staticmethod
    def _eroded_interior(class_target_2d, thr, erode_px, validity_2d):
        # class_target_2d: (B, H, W) ground-truth fraction for a single class.
        # validity_2d: (B, H, W) where height supervision is valid.
        raw = (class_target_2d > thr).float()
        if erode_px > 0:
            k = 2 * erode_px + 1
            inv = (1.0 - raw).unsqueeze(1)
            dilated_inv = F.max_pool2d(inv, kernel_size=k, stride=1, padding=erode_px)
            eroded = (1.0 - dilated_inv).squeeze(1)
        else:
            eroded = raw
        return eroded * validity_2d

    @staticmethod
    def _interior_height_tv(height_pred_2d, mask_2d):
        # height_pred_2d, mask_2d: (B, H, W). TV computed on neighbor-pairs where
        # both endpoints are inside the interior mask, then mean-normalized.
        dh = torch.abs(height_pred_2d[:, :, 1:] - height_pred_2d[:, :, :-1])
        dv = torch.abs(height_pred_2d[:, 1:, :] - height_pred_2d[:, :-1, :])
        mh = torch.minimum(mask_2d[:, :, 1:], mask_2d[:, :, :-1])
        mv = torch.minimum(mask_2d[:, 1:, :], mask_2d[:, :-1, :])
        num = (dh * mh).sum() + (dv * mv).sum()
        den = mh.sum() + mv.sum() + 1e-6
        return num / den

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
            t_water = self.tversky(torch.relu(class_pred[:, 2, :, :]),
                                   targets[:, 2, :, :], valid_mask=gm_bool)
            loss_tversky = (t_build + t_veg + t_water) / 3.0

        build_presence_mask = (targets[:, 0, :, :] > 0.1).float() * height_1ch
        veg_presence_mask = (targets[:, 1, :, :] > 0.1).float() * height_1ch
        height_err = self._height_err(
            preds[:, 3, :, :], targets[:, 3, :, :], valid_mask=height_1ch
        ) * height_1ch

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

        if not self.train_presence:
            weighted_mae = zero
            weighted_ssim = zero
            weighted_grad = zero
            weighted_tversky = zero
        if not self.train_height:
            weighted_height_boost = zero

        total_loss = weighted_mae + weighted_ssim + weighted_grad + \
            weighted_tversky + weighted_height_boost

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
                # Per-channel weighting: emphasize building if requested.
                ch_weights = torch.tensor(
                    [self.building_presence_bce_weight, 1.0, 1.0],
                    device=loss.device,
                    dtype=loss.dtype,
                ).view(1, 3, 1, 1)
                weighted_loss = loss * lc_mask * ch_weights
                # Normalize by the unweighted valid count so the per-class
                # building emphasis raises the overall loss magnitude.
                return torch.sum(weighted_loss) / (torch.sum(lc_mask) + 1e-6)

            presence_loss = masked_presence_bce(aux_outputs["presence_logits"])
            if self.loss_preset == "presence_centered":
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
                # Asymmetric per-class weighting: scale building contribution
                # to push building recall when iou_bld is the bottleneck.
                b_w = self.building_presence_tversky_weight
                presence_tversky_loss = (b_w * p_build + p_veg + p_water) / 3.0

            target_height = targets[:, 3:4, :, :]
            bld_height_mask = (targets[:, 0:1, :, :] > 0).float() * height_mask
            veg_height_mask = (targets[:, 1:2, :, :] > 0).float() * height_mask

            def masked_height(pred, target, mask):
                err = self._height_err(pred, target, valid_mask=mask)
                return torch.sum(err * mask) / (torch.sum(mask) + 1e-6)

            if "height_building" in aux_outputs:
                aux_height_building_loss = masked_height(
                    aux_outputs["height_building"], target_height, bld_height_mask
                )
            if "height_vegetation" in aux_outputs:
                aux_height_vegetation_loss = masked_height(
                    aux_outputs["height_vegetation"], target_height, veg_height_mask
                )

            presence_loss_active = presence_loss if self.train_presence else zero
            presence_tversky_active = (
                presence_tversky_loss if self.train_presence else zero
            )
            aux_height_building_active = (
                aux_height_building_loss if self.train_height else zero
            )
            aux_height_vegetation_active = (
                aux_height_vegetation_loss if self.train_height else zero
            )

            total_loss = total_loss + self.aux_weight * (
                presence_loss_active + aux_height_building_active
                + self.aux_veg_weight * aux_height_vegetation_active
            ) + self.presence_tversky_weight * presence_tversky_active

        # ── Dual-presence consistency (T-SwinUNet style) ─────────────────
        # When the head exposes a parallel auxiliary presence branch, supervise
        # it with the same BCE+Tversky AND add a soft-IoU consistency term
        # between sigmoid(main) and sigmoid(aux). Regularizes the main head by
        # forcing it to agree with a less-equipped sibling.
        dual_aux_bce_loss = zero
        dual_aux_tversky_loss = zero
        dual_consistency_loss = zero
        if (aux_outputs is not None
                and aux_outputs.get("presence_logits_aux") is not None
                and self.train_presence):
            aux_logits = aux_outputs["presence_logits_aux"]
            main_logits = aux_outputs["presence_logits"]

            # Supervise the aux branch with the same losses as main.
            dual_aux_bce_loss = masked_presence_bce(aux_logits)
            aux_prob_full = torch.sigmoid(aux_logits)
            ap_b = self.tversky(aux_prob_full[:, 0, :, :], presence_target[:, 0, :, :], valid_mask=gm_bool)
            ap_v = self.tversky(aux_prob_full[:, 1, :, :], presence_target[:, 1, :, :], valid_mask=gm_bool)
            ap_w = self.tversky(aux_prob_full[:, 2, :, :], presence_target[:, 2, :, :], valid_mask=gm_bool)
            dual_aux_tversky_loss = (ap_b + ap_v + ap_w) / 3.0

            # Soft-IoU consistency between main and aux, averaged over classes.
            main_prob = torch.sigmoid(main_logits)
            aux_prob = aux_prob_full
            mask = gm_bool.float().unsqueeze(1) if gm_bool.dim() == 3 else gm_bool.float()
            inter = (main_prob * aux_prob * mask).sum(dim=(2, 3))
            mp = (main_prob * mask).sum(dim=(2, 3))
            ap = (aux_prob * mask).sum(dim=(2, 3))
            iou = inter / (mp + ap - inter + 1e-6)
            dual_consistency_loss = (1.0 - iou).mean()

            total_loss = (
                total_loss
                + self.aux_weight * (dual_aux_bce_loss + dual_aux_tversky_loss)
                + self.dual_presence_consistency_weight * dual_consistency_loss
            )

        # ── AE-only deep supervision (CMGFNet L_RGB) ──────────────────────
        # Supervise a parallel light prediction head on pre-fusion AE features
        # with same BCE+Tversky (presence) and the same masked height loss.
        ae_only_bce_loss = zero
        ae_only_tversky_loss = zero
        ae_only_height_loss = zero
        if (aux_outputs is not None
                and aux_outputs.get("ae_only_logits") is not None
                and self.ae_only_deep_sup_weight > 0.0):
            ae_logits_full = aux_outputs["ae_only_logits"]
            ae_pres_logits = ae_logits_full[:, :3, :, :]
            ae_height_logits = ae_logits_full[:, 3:4, :, :]

            if self.train_presence:
                ae_only_bce_loss = masked_presence_bce(ae_pres_logits)
                ae_pres_prob = torch.sigmoid(ae_pres_logits)
                ap_b = self.tversky(ae_pres_prob[:, 0, :, :], presence_target[:, 0, :, :], valid_mask=gm_bool)
                ap_v = self.tversky(ae_pres_prob[:, 1, :, :], presence_target[:, 1, :, :], valid_mask=gm_bool)
                ap_w = self.tversky(ae_pres_prob[:, 2, :, :], presence_target[:, 2, :, :], valid_mask=gm_bool)
                ae_only_tversky_loss = (ap_b + ap_v + ap_w) / 3.0
            if self.train_height:
                # Softplus on logit so heights are non-negative, like main head.
                ae_height_pred = F.softplus(ae_height_logits, threshold=20.0)
                ae_only_height_err = self._height_err(
                    ae_height_pred, target_height, valid_mask=height_mask
                )
                ae_only_height_loss = (ae_only_height_err * height_mask).sum() / (height_mask.sum() + 1e-6)

            total_loss = total_loss + self.ae_only_deep_sup_weight * (
                ae_only_bce_loss + ae_only_tversky_loss + ae_only_height_loss
            )

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
            if self.train_height:
                total_loss = total_loss + self.height_bin_aux_weight * height_bin_ce_loss

        aux_height_loss = (aux_height_building_loss
                           + self.aux_veg_weight * aux_height_vegetation_loss)
        w_pres = self.aux_weight if self.train_presence else 0.0
        w_ptv = self.presence_tversky_weight if self.train_presence else 0.0
        w_axh = self.aux_weight if self.train_height else 0.0
        w_bince = self.height_bin_aux_weight if self.train_height else 0.0

        # ── Interior height-TV regularization (building + vegetation) ──────
        # Penalizes predicted height gradients inside eroded GT class footprints.
        # Buildings: encourages within-structure flatness. Vegetation: encourages
        # local smoothness across the canopy (lighter weight recommended since
        # vegetation height is not truly uniform within a tree-class patch).
        building_smooth_loss = zero
        vegetation_smooth_loss = zero
        if self.train_height and (preds.shape[1] >= 4):
            height_pred_2d = preds[:, 3, :, :]
            if self.building_smooth_weight > 0:
                interior_bld = self._eroded_interior(
                    targets[:, 0, :, :],
                    self.building_smooth_thr,
                    self.building_smooth_erode_px,
                    height_1ch,
                )
                building_smooth_loss = self._interior_height_tv(
                    height_pred_2d, interior_bld
                )
                total_loss = total_loss + self.building_smooth_weight * building_smooth_loss
            if self.vegetation_smooth_weight > 0:
                interior_veg = self._eroded_interior(
                    targets[:, 1, :, :],
                    self.vegetation_smooth_thr,
                    self.vegetation_smooth_erode_px,
                    height_1ch,
                )
                vegetation_smooth_loss = self._interior_height_tv(
                    height_pred_2d, interior_veg
                )
                total_loss = total_loss + self.vegetation_smooth_weight * vegetation_smooth_loss

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
            "height_bin_ce": height_bin_ce_loss,
            "weighted_mae": weighted_mae,
            "weighted_ssim": weighted_ssim,
            "weighted_grad": weighted_grad,
            "weighted_tversky": weighted_tversky,
            "weighted_height_boost": weighted_height_boost,
            "weighted_presence_bce": w_pres * presence_loss,
            "weighted_presence_tversky": w_ptv * presence_tversky_loss,
            "weighted_aux_height": w_axh * aux_height_loss,
            "weighted_height_bin_ce": w_bince * height_bin_ce_loss,
            "dual_aux_bce": dual_aux_bce_loss,
            "dual_aux_tversky": dual_aux_tversky_loss,
            "dual_consistency": dual_consistency_loss,
            "building_smooth": building_smooth_loss,
            "vegetation_smooth": vegetation_smooth_loss,
            "weighted_building_smooth": self.building_smooth_weight * building_smooth_loss,
            "weighted_vegetation_smooth": self.vegetation_smooth_weight * vegetation_smooth_loss,
        }
        return total_loss, components
