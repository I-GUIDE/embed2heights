"""Composite training loss.

A thin orchestrator: every term's math lives in the domain modules
(``height.py`` for height regression / bin-CE, ``segmentation.py`` for the
class-fraction / presence / topology losses). This class just builds the
supervision targets, calls those terms, and weights them into one scalar.
"""

import torch
import torch.nn as nn

from .height import height_bin_ce, height_error, masked_height
from .segmentation import (
    TverskyLoss,
    building_boundary_loss,
    building_ring,
    cl_dice_loss,
    empty_water_topk_penalty,
    masked_presence_bce,
    split_fg_bg_mae,
)


class ImprovedCompositeLoss(nn.Module):
    """Presence-centered composite loss.

    Terms: split fg/bg fraction MAE, structure-boosted height regression,
    presence BCE + Tversky (coverage>thr target) with a building boundary-ring
    upweight, an empty-water top-k penalty, per-class height-bin cross-entropy, a
    building-boundary aux head, and an optional clDice topology loss.
    """

    def __init__(self, weight_presence_tversky=1.0,
                 weight_fraction_mae=0.1, weight_height_boost=2.0,
                 bg_weight=0.05, aux_weight=0.25,
                 build_height_boost=5.0, veg_height_boost=0.0,
                 aux_veg_weight=1.0, height_bin_aux_weight=0.0,
                 height_bin_sigma_bins=1.5, tversky_water_alpha=0.3,
                 water_empty_topk=0, weight_water_empty_topk=0.0,
                 building_boundary_weight=0.0,
                 building_ring_presence_alpha=0.0,
                 building_ring_kernel=5,
                 presence_coverage_threshold=0.1,
                 cl_dice_weight=0.0,
                 cl_dice_iters=5):
        super().__init__()
        # building/veg presence Tversky share alpha=0.3 (recall-leaning); water
        # takes its own alpha (tversky_water_alpha).
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.tversky_water = TverskyLoss(alpha=tversky_water_alpha,
                                         beta=1.0 - tversky_water_alpha)

        self.w_structure = weight_height_boost
        self.bg_weight = bg_weight
        self.aux_weight = aux_weight
        self.presence_tversky_weight = weight_presence_tversky
        self.fraction_mae_weight = weight_fraction_mae
        self.build_height_boost = float(build_height_boost)
        self.veg_height_boost = float(veg_height_boost)
        self.aux_veg_weight = float(aux_veg_weight)
        self.height_bin_aux_weight = float(height_bin_aux_weight)
        self.height_bin_sigma_bins = float(height_bin_sigma_bins)
        self.water_empty_topk = max(0, int(water_empty_topk))
        self.weight_water_empty_topk = float(weight_water_empty_topk)
        self.building_boundary_weight = float(building_boundary_weight)
        self.building_ring_presence_alpha = float(building_ring_presence_alpha)
        ring_kernel = int(building_ring_kernel)
        if ring_kernel % 2 == 0:
            raise ValueError("building_ring_kernel must be odd")
        self.building_ring_kernel = ring_kernel
        # Presence target = (coverage > thr), matching the official GT (~0.10).
        self.presence_coverage_threshold = float(presence_coverage_threshold)
        self.cl_dice_weight = float(cl_dice_weight)
        self.cl_dice_iters = int(cl_dice_iters)

    def forward(self, preds, targets, valid_mask=None):
        """
        preds:      (B,4,H,W), or a dict with preds["out"] + optional aux heads.
        targets:    (B,4,H,W) ground truth (ch0-2 coverage, ch3 normalized height).
        valid_mask: (B,2,H,W): ch0 = global validity, ch1 = height validity.
        """
        aux = preds if isinstance(preds, dict) else None
        if aux is not None:
            preds = aux["out"]
        device = preds.device

        if valid_mask is None:
            valid_mask = torch.ones(preds.shape[0], 2, preds.shape[2], preds.shape[3],
                                    device=device)
        global_mask = valid_mask[:, 0:1, :, :]
        height_mask = valid_mask[:, 1:2, :, :]
        height_1ch = height_mask.squeeze(1)
        gm_bool = global_mask.squeeze(1).bool()
        lc_mask = global_mask.expand(-1, 3, -1, -1)
        zero = torch.zeros((), device=device, dtype=preds.dtype)

        class_pred = aux["fractions"] if (aux is not None and "fractions" in aux) else preds[:, :3, :, :]

        # --- fraction MAE (seg) ---
        loss_fraction_mae = split_fg_bg_mae(class_pred, targets[:, :3, :, :], lc_mask, self.bg_weight)

        # --- structure-boosted height regression ---
        build_presence = (targets[:, 0, :, :] > 0.1).float() * height_1ch
        veg_presence = (targets[:, 1, :, :] > 0.1).float() * height_1ch
        height_err = height_error(preds[:, 3, :, :], targets[:, 3, :, :]) * height_1ch
        per_pixel_weight = (1.0 + self.build_height_boost * build_presence
                            + self.veg_height_boost * veg_presence)
        loss_height_boost = torch.sum(height_err * per_pixel_weight) / (torch.sum(height_1ch) + 1e-6)

        weighted_mae = self.fraction_mae_weight * loss_fraction_mae
        weighted_height_boost = self.w_structure * loss_height_boost
        total_loss = weighted_mae + weighted_height_boost

        # --- presence terms (BCE + Tversky + empty-water + aux height heads) ---
        presence_loss = presence_tversky_loss = water_topk_loss = zero
        aux_height_building_loss = aux_height_vegetation_loss = zero
        if aux is not None and "presence_logits" in aux:
            logits = aux["presence_logits"]
            presence_target = (targets[:, :3, :, :] > self.presence_coverage_threshold).float()

            ring = None
            if self.building_ring_presence_alpha > 0:
                ring = building_ring(presence_target[:, 0:1, :, :], self.building_ring_kernel)
            presence_loss = masked_presence_bce(logits, presence_target, lc_mask,
                                                ring, self.building_ring_presence_alpha)

            prob = torch.sigmoid(logits)
            p_build = self.tversky(prob[:, 0], presence_target[:, 0], valid_mask=gm_bool)
            p_veg = self.tversky(prob[:, 1], presence_target[:, 1], valid_mask=gm_bool)
            p_water = self.tversky_water(prob[:, 2], presence_target[:, 2], valid_mask=gm_bool)
            presence_tversky_loss = (p_build + p_veg + p_water) / 3.0

            if self.water_empty_topk > 0 and self.weight_water_empty_topk > 0:
                water_topk_loss = empty_water_topk_penalty(
                    logits[:, 2:3, :, :], presence_target[:, 2:3, :, :],
                    global_mask, self.water_empty_topk)

            target_height = targets[:, 3:4, :, :]
            bld_hmask = (targets[:, 0:1, :, :] > 0).float() * height_mask
            veg_hmask = (targets[:, 1:2, :, :] > 0).float() * height_mask
            if "height_building" in aux:
                aux_height_building_loss = masked_height(aux["height_building"], target_height, bld_hmask)
            if "height_vegetation" in aux:
                aux_height_vegetation_loss = masked_height(aux["height_vegetation"], target_height, veg_hmask)

            total_loss = total_loss + self.aux_weight * (
                presence_loss + aux_height_building_loss
                + self.aux_veg_weight * aux_height_vegetation_loss
            ) + self.presence_tversky_weight * presence_tversky_loss
            total_loss = total_loss + self.weight_water_empty_topk * water_topk_loss

        # --- per-class height-bin cross-entropy (height) ---
        height_bin_ce_loss = zero
        if (aux is not None and self.height_bin_aux_weight > 0
                and "height_log_bin_centers" in aux):
            centers = aux["height_log_bin_centers"]
            th = targets[:, 3:4, :, :]
            bld_hmask = (targets[:, 0:1, :, :] > 0).float() * height_mask
            veg_hmask = (targets[:, 1:2, :, :] > 0).float() * height_mask
            sig = self.height_bin_sigma_bins
            ce_base = height_bin_ce(aux.get("height_base_logits"), th, height_mask, centers, sigma_bins=sig)
            ce_bld = height_bin_ce(aux.get("height_building_logits"), th, bld_hmask, centers, sigma_bins=sig)
            ce_veg = height_bin_ce(aux.get("height_vegetation_logits"), th, veg_hmask, centers, sigma_bins=sig)
            height_bin_ce_loss = ce_base + ce_bld + self.aux_veg_weight * ce_veg
            total_loss = total_loss + self.height_bin_aux_weight * height_bin_ce_loss

        # --- building-boundary aux head (seg) ---
        boundary_loss = zero
        if (aux is not None and self.building_boundary_weight > 0
                and "building_boundary_logits" in aux):
            boundary_loss = building_boundary_loss(
                aux["building_boundary_logits"], targets[:, :3, :, :], global_mask)
            total_loss = total_loss + self.building_boundary_weight * boundary_loss

        # --- clDice topology loss on building presence (seg) ---
        cl_dice = zero
        if self.cl_dice_weight > 0:
            bld_gt = (targets[:, 0:1, :, :] > 0.1).to(class_pred.dtype)
            cl_dice = cl_dice_loss(class_pred[:, 0:1, :, :], bld_gt, global_mask, self.cl_dice_iters)
            total_loss = total_loss + self.cl_dice_weight * cl_dice

        aux_height_loss = aux_height_building_loss + self.aux_veg_weight * aux_height_vegetation_loss
        components = {
            "fraction_mae": loss_fraction_mae,
            "height_boost": loss_height_boost,
            "presence_bce": presence_loss,
            "presence_tversky": presence_tversky_loss,
            "water_empty_topk": water_topk_loss,
            "aux_height_building": aux_height_building_loss,
            "aux_height_vegetation": aux_height_vegetation_loss,
            "aux_height": aux_height_loss,
            "height_bin_ce": height_bin_ce_loss,
            "building_boundary": boundary_loss,
            "cl_dice": cl_dice,
            "weighted_cl_dice": self.cl_dice_weight * cl_dice,
            "weighted_mae": weighted_mae,
            "weighted_height_boost": weighted_height_boost,
            "weighted_presence_bce": self.aux_weight * presence_loss,
            "weighted_presence_tversky": self.presence_tversky_weight * presence_tversky_loss,
            "weighted_water_empty_topk": self.weight_water_empty_topk * water_topk_loss,
            "weighted_aux_height": self.aux_weight * aux_height_loss,
            "weighted_height_bin_ce": self.height_bin_aux_weight * height_bin_ce_loss,
            "weighted_building_boundary": self.building_boundary_weight * boundary_loss,
        }
        return total_loss, components
