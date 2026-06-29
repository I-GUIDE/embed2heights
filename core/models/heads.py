import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import HEIGHT_NORM_CONSTANT, ConvGNAct, _group_count


class QueryShapeRefiner(nn.Module):
    """Semantic mask-classification refiner (MaskFormer-lite, shape priors).

    N learned object queries attend (transformer decoder) to a pooled feature map,
    predict per-query masks via (query-embed . per-pixel-embed), and a learned
    [N->3] linear combine produces a 3-class LOGIT correction map. The combine is
    ZERO-INITIALIZED, so at start the correction is exactly 0 (model == proven base)
    and training can only ADD object-shape corrections to the presence logits.
    This injects object-level shape priors that per-pixel convs can't model, while
    protecting the proven baseline (same safe pattern as the TM-S2 token residual).
    """

    def __init__(self, in_ch, d=128, n_queries=32, depth=2, n_heads=4, pool=32):
        super().__init__()
        self.d = int(d)
        self.pool = int(pool)
        self.pixel_proj = nn.Conv2d(in_ch, self.d, 1)
        self.kv_proj = nn.Conv2d(in_ch, self.d, 1)
        self.queries = nn.Parameter(torch.randn(int(n_queries), self.d) * 0.02)
        layer = nn.TransformerDecoderLayer(
            self.d, n_heads, dim_feedforward=self.d * 2, dropout=0.0,
            batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(depth))
        self.mask_mlp = nn.Sequential(nn.Linear(self.d, self.d), nn.GELU(),
                                      nn.Linear(self.d, self.d))
        self.to_class = nn.Linear(int(n_queries), 3, bias=False)
        nn.init.zeros_(self.to_class.weight)  # zero-init -> 0 correction at start

    def forward(self, x):
        B, C, H, W = x.shape
        pix = self.pixel_proj(x)                                   # [B,d,H,W]
        kv = self.kv_proj(F.adaptive_avg_pool2d(x, self.pool))     # [B,d,p,p]
        kv = kv.flatten(2).transpose(1, 2)                         # [B,p*p,d]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)            # [B,N,d]
        q = self.decoder(q, kv)                                    # [B,N,d]
        q = self.mask_mlp(q)                                       # [B,N,d]
        masks = torch.einsum("bnd,bdhw->bnhw", q, pix)             # [B,N,H,W] logits
        seg = torch.einsum("bnhw,nc->bchw", masks, self.to_class.weight.t())  # [B,3,H,W]
        return seg


class MultiTaskPredictionHead(nn.Module):
    """Metric-aware multi-head predictor (v2).

    Metric-aligned dual-head design (v3, 2026-04-17):
    - Deeper shared trunk (2-layer residual) for genuine multi-task sharing.
    - **Presence head** is an independent classifier on shared features (not
      a 1x1 reparam of the fraction head). Supervised by BCE on `label > 0`,
      it learns a calibrated binary mask whose channel outputs ARE the
      submission's land-cover channels. Aligns directly with the leaderboard
      metric (positive-only IoU at pred > 0.5 vs label > 0 — see
      logs/METRIC_PROBE_REPORT.md).
    - **Fraction head** remains as an auxiliary regressor on soft coverage,
      supervised by MAE/SSIM/Gradient/Tversky. Its output is NOT submitted;
      it stays inside the head to condition the height branch.
    - FiLM conditioning uses the soft `fractions` to give the height branch
      fine-grained coverage information at the feature level.
    - Height deltas are non-negative (softplus), enforcing the physical
      constraint that buildings/vegetation only add above ground.
    - **Submitted height is a presence-gated blend of class specialists**
      (`height_building`, `height_vegetation`) rather than a fraction-weighted
      sum. This aligns with the leaderboard's per-class RMSE mask (`gt_class
      > 0`): each specialist is reliable on its own class's pixels (trained
      that way via aux L1), so the gate routes each pixel to the specialist
      that matches the dominant present class. Background pixels fall back
      to `base_height`.

    Output contract: 4-channel tensor [presence_building, presence_veg,
    presence_water, height]. Channels 0-2 are calibrated probabilities in
    [0, 1] trained with BCE on `label > 0`; threshold 0.5 at inference is the
    natural decision boundary.
    """

    def __init__(self, in_ch, out_channels=4, hidden_ch=None, drop=0.05,
                 presence_extra_ch=0, height_specialist_depth=0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, presence_head_kind="shared",
                 presence_head_depth=1, presence_branch_ch=None,
                 bidirectional_ctask=False, height_blend_mode="presence_gated",
                 dual_presence=False, disable_head_film=False,
                 height_dropout=0.0,
                 use_shape_queries=False, shape_n_queries=32, shape_depth=2):
        super().__init__()
        # When True: skip the FiLM modulation on intermediate height features.
        # Labmate's finding (and our own pattern of veg-height regression when
        # the loss/arch get more complex) suggests FiLM-on-fractions can
        # destabilize height training. Try disabling to see if heights improve.
        self.disable_head_film = bool(disable_head_film)
        if out_channels != 4:
            raise ValueError("MultiTaskPredictionHead assumes 4 output channels")
        if height_gate_source not in {"alpha", "fused"}:
            raise ValueError("height_gate_source must be one of: alpha, fused")
        if height_head_kind not in {"linear", "softbin"}:
            raise ValueError("height_head_kind must be one of: linear, softbin")
        if height_blend_mode not in {"presence_gated", "max"}:
            raise ValueError("height_blend_mode must be one of: presence_gated, max")
        if presence_head_kind not in {"shared", "split_water", "split_all", "shared_split_all"}:
            raise ValueError(
                "presence_head_kind must be one of: shared, split_water, "
                "split_all, shared_split_all"
            )
        hidden_ch = hidden_ch or min(160, max(64, in_ch // 2))
        height_hidden_ch = height_hidden_ch or hidden_ch
        presence_head_depth = max(1, int(presence_head_depth))
        presence_branch_ch = int(presence_branch_ch or hidden_ch)
        self._hidden_ch = hidden_ch
        self.presence_extra_ch = presence_extra_ch
        self.presence_head_kind = presence_head_kind
        self.presence_head_depth = presence_head_depth
        self.presence_branch_ch = presence_branch_ch
        self.height_specialist_depth = int(height_specialist_depth)
        self.height_gate_source = height_gate_source
        self.height_hidden_ch = int(height_hidden_ch)
        self.height_trunk_depth = int(height_trunk_depth)
        self.height_independent_branches = bool(height_independent_branches)
        self.height_head_kind = height_head_kind
        self.height_n_bins = int(height_n_bins) if height_head_kind == "softbin" else 0
        self.height_bin_max_m = float(height_bin_max_m)
        self.bidirectional_ctask = bool(bidirectional_ctask)
        self.height_blend_mode = height_blend_mode
        self.dual_presence = bool(dual_presence)
        # Two-stage "purify" gradient cut. s=1.0 → fully coupled (default; the
        # forward stays byte-identical to the pre-existing path). s<1.0 scales the
        # gradient that segmentation/presence losses send back to the shared
        # backbone features: at s=0 the presence pathway sees x.detach(), so only
        # the height pathway re-tunes the backbone (Stage 2 of the two-stage
        # recipe). Set at runtime via model.set_presence_grad_scale().
        self.presence_grad_scale = 1.0
        # Symmetric knob for the Stage-3 "seg-purify": s<1.0 scales the gradient
        # the HEIGHT/FiLM/fraction path sends to the backbone; at s=0 the height
        # path sees x.detach() so only segmentation re-tunes the backbone (mirror
        # of presence_grad_scale). 1.0 = no-op. (labmate's seg-purify = +5th place)
        self.height_grad_scale = 1.0
        # Height-path-only dropout: regularizes the height trunk so it can't
        # memorize local (train-region) height distributions → better test/LB
        # height generalization (heights are the biggest local→public leak).
        # Applied to the FiLM'd features feeding the height trunk only; the
        # presence path is untouched. 0.0 = off (no behavior change).
        self.height_dropout = (
            nn.Dropout2d(float(height_dropout)) if float(height_dropout) > 0
            else nn.Identity()
        )
        # Shape-query refiner (zero-init residual on presence logits)
        self.shape_refiner = (
            QueryShapeRefiner(hidden_ch, n_queries=int(shape_n_queries), depth=int(shape_depth))
            if bool(use_shape_queries) else None
        )

        # --- Deeper shared trunk: 2 layers + residual ---
        self.shared = nn.Sequential(
            ConvGNAct(in_ch, hidden_ch, kernel_size=3),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
        )
        self.shared_res = nn.Sequential(
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_ch), hidden_ch),
        )
        self.shared_act = nn.GELU()

        # --- Bidirectional cross-task gate: height trunk → presence features ---
        height_hidden_ch = height_hidden_ch or hidden_ch
        if self.bidirectional_ctask:
            self.height_to_pres_gate = nn.Conv2d(int(height_hidden_ch), hidden_ch, 1)
            nn.init.zeros_(self.height_to_pres_gate.weight)
            nn.init.zeros_(self.height_to_pres_gate.bias)
        else:
            self.height_to_pres_gate = None

        # --- Fraction head (auxiliary: soft coverage regression) ---
        self.fraction_head = nn.Sequential(
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
            nn.Conv2d(hidden_ch, 3, 1),
        )

        # --- Presence head (main: binary classifier for submission channels 0-2) ---
        def _presence_head(out_ch, in_ch=hidden_ch):
            layers = [ConvGNAct(in_ch, presence_branch_ch, kernel_size=3)]
            layers.extend(
                ConvGNAct(presence_branch_ch, presence_branch_ch, kernel_size=3)
                for _ in range(presence_head_depth - 1)
            )
            layers.append(nn.Conv2d(presence_branch_ch, out_ch, 1))
            return nn.Sequential(*layers)

        def _presence_leaf(out_ch):
            layers = [
                ConvGNAct(presence_branch_ch, presence_branch_ch, kernel_size=3)
                for _ in range(presence_head_depth - 1)
            ]
            layers.append(nn.Conv2d(presence_branch_ch, out_ch, 1))
            return nn.Sequential(*layers)

        if self.presence_head_kind == "shared":
            self.presence_head = _presence_head(3)
        elif self.presence_head_kind == "split_water":
            self.presence_head = nn.ModuleDict({
                "bt": _presence_head(2),
                "water": _presence_head(1),
            })
        elif self.presence_head_kind == "shared_split_all":
            self.presence_shared = ConvGNAct(
                hidden_ch, presence_branch_ch, kernel_size=3
            )
            self.presence_head = nn.ModuleDict({
                "building": _presence_leaf(1),
                "tree": _presence_leaf(1),
                "water": _presence_leaf(1),
            })
        else:
            self.presence_head = nn.ModuleDict({
                "building": _presence_head(1),
                "tree": _presence_head(1),
                "water": _presence_head(1),
            })
        if presence_extra_ch > 0:
            def _presence_delta(out_ch):
                layer = nn.Conv2d(presence_extra_ch, out_ch, 1)
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                return layer

            if self.presence_head_kind == "shared":
                self.presence_delta_head = _presence_delta(3)
            elif self.presence_head_kind == "split_water":
                self.presence_delta_head = nn.ModuleDict({
                    "bt": _presence_delta(2),
                    "water": _presence_delta(1),
                })
            else:
                self.presence_delta_head = nn.ModuleDict({
                    "building": _presence_delta(1),
                    "tree": _presence_delta(1),
                    "water": _presence_delta(1),
                })
        else:
            self.presence_delta_head = None

        # --- Optional: parallel auxiliary presence branch (T-SwinUNet style) ---
        # Shallower than the main branch (single 3x3 + 1x1) — meant to
        # regularize the main head via an IoU-consistency loss.
        if self.dual_presence:
            self.presence_head_aux = nn.Sequential(
                ConvGNAct(hidden_ch, presence_branch_ch, kernel_size=3),
                nn.Conv2d(presence_branch_ch, 3, 1),
            )
        else:
            self.presence_head_aux = None

        # --- FiLM conditioning: soft fractions modulate height features ---
        self.film_scale = nn.Conv2d(3, hidden_ch, 1)
        self.film_shift = nn.Conv2d(3, hidden_ch, 1)

        # --- Height trunk + 3 lightweight output projections ---
        def _height_trunk():
            depth = max(0, self.height_trunk_depth)
            if depth <= 0:
                if hidden_ch == self.height_hidden_ch:
                    return nn.Identity()
                return ConvGNAct(hidden_ch, self.height_hidden_ch, kernel_size=1)

            layers = [ConvGNAct(hidden_ch, self.height_hidden_ch, kernel_size=3)]
            layers.extend(
                ConvGNAct(self.height_hidden_ch, self.height_hidden_ch, kernel_size=3)
                for _ in range(depth - 1)
            )
            return nn.Sequential(*layers)

        # Output channels per height projection. Soft-bin produces K logits per
        # pixel; the legacy linear path produces a single height value.
        proj_out_ch = self.height_n_bins if self.height_head_kind == "softbin" else 1

        def _specialist_head(depth):
            # depth=0 preserves the original single 1x1 projection.
            if depth <= 0:
                return nn.Conv2d(self.height_hidden_ch, proj_out_ch, 1)
            layers = [
                ConvGNAct(self.height_hidden_ch, self.height_hidden_ch, kernel_size=3)
                for _ in range(depth)
            ]
            layers.append(nn.Conv2d(self.height_hidden_ch, proj_out_ch, 1))
            return nn.Sequential(*layers)

        if self.height_independent_branches:
            self.height_base_trunk = _height_trunk()
            self.height_building_trunk = _height_trunk()
            self.height_vegetation_trunk = _height_trunk()
            self.height_trunk = None
        else:
            self.height_trunk = _height_trunk()
            self.height_base_trunk = None
            self.height_building_trunk = None
            self.height_vegetation_trunk = None

        self.height_base_proj = nn.Conv2d(self.height_hidden_ch, proj_out_ch, 1)
        # Names retain the historical "_delta_" segment for checkpoint compat
        # with the linear head; under softbin they output absolute heights, not
        # deltas, but the parameter shape (K logits) is the same shape across
        # base/building/vegetation so the architecture is symmetric.
        self.height_building_delta_proj = _specialist_head(self.height_specialist_depth)
        self.height_vegetation_delta_proj = _specialist_head(self.height_specialist_depth)

        if self.height_head_kind == "softbin":
            # Log-spaced bin centers, evenly partitioning log1p(meters) over
            # [0, log1p(bin_max_m)]. expm1 brings them back to meters; dividing
            # by HEIGHT_NORM_CONSTANT puts them in the model's normalized space
            # so expectation matches the targets used by the loss.
            log_max = math.log1p(self.height_bin_max_m)
            log_edges = torch.linspace(0.0, log_max, self.height_n_bins + 1)
            log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
            centers_m = torch.expm1(log_centers)
            centers_norm = centers_m / HEIGHT_NORM_CONSTANT
            self.register_buffer("height_log_bin_centers", log_centers, persistent=False)
            self.register_buffer("height_bin_centers_norm", centers_norm, persistent=False)
        else:
            self.height_log_bin_centers = None
            self.height_bin_centers_norm = None

    def _forward_presence_head(self, x):
        if self.presence_head_kind == "shared":
            return self.presence_head(x)
        if self.presence_head_kind == "split_water":
            bt = self.presence_head["bt"](x)
            water = self.presence_head["water"](x)
            return torch.cat([bt, water], dim=1)
        if self.presence_head_kind == "shared_split_all":
            x = self.presence_shared(x)
        return torch.cat([
            self.presence_head["building"](x),
            self.presence_head["tree"](x),
            self.presence_head["water"](x),
        ], dim=1)

    def _forward_presence_delta(self, presence_extra):
        if self.presence_delta_head is None:
            return None
        if self.presence_head_kind == "shared":
            return self.presence_delta_head(presence_extra)
        if self.presence_head_kind == "split_water":
            bt = self.presence_delta_head["bt"](presence_extra)
            water = self.presence_delta_head["water"](presence_extra)
            return torch.cat([bt, water], dim=1)
        return torch.cat([
            self.presence_delta_head["building"](presence_extra),
            self.presence_delta_head["tree"](presence_extra),
            self.presence_delta_head["water"](presence_extra),
        ], dim=1)

    def forward(self, x, return_aux=False, presence_extra=None,
                water_bypass_x=None):
        # Shared trunk with residual
        x = self.shared(x)
        x = self.shared_act(x + self.shared_res(x))

        # Stage-2 "purify" gradient cut: optionally detach the backbone from
        # seg/presence-loss gradients (see self.presence_grad_scale). The height
        # pathway keeps using full-gradient `x`, so only height re-tunes the
        # backbone. Identity when s == 1.0 (default).
        s_pg = self.presence_grad_scale
        x_seg = x if s_pg == 1.0 else (s_pg * x + (1.0 - s_pg) * x.detach())
        # Seg-purify mirror: cut the height path's gradient to the backbone.
        s_hg = self.height_grad_scale
        x_hgt = x if s_hg == 1.0 else (s_hg * x + (1.0 - s_hg) * x.detach())

        # Optional water-only bypass: run the same shared trunk weights on a
        # parallel feature that did not see the TerraMind cross-level adapter,
        # then route only the water presence branch through it. Building, tree,
        # height, fraction, FiLM, and Tessera-residual paths stay on `x`.
        bypass_h = None
        if water_bypass_x is not None and self.presence_head_kind == "split_all":
            wb = (water_bypass_x if s_pg == 1.0
                  else s_pg * water_bypass_x + (1.0 - s_pg) * water_bypass_x.detach())
            bypass_h = self.shared(wb)
            bypass_h = self.shared_act(bypass_h + self.shared_res(bypass_h))

        # Auxiliary soft fraction (for height gating + regression losses).
        # Routed through x_hgt so the height/fraction path can be grad-cut from
        # the backbone in the seg-purify stage.
        fraction_logits = self.fraction_head(x_hgt)
        fractions = torch.sigmoid(fraction_logits)

        # FiLM conditioning uses soft fractions (fine-grained coverage signal).
        # Optionally bypass to test if heights improve without modulation.
        if self.disable_head_film:
            h = x_hgt
        else:
            scale = self.film_scale(fractions)
            shift = self.film_shift(fractions)
            h = x_hgt * (1.0 + scale) + shift

        # Height-path regularization (off by default): see self.height_dropout.
        h = self.height_dropout(h)

        if self.height_independent_branches:
            h_base = self.height_base_trunk(h)
            h_building = self.height_building_trunk(h)
            h_vegetation = self.height_vegetation_trunk(h)
            h_shared = h_base
        else:
            h_shared = self.height_trunk(h)
            h_base = h_shared
            h_building = h_shared
            h_vegetation = h_shared

        # Bidirectional cross-task: height trunk features gate presence input
        # F_presence ← x * (0.5 + σ(Conv1×1(h_height))), zero-init → identity at start
        x_pres = x_seg
        if self.height_to_pres_gate is not None:
            gate_src = (h_shared if s_pg == 1.0
                        else s_pg * h_shared + (1.0 - s_pg) * h_shared.detach())
            x_pres = x_seg * (0.5 + torch.sigmoid(self.height_to_pres_gate(gate_src)))

        # Main presence classifier (submission channels 0-2).
        if bypass_h is not None:
            alpha_presence_logits = torch.cat([
                self.presence_head["building"](x_pres),
                self.presence_head["tree"](x_pres),
                self.presence_head["water"](bypass_h),
            ], dim=1)
        else:
            alpha_presence_logits = self._forward_presence_head(x_pres)
        if presence_extra is not None:
            if self.presence_delta_head is None:
                raise ValueError("presence_extra was provided but this head has no residual branch")
            presence_delta_logits = self._forward_presence_delta(presence_extra)
            presence_logits = alpha_presence_logits + presence_delta_logits
        else:
            presence_delta_logits = None
            presence_logits = alpha_presence_logits
        # Shape-query refiner: zero-init residual injecting object-shape priors.
        if self.shape_refiner is not None:
            presence_logits = presence_logits + self.shape_refiner(x_pres)
        presence_prob = torch.sigmoid(presence_logits)

        # Auxiliary parallel presence branch (consistency-regularized; T-SwinUNet idea).
        # Runs on the bare shared trunk features `x` (no FiLM, no bidir gate, no
        # Tessera residual) so it gives a maximally diverse view of the same scene.
        presence_logits_aux = (
            self.presence_head_aux(x_seg) if self.presence_head_aux is not None else None
        )

        base_logits = self.height_base_proj(h_base)
        building_logits = self.height_building_delta_proj(h_building)
        vegetation_logits = self.height_vegetation_delta_proj(h_vegetation)

        if self.height_head_kind == "softbin":
            # Softmax over K log-spaced bin centers, take expectation. Centers
            # are non-negative so each output height is non-negative without
            # softplus. Absolute (not delta) heights per class — the bin-CE
            # aux loss in core.losses forces commitment to the correct bin
            # rather than letting the expectation collapse to a safe mean.
            centers = self.height_bin_centers_norm.view(1, -1, 1, 1)
            base_height = (F.softmax(base_logits, dim=1) * centers).sum(dim=1, keepdim=True)
            building_height = (F.softmax(building_logits, dim=1) * centers).sum(dim=1, keepdim=True)
            vegetation_height = (F.softmax(vegetation_logits, dim=1) * centers).sum(dim=1, keepdim=True)
        else:
            base_height = F.softplus(base_logits, threshold=20.0)
            # Deltas are non-negative: buildings/vegetation only add height
            building_delta = F.softplus(building_logits, threshold=20.0)
            vegetation_delta = F.softplus(vegetation_logits, threshold=20.0)
            building_height = base_height + building_delta
            vegetation_height = base_height + vegetation_delta

        # Presence-gated specialist selection for the single submitted height.
        # Rationale: leaderboard's per-class RMSE masks pixels by `gt_class > 0`,
        # which matches the presence head's supervision. `height_building` /
            # `height_vegetation` are L1-trained on their class mask (core.losses),
        # so each is reliable ONLY on that class's pixels. We therefore route
        # each pixel to its relevant specialist by presence, and fall back to
        # `base_height` on background pixels.
        height_gate_logits = (
            presence_logits
            if self.height_gate_source == "fused"
            else alpha_presence_logits
        )
        height_presence_prob = torch.sigmoid(height_gate_logits)
        if self.height_blend_mode == "max":
            # Decoupled from presence: take the max of class specialists.
            # Background falls back to base_height (deltas → 0). Robust to
            # presence-head miscalibration: pixel routing follows whichever
            # specialist actually predicts a tall structure, not the gate.
            height = torch.maximum(
                torch.maximum(building_height, vegetation_height),
                base_height,
            )
        else:
            p_b = height_presence_prob[:, 0:1, :, :]
            p_v = height_presence_prob[:, 1:2, :, :]
            p_fg = 1.0 - (1.0 - p_b) * (1.0 - p_v)           # P(any of {b,v} present)
            denom = p_b + p_v + 1e-6
            w_b = p_b / denom
            w_v = p_v / denom
            h_fg = w_b * building_height + w_v * vegetation_height
            height = p_fg * h_fg + (1.0 - p_fg) * base_height

        # Submission: channels 0-2 are presence_prob (binary-aligned),
        # channel 3 is the presence-gated specialist height.
        out = torch.cat([presence_prob, height], dim=1)

        if not return_aux:
            return out
        aux = {
            "out": out,
            "fraction_logits": fraction_logits,
            "fractions": fractions,
            "presence_logits": presence_logits,
            "presence_prob": presence_prob,
            "alpha_presence_logits": alpha_presence_logits,
            "alpha_presence_prob": height_presence_prob,
            "presence_delta_logits": presence_delta_logits,
            "presence_logits_aux": presence_logits_aux,
            "height_base": base_height,
            "height_building": building_height,
            "height_vegetation": vegetation_height,
        }
        if self.height_head_kind == "softbin":
            aux["height_base_logits"] = base_logits
            aux["height_building_logits"] = building_logits
            aux["height_vegetation_logits"] = vegetation_logits
            aux["height_log_bin_centers"] = self.height_log_bin_centers
            aux["height_bin_centers_norm"] = self.height_bin_centers_norm
        return aux
