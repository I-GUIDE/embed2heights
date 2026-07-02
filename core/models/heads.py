import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.data.datasets import HEIGHT_NORM_CONSTANT
from .backbones import ConvGNAct, _group_count


class MultiTaskPredictionHead(nn.Module):
    """Metric-aware multi-task predictor: presence (ch0-2) + height (ch3).

    - A residual shared trunk feeds a presence classifier and (optionally) an
      auxiliary soft-fraction regressor. With ``split_trunk`` the height path
      gets its own trunk reading the head input directly, so height RMSE and
      presence BCE gradients never touch the same parameters.
    - Presence is a per-class (building/tree/water) binary classifier; its
      sigmoid outputs ARE the submitted land-cover channels (BCE on coverage>thr,
      threshold 0.5 at inference).
    - Height uses a soft-bin head: per class, softmax over K log-spaced bin
      centers, take the expectation (non-negative by construction). Three
      independent branches (base / building / vegetation) are trained with L1 +
      bin-CE; the submitted height is a presence-gated blend of the building and
      vegetation specialists, falling back to ``base`` on background pixels.
    - An optional building-boundary aux head supervises the edge ring (training
      only; does not affect the submitted channels).

    Output: 4-channel [presence_building, presence_veg, presence_water, height].
    """

    def __init__(self, in_ch, out_channels=4, hidden_ch=None, drop=0.05,
                 height_specialist_depth=0, height_hidden_ch=None,
                 height_trunk_depth=2, height_n_bins=64, height_bin_max_m=80.0,
                 use_fraction_aux=True, presence_head_depth=1, presence_branch_ch=None,
                 use_boundary_head=False, presence_tower_depth=0, split_trunk=False,
                 presence_trunk_grad_scale=1.0, height_trunk_grad_scale=1.0):
        super().__init__()
        if out_channels != 4:
            raise ValueError("MultiTaskPredictionHead assumes 4 output channels")
        hidden_ch = hidden_ch or min(160, max(64, in_ch // 2))
        height_hidden_ch = height_hidden_ch or hidden_ch
        presence_head_depth = max(1, int(presence_head_depth))
        presence_branch_ch = int(presence_branch_ch or hidden_ch)
        self.height_specialist_depth = int(height_specialist_depth)
        self.height_hidden_ch = int(height_hidden_ch)
        self.height_trunk_depth = int(height_trunk_depth)
        self.height_n_bins = int(height_n_bins)
        self.height_bin_max_m = float(height_bin_max_m)
        self.use_fraction_aux = bool(use_fraction_aux)
        self.use_boundary_head = bool(use_boundary_head)
        self.presence_tower_depth = max(0, int(presence_tower_depth))
        self.split_trunk = bool(split_trunk)
        # Soft one-way gradient decouple (forward identity). presence side into
        # the shared backbone, and (split_trunk) height side into it. 1.0=coupled,
        # 0.0=detached — the purify stages flip one of these to 0.
        self.presence_trunk_grad_scale = min(1.0, max(0.0, float(presence_trunk_grad_scale)))
        self.height_trunk_grad_scale = min(1.0, max(0.0, float(height_trunk_grad_scale)))

        # --- shared segmentation trunk: 2 layers + residual ---
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

        # --- split-trunk: height path gets its own trunk on the head input ---
        if self.split_trunk:
            self.height_shared = nn.Sequential(
                ConvGNAct(in_ch, hidden_ch, kernel_size=3),
                nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            )
            self.height_shared_res = nn.Sequential(
                ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
                nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(hidden_ch), hidden_ch),
            )
        else:
            self.height_shared = None
            self.height_shared_res = None

        def _task_tower(depth):
            if depth <= 0:
                return nn.Identity()
            return nn.Sequential(*[
                ConvGNAct(hidden_ch, hidden_ch, kernel_size=3) for _ in range(depth)
            ])

        self.presence_tower = _task_tower(self.presence_tower_depth)

        # --- building-boundary aux head (training-only edge supervision) ---
        if self.use_boundary_head:
            self.boundary_head = nn.Sequential(
                ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
                nn.Conv2d(hidden_ch, 1, 1),
            )
        else:
            self.boundary_head = None

        # --- auxiliary soft-fraction head (conditions nothing; aux target) ---
        self.fraction_head = (
            nn.Sequential(
                ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
                nn.Conv2d(hidden_ch, 3, 1),
            )
            if self.use_fraction_aux else None
        )

        # --- presence head: independent per-class binary classifiers ---
        def _presence_head(out_ch):
            layers = [ConvGNAct(hidden_ch, presence_branch_ch, kernel_size=3)]
            layers.extend(
                ConvGNAct(presence_branch_ch, presence_branch_ch, kernel_size=3)
                for _ in range(presence_head_depth - 1)
            )
            layers.append(nn.Conv2d(presence_branch_ch, out_ch, 1))
            return nn.Sequential(*layers)

        self.presence_head = nn.ModuleDict({
            "building": _presence_head(1),
            "tree": _presence_head(1),
            "water": _presence_head(1),
        })

        # --- height trunk(s) + soft-bin projections ---
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

        def _specialist_head(depth):
            if depth <= 0:
                return nn.Conv2d(self.height_hidden_ch, self.height_n_bins, 1)
            layers = [
                ConvGNAct(self.height_hidden_ch, self.height_hidden_ch, kernel_size=3)
                for _ in range(depth)
            ]
            layers.append(nn.Conv2d(self.height_hidden_ch, self.height_n_bins, 1))
            return nn.Sequential(*layers)

        self.height_base_trunk = _height_trunk()
        self.height_building_trunk = _height_trunk()
        self.height_vegetation_trunk = _height_trunk()
        self.height_base_proj = nn.Conv2d(self.height_hidden_ch, self.height_n_bins, 1)
        # Names keep the historical "_delta_" segment for checkpoint compatibility.
        self.height_building_delta_proj = _specialist_head(self.height_specialist_depth)
        self.height_vegetation_delta_proj = _specialist_head(self.height_specialist_depth)

        # Log-spaced bin centers over [0, log1p(bin_max_m)] -> meters -> normalized.
        log_max = math.log1p(self.height_bin_max_m)
        log_edges = torch.linspace(0.0, log_max, self.height_n_bins + 1)
        log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
        centers_norm = torch.expm1(log_centers) / HEIGHT_NORM_CONSTANT
        self.register_buffer("height_log_bin_centers", log_centers, persistent=False)
        self.register_buffer("height_bin_centers_norm", centers_norm, persistent=False)

    def _forward_presence_head(self, x):
        return torch.cat([
            self.presence_head["building"](x),
            self.presence_head["tree"](x),
            self.presence_head["water"](x),
        ], dim=1)

    def _run_seg_trunk(self, feat):
        h = self.shared(feat)
        return self.shared_act(h + self.shared_res(h))

    def _run_height_trunk(self, feat):
        h = self.height_shared(feat)
        return self.shared_act(h + self.height_shared_res(h))

    def forward(self, x, return_aux=False):
        head_input = x
        s = self.presence_trunk_grad_scale
        sh = self.height_trunk_grad_scale

        def _scale_grad(t, g):
            # forward identity: g*t + (1-g)*t.detach() == t
            if g >= 1.0:
                return t
            if g <= 0.0:
                return t.detach()
            return g * t + (1.0 - g) * t.detach()

        def _attenuate(t):
            return _scale_grad(t, s)

        # Segmentation trunk. With split_trunk the presence-gradient cut sits at
        # the trunk input (so the whole seg side stops writing into the backbone);
        # otherwise the cut is at the presence-tower input.
        x = self._run_seg_trunk(_attenuate(head_input) if self.split_trunk else head_input)
        if self.split_trunk:
            x_height = self._run_height_trunk(_scale_grad(head_input, sh))
        else:
            x_height = x
        x_presence = self.presence_tower(x if self.split_trunk else _attenuate(x))

        if self.fraction_head is not None:
            fraction_logits = self.fraction_head(x)
            fractions = torch.sigmoid(fraction_logits)
        else:
            fraction_logits = None
            fractions = None

        presence_logits = self._forward_presence_head(x_presence)
        presence_prob = torch.sigmoid(presence_logits)

        h = x_height
        h_base = self.height_base_trunk(h)
        h_building = self.height_building_trunk(h)
        h_vegetation = self.height_vegetation_trunk(h)
        base_logits = self.height_base_proj(h_base)
        building_logits = self.height_building_delta_proj(h_building)
        vegetation_logits = self.height_vegetation_delta_proj(h_vegetation)

        # Soft-bin: softmax over log-spaced centers, take expectation (>=0).
        centers = self.height_bin_centers_norm.view(1, -1, 1, 1)
        base_height = (F.softmax(base_logits, dim=1) * centers).sum(dim=1, keepdim=True)
        building_height = (F.softmax(building_logits, dim=1) * centers).sum(dim=1, keepdim=True)
        vegetation_height = (F.softmax(vegetation_logits, dim=1) * centers).sum(dim=1, keepdim=True)

        # Presence-gated specialist selection for the single submitted height:
        # route each pixel to its class specialist (their RMSE mask is gt_class>0),
        # fall back to base on background.
        p = torch.sigmoid(presence_logits)
        p_b = p[:, 0:1, :, :]
        p_v = p[:, 1:2, :, :]
        p_fg = 1.0 - (1.0 - p_b) * (1.0 - p_v)
        denom = p_b + p_v + 1e-6
        h_fg = (p_b / denom) * building_height + (p_v / denom) * vegetation_height
        height = p_fg * h_fg + (1.0 - p_fg) * base_height

        out = torch.cat([presence_prob, height], dim=1)
        if not return_aux:
            return out

        aux = {
            "out": out,
            "presence_logits": presence_logits,
            "height_building": building_height,
            "height_vegetation": vegetation_height,
            "height_base_logits": base_logits,
            "height_building_logits": building_logits,
            "height_vegetation_logits": vegetation_logits,
            "height_log_bin_centers": self.height_log_bin_centers,
        }
        if self.boundary_head is not None:
            aux["building_boundary_logits"] = self.boundary_head(x)
        if fractions is not None:
            aux["fractions"] = fractions
        return aux
