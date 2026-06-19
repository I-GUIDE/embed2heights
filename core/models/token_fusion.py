import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ChannelCalibration, ConvGNAct, _group_count
from .backbones import LightUNet, LightUNetPP


def _build_pixel_backbone(kind, in_channels, n_classes, base_ch, norm_kind):
    """Pick the pixel UNet variant: 'unet' (default LightUNet) or 'unetpp' (LightUNetPP)."""
    k = (kind or "unet").lower()
    if k in ("unet", "lightunet"):
        return LightUNet(in_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind)
    if k in ("unetpp", "unet++", "lightunetpp"):
        return LightUNetPP(in_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind)
    raise ValueError(f"Unknown pixel_backbone_kind={kind!r}; expected 'unet' or 'unetpp'.")
from .heads import MultiTaskPredictionHead
from .pixel_fusion import (
    _apply_fusion_gate,
    _build_fusion_gate,
    _maybe_drop_modality,
    _maybe_drop_modality_symmetric,
)


def _make_token_proj(in_ch, out_ch, depth=1):
    if depth <= 1:
        return nn.Conv2d(in_ch, out_ch, 1, bias=False)
    layers = [
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.GroupNorm(_group_count(out_ch), out_ch),
        nn.GELU(),
    ]
    for _ in range(depth - 2):
        layers += [
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.GELU(),
        ]
    layers.append(nn.Conv2d(out_ch, out_ch, 1, bias=False))
    return nn.Sequential(*layers)


class CrossSourceHybridFiLMFusion(nn.Module):
    """xf085 SoTA fusion: cross-source self-attention + per-source FiLM + additive + spatial gate.

    Stage 1 (16x16 token scale): N token sources cross-attend to each other via
    one self-attention layer with learned modality embeddings + 2D positional
    encoding. Output projection is zero-initialised → refined ≈ projected at init.

    Stage 2 (H x W pixel scale): each refined source contributes three zero-init
    residuals applied as
        delta_i = sigmoid(g_i) * (gamma_i * F_pixel + beta_i + A_i)
        F_out   = F_pixel + sum_i delta_i

    All three per-source pathways (FiLM γ/β, additive A_i, spatial gate σ(g_i))
    are always built. Score-based attribution (tools/attribute_token_fusion_score.py)
    showed each off-toggle degraded the leaderboard Score, so they are no longer
    configurable.
    """

    _TOKEN_INPUT_CLAMP = 50.0
    _CTX_CLAMP = 50.0
    _FILM_PARAM_CLAMP = 4.0
    _ADD_CLAMP = 4.0

    def __init__(self, pixel_ch, token_channels, token_source_ch=768,
                 ctx_ch=96, token_calibration=False, token_proj_depth=1,
                 attn_heads=4, attn_dropout=0.05, use_additive=True,
                 token_calibration_source_indices=None,
                 token_in_source_attn=False,
                 token_cross_source_attn=True,
                 token_input_clamp=None):
        super().__init__()
        # When set, overrides _TOKEN_INPUT_CLAMP for the raw post-calib token clamp
        # (L233). Used by xf119 ablation to test whether the ±50 clamp on uncalibrated
        # THOR (raw ±17000) is hurting token information — raising this lets THOR
        # contribute graded magnitude instead of near-binary saturation.
        self.token_input_clamp = (
            float(token_input_clamp)
            if token_input_clamp is not None else self._TOKEN_INPUT_CLAMP
        )
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"CrossSourceHybridFiLMFusion: token_channels={token_channels} must be "
                f"divisible by token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.n_sources = token_channels // token_source_ch
        self.ctx_ch = int(ctx_ch)
        self.attn_heads = int(attn_heads)
        self.use_additive = bool(use_additive)
        self.token_in_source_attn = bool(token_in_source_attn)
        self.token_cross_source_attn = bool(token_cross_source_attn)

        # Selective calibration: when token_calibration_source_indices is given,
        # only those source indices get a learnable ChannelCalibration; others
        # bypass through nn.Identity (forward indexes unchanged). xf095 uses
        # indices=[0,1] (terramind only) — THOR's raw ±17000 scale must NOT be
        # calibrated or iou_wat collapses.
        if token_calibration:
            if token_calibration_source_indices is None:
                calib_set = set(range(self.n_sources))
            else:
                calib_set = {int(i) for i in token_calibration_source_indices}
            self.token_calibs = nn.ModuleList([
                ChannelCalibration(token_source_ch) if i in calib_set
                else nn.Identity()
                for i in range(self.n_sources)
            ])
        else:
            self.token_calibs = None
        self.token_projs = nn.ModuleList([
            _make_token_proj(token_source_ch, ctx_ch, depth=token_proj_depth)
            for _ in range(self.n_sources)
        ])

        if self.token_in_source_attn and self.attn_heads >= 1:
            self.in_source_norms = nn.ModuleList([
                nn.LayerNorm(ctx_ch) for _ in range(self.n_sources)
            ])
            self.in_source_attns = nn.ModuleList([
                nn.MultiheadAttention(
                    ctx_ch, self.attn_heads,
                    dropout=attn_dropout, batch_first=True,
                )
                for _ in range(self.n_sources)
            ])
            for attn in self.in_source_attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)
        else:
            self.in_source_norms = None
            self.in_source_attns = None

        needs_pos = (
            self.attn_heads >= 1
            and (self.token_in_source_attn or self.token_cross_source_attn)
        )
        if needs_pos:
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, ctx_ch),
                nn.GELU(),
                nn.Linear(ctx_ch, ctx_ch),
            )
        else:
            self.pos_mlp = None

        if self.token_cross_source_attn and self.attn_heads >= 1:
            self.modality_embed = nn.Parameter(torch.zeros(self.n_sources, ctx_ch))
            nn.init.normal_(self.modality_embed, std=0.02)
            self.attn_norm = nn.LayerNorm(ctx_ch)
            self.cross_source_attn = nn.MultiheadAttention(
                ctx_ch, self.attn_heads,
                dropout=attn_dropout, batch_first=True,
            )
            nn.init.zeros_(self.cross_source_attn.out_proj.weight)
            nn.init.zeros_(self.cross_source_attn.out_proj.bias)
        else:
            self.modality_embed = None
            self.attn_norm = None
            self.cross_source_attn = None

        self.film_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, pixel_ch * 2, 1) for _ in range(self.n_sources)
        ])
        if self.use_additive:
            self.add_convs = nn.ModuleList([
                nn.Conv2d(ctx_ch, pixel_ch, 1) for _ in range(self.n_sources)
            ])
        else:
            self.add_convs = None
        self.gate_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, 1, 1) for _ in range(self.n_sources)
        ])
        for module_list in (self.film_convs, self.gate_convs):
            for conv in module_list:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
        if self.add_convs is not None:
            for conv in self.add_convs:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

    def _pos_tokens(self, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2)
        return self.pos_mlp(coords)

    def _refine_in_sources(self, ctx_list):
        if self.in_source_attns is None:
            return ctx_list
        b, _, h, w = ctx_list[0].shape
        pos = self._pos_tokens(h, w, ctx_list[0].device, ctx_list[0].dtype)
        refined = []
        for i, ctx in enumerate(ctx_list):
            tokens = ctx.flatten(2).transpose(1, 2)
            tokens = tokens + pos
            x_norm = self.in_source_norms[i](tokens)
            with torch.amp.autocast("cuda", enabled=False):
                attn_out, _ = self.in_source_attns[i](
                    x_norm.float(), x_norm.float(), x_norm.float(),
                    need_weights=False,
                )
            delta = attn_out.to(tokens.dtype).transpose(1, 2).reshape(
                b, self.ctx_ch, h, w
            )
            refined.append(ctx + delta)
        return refined

    def _refine_sources(self, ctx_list):
        if self.cross_source_attn is None:
            return ctx_list
        b, _, h, w = ctx_list[0].shape
        pos = self._pos_tokens(h, w, ctx_list[0].device, ctx_list[0].dtype)
        seqs = []
        for i, ctx in enumerate(ctx_list):
            tokens = ctx.flatten(2).transpose(1, 2)
            tokens = tokens + pos
            tokens = tokens + self.modality_embed[i].view(1, 1, -1)
            seqs.append(tokens)
        x = torch.cat(seqs, dim=1)

        x_norm = self.attn_norm(x)
        with torch.amp.autocast("cuda", enabled=False):
            attn_out, _ = self.cross_source_attn(
                x_norm.float(), x_norm.float(), x_norm.float(),
                need_weights=False,
            )
        # Zero-init out_proj => attn_out ~ 0 at init => refined ~ ctx_list[i]
        chunks = attn_out.to(x.dtype).split(h * w, dim=1)
        refined = []
        for i, ck in enumerate(chunks):
            delta = ck.transpose(1, 2).reshape(b, self.ctx_ch, h, w)
            refined.append(ctx_list[i] + delta)
        return refined

    def forward(self, F_pixel, token):
        """
        F_pixel: (B, pixel_ch, H, W)
        token:   (B, n_sources * token_source_ch, h, w)  [e.g. 4x768 at 16x16]
        returns: (B, pixel_ch, H, W)
        """
        H, W = F_pixel.shape[-2:]
        parts = token.float().split(self.token_source_ch, dim=1)
        with torch.amp.autocast("cuda", enabled=False):
            ctx_list = []
            for i, src in enumerate(parts):
                if self.token_calibs is not None:
                    src = self.token_calibs[i](src)
                src = src.clamp(-self.token_input_clamp, self.token_input_clamp)
                ctx = self.token_projs[i](src).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
                ctx_list.append(ctx)

            ctx_list = self._refine_in_sources(ctx_list)
            refined = self._refine_sources(ctx_list)
            refined = [r.clamp(-self._CTX_CLAMP, self._CTX_CLAMP) for r in refined]

            F_p_f = F_pixel.float()
            delta = torch.zeros_like(F_p_f)
            for i, ctx in enumerate(refined):
                ctx_up = F.interpolate(
                    ctx, size=(H, W), mode="bilinear", align_corners=False
                )
                gamma, beta = self.film_convs[i](ctx_up).chunk(2, dim=1)
                gamma = gamma.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                beta = beta.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                g = torch.sigmoid(self.gate_convs[i](ctx_up))
                if self.use_additive:
                    add = self.add_convs[i](ctx_up).clamp(-self._ADD_CLAMP, self._ADD_CLAMP)
                    delta = delta + g * (gamma * F_p_f + beta + add)
                else:
                    delta = delta + g * (gamma * F_p_f + beta)

            out = (F_p_f + delta).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
        return out.to(F_pixel.dtype)


class GatedPixelFusionPerSourceEnsembleLightUNet(nn.Module):
    """xf100: 4 single-source fusion branches + shared head + output averaging.

    Mimics the 4-tri-modal ensemble pattern (alpha+tessera+single token, 4 such
    models, averaged at inference) inside one end-to-end model. Pixel backbone
    is the standard alpha_unet + tessera_unet + gated combination. The single
    bottleneck CrossSourceHybridFiLMFusion of xf086 is replaced by 4 parallel
    SINGLE-source fusion branches (each is CrossSourceHybridFiLMFusion with
    n_sources=1 → no cross-source attention coupling between sources). Each
    branch produces its own feat_i; the head is invoked 4 times and the dict
    outputs are averaged.

    Motivation: the user's 4-tri-modal ensemble outperforms xf086, indicating
    that giving each token source its own independent gradient pathway and
    output head matters more than coupling them in one fusion attention. The
    multi-position experiment (xf099) tests adding more positions; this run
    tests removing per-source competition entirely.
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, tessera_presence_ch=0,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 gate_init_bias=4.0, gate_mode="simple", gate_untied=False,
                 modality_dropout=0.0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, use_fraction_film=True,
                 use_fraction_aux=None, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, token_calibration=False,
                 token_ctx_ch=96, attn_heads=4, use_additive=True,
                 token_source_ch=768,
                 token_proj_depth=1,
                 height_from_pixel=False,
                 feat_aggregation="mean",
                 pixel_noise_std=0.0,
                 use_boundary_head=False,
                 presence_tower_depth=0,
                 split_trunk=False,
                 presence_trunk_grad_scale=1.0,
                 height_trunk_grad_scale=1.0,
                 **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError(
                "GatedPixelFusionPerSourceEnsembleLightUNet assumes 4 output channels"
            )
        # xf117: per-branch gaussian noise on F_pixel before each branch's
        # single-source FiLM fusion. Each of the 4 branches samples its own
        # noise, giving an ADDITIONAL diversity source on top of the real
        # tokens already differentiating branches. Off by default (0.0).
        self.pixel_noise_std = float(pixel_noise_std)
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionPerSourceEnsembleLightUNet expects AlphaEarth+Tessera "
                f"pixel input with >{alpha_channels} channels, got {pixel_channels}"
            )
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.n_sources = token_channels // token_source_ch
        self.token_source_ch = int(token_source_ch)

        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.gate_untied = bool(gate_untied)
        self.modality_dropout = float(modality_dropout)
        self.height_from_pixel = bool(height_from_pixel)
        if feat_aggregation not in {"mean", "concat"}:
            raise ValueError(
                f"feat_aggregation must be 'mean' or 'concat', got {feat_aggregation!r}"
            )
        self.feat_aggregation = str(feat_aggregation)
        tessera_channels = pixel_channels - alpha_channels

        self.alpha_unet = LightUNet(
            alpha_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind
        )
        self.alpha_unet.head = nn.Identity()

        self.tessera_entry = nn.Sequential(
            ChannelCalibration(tessera_channels),
            ConvGNAct(tessera_channels, tessera_channels, kernel_size=1, padding=0),
        )
        self.tessera_unet = LightUNet(
            tessera_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind
        )
        self.tessera_unet.head = nn.Identity()

        self.gate_conv = _build_fusion_gate(
            base_ch,
            mode=gate_mode,
            untied=self.gate_untied,
            init_bias=gate_init_bias,
        )

        # 4 independent single-source fusion branches. Each is
        # CrossSourceHybridFiLMFusion with n_sources=1 → its internal
        # MultiheadAttention becomes intra-source spatial attention (256 tokens
        # of dim ctx_ch over a single source) rather than cross-source coupling.
        self.branch_fusions = nn.ModuleList([
            CrossSourceHybridFiLMFusion(
                pixel_ch=base_ch,
                token_channels=token_source_ch,
                token_source_ch=token_source_ch,
                ctx_ch=token_ctx_ch,
                token_calibration=token_calibration,
                token_proj_depth=token_proj_depth,
                attn_heads=attn_heads,
                use_additive=use_additive,
            )
            for _ in range(self.n_sources)
        ])

        # Optional feature-concat aggregator (xf112). When enabled, the 4
        # per-source feat_i are concatenated along channels and reduced back
        # to base_ch by a 1x1 conv mixer. The head is then invoked ONCE on
        # the mixed feature instead of 4 times on each feat_i. Per-branch
        # head calls still run when return_aux=True so branch_outs feeds the
        # deep_supervision aux loss.
        if self.feat_aggregation == "concat":
            self.feat_mixer = ConvGNAct(
                self.n_sources * base_ch, base_ch, kernel_size=1, padding=0
            )
        else:
            self.feat_mixer = None

        self.tessera_presence_ch = int(tessera_presence_ch)
        self.presence_extra_proj = (
            nn.Conv2d(base_ch, self.tessera_presence_ch, 1)
            if self.tessera_presence_ch > 0 else None
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            presence_extra_ch=self.tessera_presence_ch,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            use_fraction_film=use_fraction_film,
            use_fraction_aux=use_fraction_aux,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            use_boundary_head=use_boundary_head,
            presence_tower_depth=presence_tower_depth,
            split_trunk=split_trunk,
                presence_trunk_grad_scale=presence_trunk_grad_scale,
                height_trunk_grad_scale=height_trunk_grad_scale,
        )

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "GatedPixelFusionPerSourceEnsembleLightUNet expects (pixel, token) input"
            )
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels]
        tessera = pixel[:, self.alpha_channels:]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_unet.forward_features(self.tessera_entry(tessera))
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )

        token_parts = token.split(self.token_source_ch, dim=1)
        height_feature_x = fused if self.height_from_pixel else None

        # Per-branch F_pixel noise (xf117): each branch's fusion sees a
        # different noisy F_pixel during training. Provides an extra symmetry
        # breaker on top of the natural token-source differences. Off when
        # pixel_noise_std=0.0 OR in eval mode.
        def _branch_F_pixel(_i):
            if self.training and self.pixel_noise_std > 0.0:
                return fused + self.pixel_noise_std * torch.randn_like(fused)
            return fused

        # Always compute per-source feat_i (cheap relative to the head).
        feats = [self.branch_fusions[i](_branch_F_pixel(i), tok_i)
                 for i, tok_i in enumerate(token_parts)]

        if self.feat_aggregation == "concat":
            feat_main = self.feat_mixer(torch.cat(feats, dim=1))
            out_main = self.head(
                feat_main, return_aux=return_aux, presence_extra=presence_extra,
                height_feature_x=height_feature_x,
            )
            if not return_aux:
                return out_main
            # Training path: head is also called per-branch so branch_outs
            # can feed the deep_supervision aux loss in train_loop.
            branch_outs = [
                self.head(
                    fi, return_aux=True, presence_extra=presence_extra,
                    height_feature_x=height_feature_x,
                )
                for fi in feats
            ]
            out_main["branch_outs"] = branch_outs
            return out_main

        # Default xf107 path: head per branch, mean-aggregate outputs.
        branch_outs = [
            self.head(
                fi, return_aux=return_aux, presence_extra=presence_extra,
                height_feature_x=height_feature_x,
            )
            for fi in feats
        ]

        if not return_aux:
            return torch.stack(branch_outs, dim=0).mean(0)

        averaged = {}
        for k in branch_outs[0].keys():
            vals = [b[k] for b in branch_outs]
            if all(v is None for v in vals):
                averaged[k] = None
            elif any(v is None for v in vals):
                stack = torch.stack([v for v in vals if v is not None], dim=0)
                averaged[k] = stack.mean(0)
            else:
                averaged[k] = torch.stack(vals, dim=0).mean(0)
        averaged["branch_outs"] = branch_outs
        return averaged


class GatedPixelFusionHybridLightUNet(nn.Module):
    """xfusion_085 SoTA: dual-LightUNet pixel backbone + cross-source hybrid token fusion.

    Pixel backbone: symmetric AlphaEarth + Tessera LightUNet branches merged by
    a learned spatial gate. Token conditioning is CrossSourceHybridFiLMFusion:
    the N token sources are refined via cross-source self-attention, then each
    refined source contributes a zero-init (FiLM γ/β + additive A + spatial-gate
    σ(g)) residual.
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, tessera_presence_ch=0,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 gate_init_bias=4.0, gate_mode="simple", gate_untied=False,
                 modality_dropout=0.0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, use_fraction_film=True,
                 use_fraction_aux=None, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, token_calibration=False,
                 token_ctx_ch=96, attn_heads=4, use_additive=True,
                 token_calibration_source_indices=None,
                 token_in_source_attn=False,
                 token_cross_source_attn=True,
                 pixel_noise_std=0.0,
                 n_head_replicas=1,
                 token_input_clamp=None,
                 symmetric_modality_dropout=0.0,
                 symmetric_modality_dropout_alpha_share=0.5,
                 pixel_backbone_kind="unet",
                 use_boundary_head=False,
                 presence_tower_depth=0,
                 split_trunk=False,
                 presence_trunk_grad_scale=1.0,
                 height_trunk_grad_scale=1.0,
                 **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError(
                "GatedPixelFusionHybridLightUNet assumes 4 output channels"
            )
        self.pixel_backbone_kind = (pixel_backbone_kind or "unet").lower()
        self.pixel_noise_std = float(pixel_noise_std)
        self.n_head_replicas = max(1, int(n_head_replicas))
        self.symmetric_modality_dropout = float(symmetric_modality_dropout)
        self.symmetric_modality_dropout_alpha_share = float(symmetric_modality_dropout_alpha_share)
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionHybridLightUNet expects AlphaEarth+Tessera "
                f"pixel input with >{alpha_channels} channels, got {pixel_channels}"
            )

        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.gate_untied = bool(gate_untied)
        self.modality_dropout = float(modality_dropout)
        tessera_channels = pixel_channels - alpha_channels

        self.alpha_unet = _build_pixel_backbone(
            self.pixel_backbone_kind, alpha_channels, n_classes,
            base_ch=base_ch, norm_kind=norm_kind,
        )
        self.alpha_unet.head = nn.Identity()

        self.tessera_entry = nn.Sequential(
            ChannelCalibration(tessera_channels),
            ConvGNAct(tessera_channels, tessera_channels, kernel_size=1, padding=0),
        )
        self.tessera_unet = _build_pixel_backbone(
            self.pixel_backbone_kind, tessera_channels, n_classes,
            base_ch=base_ch, norm_kind=norm_kind,
        )
        self.tessera_unet.head = nn.Identity()

        self.gate_conv = _build_fusion_gate(
            base_ch,
            mode=gate_mode,
            untied=self.gate_untied,
            init_bias=gate_init_bias,
        )

        self.hybrid_fusion = CrossSourceHybridFiLMFusion(
            pixel_ch=base_ch,
            token_channels=token_channels,
            ctx_ch=token_ctx_ch,
            token_calibration=token_calibration,
            token_calibration_source_indices=token_calibration_source_indices,
            attn_heads=attn_heads,
            use_additive=use_additive,
            token_in_source_attn=token_in_source_attn,
            token_cross_source_attn=token_cross_source_attn,
            token_input_clamp=token_input_clamp,
        )

        self.tessera_presence_ch = int(tessera_presence_ch)
        self.presence_extra_proj = (
            nn.Conv2d(base_ch, self.tessera_presence_ch, 1)
            if self.tessera_presence_ch > 0 else None
        )

        def _build_head():
            return MultiTaskPredictionHead(
                in_ch=base_ch,
                out_channels=n_classes,
                presence_extra_ch=self.tessera_presence_ch,
                height_specialist_depth=height_specialist_depth,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                use_fraction_film=use_fraction_film,
                use_fraction_aux=use_fraction_aux,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
                use_boundary_head=use_boundary_head,
                presence_tower_depth=presence_tower_depth,
                split_trunk=split_trunk,
                presence_trunk_grad_scale=presence_trunk_grad_scale,
                height_trunk_grad_scale=height_trunk_grad_scale,
            )

        if self.n_head_replicas == 1:
            self.head = _build_head()
            self.heads = None
        else:
            # 4-replica hybrid (xf116): independent head weights per replica,
            # different random init seed-by-seed → diverged solutions after training
            # with per-replica F_pixel noise. Output is averaged at inference.
            self.head = None
            self.heads = nn.ModuleList([_build_head() for _ in range(self.n_head_replicas)])

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "GatedPixelFusionHybridLightUNet expects (pixel, token) input"
            )
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels]
        tessera = pixel[:, self.alpha_channels:]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_unet.forward_features(self.tessera_entry(tessera))
        # Asymmetric tessera-only dropout (legacy, default behavior).
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        # Symmetric at-most-one pixel-modality dropout (xf95-style experiments).
        # When symmetric_modality_dropout > 0, applied on top of the asymmetric
        # tessera dropout above. Independent randomness per call.
        if self.symmetric_modality_dropout > 0.0:
            alpha_feat, tessera_feat = _maybe_drop_modality_symmetric(
                alpha_feat, tessera_feat,
                self.symmetric_modality_dropout, self.training,
                alpha_share=self.symmetric_modality_dropout_alpha_share,
            )
        F_pixel = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )

        # Inject gaussian noise on F_pixel (post-gate, pre-FiLM) during training.
        # Token information path (token_projs + cross_source_attn + film_convs)
        # is UNTOUCHED — keeps real multimodal signal intact. See
        # project_noisetok_ablation: xf107's +0.0084 over pixel-only baseline
        # appears to be entirely noise-driven, so adding controlled noise to
        # xf085's coupled fusion may close the gap while preserving token use.
        if self.n_head_replicas == 1:
            if self.training and self.pixel_noise_std > 0.0:
                F_pixel_inj = F_pixel + self.pixel_noise_std * torch.randn_like(F_pixel)
            else:
                F_pixel_inj = F_pixel
            fused = self.hybrid_fusion(F_pixel_inj, token)
            return self.head(fused, return_aux=return_aux, presence_extra=presence_extra)

        # xf116 4-head hybrid: per-replica F_pixel noise + per-replica fusion +
        # per-replica head. deep_supervision_weight in train_loop will operate
        # on the returned branch_outs list when return_aux=True.
        branch_outs = []
        for head in self.heads:
            if self.training and self.pixel_noise_std > 0.0:
                F_pix_i = F_pixel + self.pixel_noise_std * torch.randn_like(F_pixel)
            else:
                F_pix_i = F_pixel
            fused_i = self.hybrid_fusion(F_pix_i, token)
            branch_outs.append(
                head(fused_i, return_aux=return_aux, presence_extra=presence_extra)
            )

        if not return_aux:
            return torch.stack(branch_outs, dim=0).mean(0)

        averaged = {}
        for k in branch_outs[0].keys():
            vals = [b[k] for b in branch_outs]
            if all(v is None for v in vals):
                averaged[k] = None
            elif any(v is None for v in vals):
                stack = torch.stack([v for v in vals if v is not None], dim=0)
                averaged[k] = stack.mean(0)
            else:
                averaged[k] = torch.stack(vals, dim=0).mean(0)
        averaged["branch_outs"] = branch_outs
        return averaged
