import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ChannelCalibration, ConvGNAct, _group_count
from .backbones import LightUNet
from .heads import MultiTaskPredictionHead
from .pixel_fusion import (
    TesseraCompressionStem,
    _apply_fusion_gate,
    _build_fusion_gate,
    _maybe_drop_modality,
)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.05):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query, key_value, key_bias=None):
        q = self.q_norm(query)
        kv = self.kv_norm(key_value)
        attn_mask = None
        if key_bias is not None:
            if key_bias.shape != key_value.shape[:2]:
                raise ValueError(
                    "key_bias must have shape [batch, key_len], got "
                    f"{tuple(key_bias.shape)} for key_value {tuple(key_value.shape)}"
                )
            b, query_len, _ = q.shape
            key_len = kv.size(1)
            attn_mask = key_bias.float().unsqueeze(1).expand(
                b, query_len, key_len
            )
            attn_mask = attn_mask.repeat_interleave(
                self.attn.num_heads, dim=0
            )
        # Run attention in float32: float16 QK^T can overflow (max ~65504),
        # causing softmax to produce NaN under AMP autocast.
        with torch.amp.autocast("cuda", enabled=False):
            out, _ = self.attn(
                q.float(), kv.float(), kv.float(),
                attn_mask=attn_mask,
                need_weights=False,
            )
        return query + out.to(query.dtype)


class BottleneckTwoGateAttentionFusion(nn.Module):
    """Token cross-attention injected at the UNet bottleneck (32×32).

    Key difference from TwoGateAttentionFusion (output-level):
    - No pooling: bottleneck is already at 32×32, matching token scale closely.
    - No upsampling: attention output stays at 32×32 = bottleneck resolution.
    - Q=1024 (32×32 bottleneck), KV=4×256=1024 (four 16×16 token sources) — balanced.
    - The 8× spatial upsampling artefact of output-level fusion is eliminated.
    """

    def __init__(self, bottleneck_ch, token_channels, dim=96,
                 token_source_ch=768, num_heads=4, dropout=0.05):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        self.dim = int(dim)

        self.main_proj = nn.Conv2d(bottleneck_ch, dim, 1)
        self.token_projs = nn.ModuleList(
            nn.Conv2d(token_source_ch, dim, 1)
            for _ in range(self.num_token_sources)
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.modality_embed = nn.Parameter(torch.zeros(1 + self.num_token_sources, dim))
        nn.init.normal_(self.modality_embed, std=0.02)

        self.cross_attn = _CrossAttentionBlock(dim, num_heads=num_heads, dropout=dropout)

        self.mix_proj = nn.Sequential(
            ConvGNAct(dim, dim, kernel_size=3),
            nn.Conv2d(dim, bottleneck_ch, 1),
        )

        gate_hidden = max(bottleneck_ch // 4, 32)
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * bottleneck_ch, gate_hidden, 1, bias=False),
            nn.GroupNorm(_group_count(gate_hidden), gate_hidden),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 2, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        self.gate_net[-1].bias.data.copy_(
            torch.tensor([2.0, -2.0], dtype=self.gate_net[-1].bias.dtype)
        )

    def _pos_tokens(self, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2)
        return self.pos_mlp(coords)

    def _tokens_from_map(self, feat, modality_idx):
        b, _, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens + self._pos_tokens(h, w, feat.device, feat.dtype)
        return tokens + self.modality_embed[modality_idx].view(1, 1, -1)

    def forward(self, bottleneck, token):
        b, _, h, w = bottleneck.shape

        main_feat = self.main_proj(bottleneck)
        query = self._tokens_from_map(main_feat, 0)

        token_parts = torch.split(token, self.token_source_ch, dim=1)
        kv_tokens = [
            self._tokens_from_map(proj(tp), idx)
            for idx, (proj, tp) in enumerate(zip(self.token_projs, token_parts), start=1)
        ]
        key_value = torch.cat(kv_tokens, dim=1)

        mix_tokens = self.cross_attn(query, key_value)

        mix = mix_tokens.transpose(1, 2).reshape(b, self.dim, h, w)
        mix = self.mix_proj(mix)

        gates = torch.sigmoid(self.gate_net(torch.cat([bottleneck, mix], dim=1)))
        return gates[:, 0:1] * bottleneck + gates[:, 1:2] * mix


class GatedPixelFusionTwoGateBnAttentionLightUNet(nn.Module):
    """027 architecture with token cross-attention at the UNet bottleneck.

    Replaces the output-level TwoGateAttentionFusion with bottleneck injection:
    token features are fused at the 32×32 bottleneck before the decoder runs,
    so the token signal propagates through all decoder skip connections. This
    eliminates the 8× upsampling artefact of output-level fusion and places
    the token injection at the spatial scale closest to the token grid (16×16).
    Tessera pixel fusion remains at the output level (same as 027).
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
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionTwoGateBnAttentionLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionTwoGateBnAttentionLightUNet expects AlphaEarth+Tessera "
                f"pixel input with >{alpha_channels} channels, got {pixel_channels}"
            )

        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.gate_untied = bool(gate_untied)
        self.modality_dropout = float(modality_dropout)
        tessera_channels = pixel_channels - alpha_channels
        bottleneck_ch = base_ch * 8

        self.alpha_unet = LightUNet(
            alpha_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind
        )
        self.alpha_unet.head = nn.Identity()
        self.tessera_feature_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=base_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        self.gate_conv = _build_fusion_gate(
            base_ch,
            mode=gate_mode,
            untied=self.gate_untied,
            init_bias=gate_init_bias,
        )
        self.bottleneck_token_fusion = BottleneckTwoGateAttentionFusion(
            bottleneck_ch=bottleneck_ch,
            token_channels=token_channels,
        )

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
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
        )

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("GatedPixelFusionTwoGateBnAttentionLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        x1 = self.alpha_unet.inc(alpha)
        x2 = self.alpha_unet.down1(x1)
        x3 = self.alpha_unet.down2(x2)
        x4 = self.alpha_unet.down3(x3)

        x4 = self.bottleneck_token_fusion(x4, token)

        feat = self.alpha_unet.up1(x4)
        feat = torch.cat([x3, feat], dim=1)
        feat = self.alpha_unet.conv1(feat)

        feat = self.alpha_unet.up2(feat)
        feat = torch.cat([x2, feat], dim=1)
        feat = self.alpha_unet.conv2(feat)

        feat = self.alpha_unet.up3(feat)
        feat = torch.cat([x1, feat], dim=1)
        feat = self.alpha_unet.conv3(feat)

        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, feat, tessera_feat, untied=self.gate_untied
        )
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        return self.head(fused, return_aux=return_aux, presence_extra=presence_extra)


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


class PerModalityFiLMFusion(nn.Module):
    """Per-source FiLM: each token source independently modulates pixel features.

    Each source i:
        token_src_i (768ch) → [calib →] proj_i(ctx_ch) → upsample(H, W)
        → film_conv_i(ctx_ch → 2*pixel_ch) → γ_i, β_i  [zero-initialized]

    Output: F_pixel + Σ_i (γ_i ⊙ F_pixel + β_i)

    Zero-init → identity at init; each source independently learns to contribute.
    Unhelpful sources converge γ_i ≈ 0, β_i ≈ 0 without penalising others.
    token_proj_depth controls projection depth: 1=linear (default), 2=GN+GELU
    between two conv layers, allowing non-linear token feature extraction.
    """

    _TOKEN_INPUT_CLAMP = 50.0
    _CTX_CLAMP = 50.0
    _FILM_PARAM_CLAMP = 4.0

    def __init__(self, pixel_ch, token_channels, token_source_ch=768,
                 ctx_ch=96, token_calibration=False, token_proj_depth=1):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"PerModalityFiLMFusion: token_channels={token_channels} must be "
                f"divisible by token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.n_sources = token_channels // token_source_ch

        self.token_calibs = (
            nn.ModuleList(
                ChannelCalibration(token_source_ch) for _ in range(self.n_sources)
            )
            if token_calibration else None
        )
        self.token_projs = nn.ModuleList([
            _make_token_proj(token_source_ch, ctx_ch, depth=token_proj_depth)
            for _ in range(self.n_sources)
        ])
        self.film_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, pixel_ch * 2, 1)
            for _ in range(self.n_sources)
        ])
        for conv in self.film_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, F_pixel, token):
        """
        F_pixel: (B, pixel_ch, H, W)
        token:   (B, n_sources * token_source_ch, h, w)  [e.g. 4×768 at 16×16]
        returns: (B, pixel_ch, H, W)
        """
        H, W = F_pixel.shape[-2:]
        parts = token.float().split(self.token_source_ch, dim=1)
        with torch.amp.autocast("cuda", enabled=False):
            F_p_f = F_pixel.float()
            delta = torch.zeros_like(F_p_f)
            for i, (src, proj, film_conv) in enumerate(
                zip(parts, self.token_projs, self.film_convs)
            ):
                if self.token_calibs is not None:
                    src = self.token_calibs[i](src)
                src = src.clamp(-self._TOKEN_INPUT_CLAMP, self._TOKEN_INPUT_CLAMP)
                ctx = proj(src).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
                ctx_up = F.interpolate(ctx, size=(H, W), mode="bilinear", align_corners=False)
                gamma, beta = film_conv(ctx_up).chunk(2, dim=1)
                gamma = gamma.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                beta = beta.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                delta = delta + gamma * F_p_f + beta
            out = (F_p_f + delta).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
        return out.to(F_pixel.dtype)

    def forward_with_deltas(self, F_pixel, token):
        """Like forward() but also returns per-source deltas as a list of [B, C, H, W].

        Returns:
            out    : (B, pixel_ch, H, W)  — same as forward()
            deltas : list of n_sources tensors, each (B, pixel_ch, H, W)
                     where deltas[i] = gamma_i * F_pixel + beta_i
        """
        H, W = F_pixel.shape[-2:]
        parts = token.float().split(self.token_source_ch, dim=1)
        with torch.amp.autocast("cuda", enabled=False):
            F_p_f = F_pixel.float()
            delta_list = []
            for i, (src, proj, film_conv) in enumerate(
                zip(parts, self.token_projs, self.film_convs)
            ):
                if self.token_calibs is not None:
                    src = self.token_calibs[i](src)
                src = src.clamp(-self._TOKEN_INPUT_CLAMP, self._TOKEN_INPUT_CLAMP)
                ctx = proj(src).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
                ctx_up = F.interpolate(ctx, size=(H, W), mode="bilinear", align_corners=False)
                gamma, beta = film_conv(ctx_up).chunk(2, dim=1)
                gamma = gamma.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                beta = beta.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                delta_list.append(gamma * F_p_f + beta)
            out = (F_p_f + sum(delta_list)).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
        dtype = F_pixel.dtype
        return out.to(dtype), [d.to(dtype) for d in delta_list]


class GatedPixelFusionFiLMPerModalityLightUNet(nn.Module):
    """xfusion_076: dual-LightUNet pixel backbone + per-source FiLM token conditioning.

    Pixel backbone (symmetric LightUNet for both AE and Tessera):
        AE(64)       → LightUNet → alpha_feat (base_ch)
        Tessera(128) → tessera_entry → LightUNet → tessera_feat (base_ch)
        rich gate → fused (base_ch)

    Token conditioning: each of N token sources contributes an independent
    FiLM residual (zero-initialized):
        F_fused = fused + Σ_i (γ_i ⊙ fused + β_i)
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
                 token_ctx_ch=96, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError(
                "GatedPixelFusionFiLMPerModalityLightUNet assumes 4 output channels"
            )
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionFiLMPerModalityLightUNet expects AlphaEarth+Tessera "
                f"pixel input with >{alpha_channels} channels, got {pixel_channels}"
            )

        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.gate_untied = bool(gate_untied)
        self.modality_dropout = float(modality_dropout)
        tessera_channels = pixel_channels - alpha_channels

        self.alpha_unet = LightUNet(
            alpha_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind
        )
        self.alpha_unet.head = nn.Identity()

        # Symmetric LightUNet for Tessera (xf076-style), preceded by a small
        # channel-calibration + 1x1 entry block to align scale across the
        # 128-dim Tessera embedding.
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

        self.film_per_modality = PerModalityFiLMFusion(
            pixel_ch=base_ch,
            token_channels=token_channels,
            ctx_ch=token_ctx_ch,
            token_calibration=token_calibration,
        )

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
        )

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "GatedPixelFusionFiLMPerModalityLightUNet expects (pixel, token) input"
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
        fused = self.film_per_modality(fused, token)
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        return self.head(fused, return_aux=return_aux, presence_extra=presence_extra)


class CrossSourceHybridFiLMFusion(nn.Module):
    """xf083 fusion: cross-source self-attention + hybrid affine/additive modulation.

    Stage 1 (16x16 token scale): the N token sources cross-attend to each other
    via one self-attention layer with learned modality embeddings + 2D positional
    encoding. The attention output projection is zero-initialised, so the refined
    source equals the projected source at init.

    Stage 2 (H x W pixel scale): each refined source contributes three zero-init
    residuals applied as
        delta_i = sigmoid(g_i) * (gamma_i * F_pixel + beta_i + A_i)
        F_out   = F_pixel + sum_i delta_i
    The A_i path is the new additive content that bypasses FiLM's affine bottleneck.
    """

    _TOKEN_INPUT_CLAMP = 50.0
    _CTX_CLAMP = 50.0
    _FILM_PARAM_CLAMP = 4.0
    _ADD_CLAMP = 4.0

    def __init__(self, pixel_ch, token_channels, token_source_ch=768,
                 ctx_ch=96, token_calibration=False, token_proj_depth=1,
                 attn_heads=4, attn_dropout=0.05,
                 use_additive=True, use_spatial_gate=True):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"CrossSourceHybridFiLMFusion: token_channels={token_channels} must be "
                f"divisible by token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.n_sources = token_channels // token_source_ch
        self.ctx_ch = int(ctx_ch)
        self.use_additive = bool(use_additive)
        self.use_spatial_gate = bool(use_spatial_gate)
        self.attn_heads = int(attn_heads)
        self.attn_enabled = self.attn_heads > 0

        self.token_calibs = (
            nn.ModuleList(
                ChannelCalibration(token_source_ch) for _ in range(self.n_sources)
            )
            if token_calibration else None
        )
        self.token_projs = nn.ModuleList([
            _make_token_proj(token_source_ch, ctx_ch, depth=token_proj_depth)
            for _ in range(self.n_sources)
        ])

        if self.attn_enabled:
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, ctx_ch),
                nn.GELU(),
                nn.Linear(ctx_ch, ctx_ch),
            )
            self.modality_embed = nn.Parameter(
                torch.zeros(self.n_sources, ctx_ch)
            )
            nn.init.normal_(self.modality_embed, std=0.02)
            self.attn_norm = nn.LayerNorm(ctx_ch)
            self.cross_source_attn = nn.MultiheadAttention(
                ctx_ch, self.attn_heads,
                dropout=attn_dropout, batch_first=True,
            )
            nn.init.zeros_(self.cross_source_attn.out_proj.weight)
            nn.init.zeros_(self.cross_source_attn.out_proj.bias)

        self.film_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, pixel_ch * 2, 1) for _ in range(self.n_sources)
        ])
        for conv in self.film_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

        if self.use_additive:
            self.add_convs = nn.ModuleList([
                nn.Conv2d(ctx_ch, pixel_ch, 1) for _ in range(self.n_sources)
            ])
            for conv in self.add_convs:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
        else:
            self.add_convs = None

        if self.use_spatial_gate:
            self.gate_convs = nn.ModuleList([
                nn.Conv2d(ctx_ch, 1, 1) for _ in range(self.n_sources)
            ])
            for conv in self.gate_convs:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
        else:
            self.gate_convs = None

    def _pos_tokens(self, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2)
        return self.pos_mlp(coords)

    def _refine_sources(self, ctx_list):
        if not self.attn_enabled:
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
                src = src.clamp(-self._TOKEN_INPUT_CLAMP, self._TOKEN_INPUT_CLAMP)
                ctx = self.token_projs[i](src).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
                ctx_list.append(ctx)

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
                src_delta = gamma * F_p_f + beta

                if self.add_convs is not None:
                    add = self.add_convs[i](ctx_up).clamp(
                        -self._ADD_CLAMP, self._ADD_CLAMP
                    )
                    src_delta = src_delta + add

                if self.gate_convs is not None:
                    g = torch.sigmoid(self.gate_convs[i](ctx_up))
                    src_delta = g * src_delta

                delta = delta + src_delta

            out = (F_p_f + delta).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
        return out.to(F_pixel.dtype)


class GatedPixelFusionHybridLightUNet(nn.Module):
    """xfusion_083: dual-LightUNet pixel backbone + cross-source hybrid token fusion.

    Pixel backbone is identical to GatedPixelFusionFiLMPerModalityLightUNet
    (xfusion_082). Token conditioning swaps PerModalityFiLMFusion for
    CrossSourceHybridFiLMFusion: token sources are first refined via cross-source
    self-attention, then each refined source contributes a zero-init
    (FiLM + additive + spatial-gate) residual.
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
                 token_ctx_ch=96,
                 attn_heads=4, use_additive=True, use_spatial_gate=True,
                 **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError(
                "GatedPixelFusionHybridLightUNet assumes 4 output channels"
            )
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

        self.hybrid_fusion = CrossSourceHybridFiLMFusion(
            pixel_ch=base_ch,
            token_channels=token_channels,
            ctx_ch=token_ctx_ch,
            token_calibration=token_calibration,
            attn_heads=attn_heads,
            use_additive=use_additive,
            use_spatial_gate=use_spatial_gate,
        )

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
        )

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
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.hybrid_fusion(fused, token)
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        return self.head(fused, return_aux=return_aux, presence_extra=presence_extra)
