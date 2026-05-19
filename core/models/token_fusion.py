import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import HEIGHT_NORM_CONSTANT, ChannelCalibration, ConvGNAct, _group_count
from .backbones import LightUNet
from .heads import MultiTaskPredictionHead
from .pixel_fusion import (
    TesseraCompressionStem,
    _apply_fusion_gate,
    _build_fusion_gate,
    _maybe_drop_modality,
)


class TokenPyramidNeck(nn.Module):
    """Pseudo-pyramid for one 16x16 token source.

    Attribute names match the earlier implementation so active xfusion
    checkpoints remain loadable.
    """

    def __init__(self, in_ch=768, level_channels=(256, 128, 64, 32)):
        super().__init__()
        if len(level_channels) != 4:
            raise ValueError("TokenPyramidNeck expects 4 level channel sizes")
        c16, c32, c64, c128 = level_channels
        self.level_16 = ConvGNAct(in_ch, c16, kernel_size=1, padding=0)
        self.level_32 = nn.Sequential(
            ConvGNAct(in_ch, c32, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(c32, c32, kernel_size=3),
        )
        self.level_64 = nn.Sequential(
            ConvGNAct(in_ch, c64, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvGNAct(c64, c64, kernel_size=3),
        )
        self.level_128 = nn.Sequential(
            ConvGNAct(in_ch, c128, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            ConvGNAct(c128, c128, kernel_size=3),
        )

    def forward(self, x):
        return {
            16: self.level_16(x),
            32: self.level_32(x),
            64: self.level_64(x),
            128: self.level_128(x),
        }


class TokenPyramidProvider(nn.Module):
    """Active xfusion token path: a single TerraMind-S2 token grid."""

    def __init__(self, token_channels, level_channels=(256, 128, 64, 32),
                 **unused):
        super().__init__()
        self.token_norm = nn.Identity()
        self.fusion = nn.Identity()
        self.neck = TokenPyramidNeck(token_channels, level_channels=level_channels)

    def forward(self, token):
        token = self.fusion(self.token_norm(token))
        return self.neck(token)


class GatedTokenScaleResidual(nn.Module):
    """Zero-initialized token residual for one AlphaEarth U-Net scale."""

    def __init__(self, token_ch, target_ch, hidden_ch=None):
        super().__init__()
        hidden_ch = hidden_ch or min(max(target_ch, 64), 256)
        self.net = nn.Sequential(
            ConvGNAct(token_ch, hidden_ch, kernel_size=1, padding=0),
            ConvGNAct(hidden_ch, target_ch, kernel_size=3),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, target, token_feat):
        if token_feat.shape[-2:] != target.shape[-2:]:
            token_feat = F.interpolate(
                token_feat,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return target + self.gate * self.net(token_feat)


class TesseraTokenCrossLevelFusionLightUNet(nn.Module):
    """Active three-modal model: AE+Tessera pixels plus TerraMind-S2 tokens."""

    _TOKEN_LEVEL_CHANNELS = {32: 128, 64: 64, 128: 32}

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, tessera_presence_ch=16,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 height_gate_source="fused", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, token_fusion_kind="single",
                 fusion_points=("decoder64",), normalize_tokens=False,
                 presence_head_kind="split_all", presence_head_depth=2,
                 presence_branch_ch=48, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraTokenCrossLevelFusionLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "TesseraTokenCrossLevelFusionLightUNet expects AlphaEarth+Tessera pixel "
                f"input with >{alpha_channels} channels, got {pixel_channels}"
            )
        if token_fusion_kind != "single":
            raise ValueError("Active xfusion keeps only token_fusion_kind='single'")
        if normalize_tokens:
            raise ValueError("Active xfusion keeps raw TerraMind-S2 tokens without normalization")

        allowed_points = {"bottleneck", "decoder64", "decoder128"}
        self.fusion_points = tuple(fusion_points)
        unknown = set(self.fusion_points) - allowed_points
        if unknown:
            raise ValueError(f"Unknown cross-level fusion point(s): {sorted(unknown)}")

        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = pixel_channels - alpha_channels
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = (
            TesseraCompressionStem(
                tessera_channels,
                out_ch=tessera_presence_ch,
                hidden_ch=tessera_hidden_ch,
                hidden_depth=tessera_hidden_depth,
            )
            if tessera_presence_ch > 0 else None
        )
        self.token_pyramid = TokenPyramidProvider(token_channels)
        self._token_input_channels = dict(self._TOKEN_LEVEL_CHANNELS)

        self.bottleneck_adapter = (
            GatedTokenScaleResidual(self._token_input_channels[32], c4)
            if "bottleneck" in self.fusion_points else None
        )
        self.decoder64_adapter = (
            GatedTokenScaleResidual(self._token_input_channels[64], c3)
            if "decoder64" in self.fusion_points else None
        )
        self.decoder128_adapter = (
            GatedTokenScaleResidual(self._token_input_channels[128], c2)
            if "decoder128" in self.fusion_points else None
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            presence_extra_ch=tessera_presence_ch,
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
            raise ValueError("TesseraTokenCrossLevelFusionLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]
        token_pyr = self.token_pyramid(token)

        x1 = self.alpha_unet.inc(alpha)
        x2 = self.alpha_unet.down1(x1)
        x3 = self.alpha_unet.down2(x2)
        x4 = self.alpha_unet.down3(x3)
        if self.bottleneck_adapter is not None:
            x4 = self.bottleneck_adapter(x4, token_pyr[32])

        feat = self.alpha_unet.up1(x4)
        feat = torch.cat([x3, feat], dim=1)
        feat = self.alpha_unet.conv1(feat)
        if self.decoder64_adapter is not None:
            feat = self.decoder64_adapter(feat, token_pyr[64])

        feat = self.alpha_unet.up2(feat)
        feat = torch.cat([x2, feat], dim=1)
        feat = self.alpha_unet.conv2(feat)
        if self.decoder128_adapter is not None:
            feat = self.decoder128_adapter(feat, token_pyr[128])

        feat = self.alpha_unet.up3(feat)
        feat = torch.cat([x1, feat], dim=1)
        feat = self.alpha_unet.conv3(feat)

        presence_extra = (
            self.tessera_stem(tessera) if self.tessera_stem is not None else None
        )
        return self.head(feat, return_aux=return_aux,
                         presence_extra=presence_extra)


class GatedPixelFusionFiLMPerModalityLightUNet(nn.Module):
    """xfusion_062: 027 LightUNet pixel backbone + per-source FiLM token conditioning.

    Pixel backbone identical to xfusion_027:
        AE  → LightUNet → alpha_feat (base_ch)
        Tessera → TesseraCompressionStem → tessera_feat (base_ch)
        rich gate → fused (base_ch)

    Token conditioning replaces TwoGateAttentionFusion with PerModalityFiLMFusion:
    each token source contributes an independent FiLM residual (zero-initialized):
        F_fused = fused + Σ_i (γ_i ⊙ fused + β_i)

    Head unchanged from 027.
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

        # Pixel backbone (identical to 027)
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

        # Per-modality FiLM (replaces TwoGateAttentionFusion)
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
        tessera_feat = self.tessera_feature_stem(tessera)
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

class GatedPixelFusionTerraMindNonWaterLightUNet(nn.Module):
    """AE+Tessera gated fusion plus TerraMind building/tree residuals.

    This keeps the proven ``ae_tessera_gated`` pixel path intact: AlphaEarth
    goes through LightUNet, Tessera is compressed to a same-resolution feature,
    and the two are combined by the rich gate. TerraMind is routed only through
    a zero-initialized non-water presence residual so the first ablation tests
    whether TerraMind helps structure classes without perturbing water logits.
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, tessera_presence_ch=16,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 gate_init_bias=4.0, gate_mode="simple", gate_untied=False,
                 modality_dropout=0.0,
                 height_gate_source="fused", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="split_all", presence_head_depth=2,
                 presence_branch_ch=48, token_hidden_ch=64):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionTerraMindNonWaterLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionTerraMindNonWaterLightUNet expects AlphaEarth+Tessera "
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

        self.tessera_presence_ch = int(tessera_presence_ch)
        self.presence_extra_proj = (
            nn.Conv2d(base_ch, self.tessera_presence_ch, 1)
            if self.tessera_presence_ch > 0 else None
        )

        self.token_pyramid = TokenPyramidProvider(
            token_channels,
            level_channels=(256, 128, token_hidden_ch, 32),
        )
        self.token_nonwater_delta = nn.Conv2d(token_hidden_ch, 2, 1)
        nn.init.zeros_(self.token_nonwater_delta.weight)
        nn.init.zeros_(self.token_nonwater_delta.bias)

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
            raise ValueError("GatedPixelFusionTerraMindNonWaterLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
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

        token64 = self.token_pyramid(token)[64]
        if token64.shape[-2:] != fused.shape[-2:]:
            token64 = F.interpolate(
                token64,
                size=fused.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        nonwater_delta = self.token_nonwater_delta(token64)
        water_delta = nonwater_delta.new_zeros(
            nonwater_delta.size(0), 1, nonwater_delta.size(2), nonwater_delta.size(3)
        )
        presence_logit_delta = torch.cat(
            [nonwater_delta[:, 0:1], nonwater_delta[:, 1:2], water_delta],
            dim=1,
        )

        return self.head(
            fused,
            return_aux=return_aux,
            presence_extra=presence_extra,
            presence_logit_delta=presence_logit_delta,
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


class _LatentTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.05):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        xn = self.norm1(x)
        with torch.amp.autocast("cuda", enabled=False):
            h, _ = self.attn(xn.float(), xn.float(), xn.float(), need_weights=False)
        x = x + h.to(x.dtype)
        return x + self.mlp(self.norm2(x))


class BottleneckAdaptiveFusion(nn.Module):
    """Small Perceiver-style fusion controller for AE+Tessera + token sources."""

    def __init__(self, base_ch, token_channels, dim=96, num_latents=16,
                 token_source_ch=768, token_grid=32, num_heads=4,
                 latent_layers=1, mlp_ratio=2.0, dropout=0.05):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        self.token_grid = int(token_grid)
        self.dim = int(dim)

        self.pixel_pool = nn.AdaptiveAvgPool2d((self.token_grid, self.token_grid))
        self.pixel_proj = nn.Conv2d(base_ch, dim, 1)
        self.token_projs = nn.ModuleList(
            nn.Conv2d(token_source_ch, dim, 1)
            for _ in range(self.num_token_sources)
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.modality_embed = nn.Parameter(
            torch.zeros(1 + self.num_token_sources, dim)
        )
        nn.init.normal_(self.modality_embed, std=0.02)

        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.read = _CrossAttentionBlock(dim, num_heads=num_heads, dropout=dropout)
        self.blocks = nn.Sequential(*[
            _LatentTransformerBlock(
                dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
            )
            for _ in range(latent_layers)
        ])
        self.spatial_queries = nn.Parameter(
            torch.randn(self.token_grid * self.token_grid, dim) * 0.02
        )
        self.write = _CrossAttentionBlock(dim, num_heads=num_heads, dropout=dropout)
        self.delta_proj = nn.Sequential(
            ConvGNAct(dim, base_ch, kernel_size=3),
            nn.Conv2d(base_ch, base_ch, 1),
        )
        self.residual_gate = nn.Parameter(torch.full((1,), 0.1))

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

    def forward(self, fused, token):
        pixel_feat = self.pixel_proj(self.pixel_pool(fused))
        token_parts = torch.split(token, self.token_source_ch, dim=1)
        tokens = [self._tokens_from_map(pixel_feat, 0)]
        for idx, (proj, token_part) in enumerate(zip(self.token_projs, token_parts), start=1):
            tokens.append(self._tokens_from_map(proj(token_part), idx))
        all_tokens = torch.cat(tokens, dim=1)

        b = fused.size(0)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        latents = self.read(latents, all_tokens)
        latents = self.blocks(latents)

        queries = self.spatial_queries.unsqueeze(0).expand(b, -1, -1)
        dense_tokens = self.write(queries, latents)
        dense = dense_tokens.transpose(1, 2).reshape(
            b, self.dim, self.token_grid, self.token_grid
        )
        dense = F.interpolate(
            dense, size=fused.shape[-2:], mode="bilinear", align_corners=False
        )
        delta = self.delta_proj(dense)
        return fused + self.residual_gate * delta


class GatedPixelFusionBottleneckAdaptiveLightUNet(nn.Module):
    """AE+Tessera gated trunk with small bottleneck fusion for token modalities."""

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
                 presence_branch_ch=None):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionBottleneckAdaptiveLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionBottleneckAdaptiveLightUNet expects AlphaEarth+Tessera "
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
        self.bottleneck_fusion = BottleneckAdaptiveFusion(
            base_ch=base_ch,
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
            raise ValueError("GatedPixelFusionBottleneckAdaptiveLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.bottleneck_fusion(fused, token)
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        return self.head(fused, return_aux=return_aux,
                         presence_extra=presence_extra)


def _make_token_extractor(in_ch, out_ch, mid_ch=None):
    """Per-source feature extractor: 1x1 reduce → 3x3 spatial → 1x1 project."""
    mid_ch = mid_ch if mid_ch is not None else in_ch // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, 1, bias=False),
        nn.GroupNorm(_group_count(mid_ch), mid_ch),
        nn.GELU(),
        nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_ch), out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 1),
    )


class TwoGateAttentionFusion(nn.Module):
    """Attention mix feature with untied gates for main and token features."""

    def __init__(self, base_ch, token_channels, dim=96, token_source_ch=768,
                 token_grid=32, num_heads=4, dropout=0.05,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_calibration=False, token_gate_init_bias=(2.0, -2.0)):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        self.token_grid = int(token_grid)
        self.dim = int(dim)
        self.token_calibs = (
            nn.ModuleList(ChannelCalibration(token_source_ch)
                          for _ in range(self.num_token_sources))
            if token_calibration else None
        )

        self.main_pool = nn.AdaptiveAvgPool2d((self.token_grid, self.token_grid))
        self.main_proj = nn.Conv2d(base_ch, dim, 1)
        if use_token_extractor:
            self.token_projs = nn.ModuleList(
                _make_token_extractor(token_source_ch, dim, mid_ch=token_extractor_mid_ch)
                for _ in range(self.num_token_sources)
            )
        else:
            self.token_projs = nn.ModuleList(
                nn.Conv2d(token_source_ch, dim, 1)
                for _ in range(self.num_token_sources)
            )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.modality_embed = nn.Parameter(
            torch.zeros(1 + self.num_token_sources, dim)
        )
        nn.init.normal_(self.modality_embed, std=0.02)

        self.cross_attn = _CrossAttentionBlock(
            dim, num_heads=num_heads, dropout=dropout
        )
        self.mix_proj = nn.Sequential(
            ConvGNAct(dim, base_ch, kernel_size=3),
            ConvGNAct(base_ch, base_ch, kernel_size=3),
        )
        gate_hidden = max(base_ch, 32)
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * base_ch, gate_hidden, 1, bias=False),
            nn.GroupNorm(_group_count(gate_hidden), gate_hidden),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 2, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        self.gate_net[-1].bias.data.copy_(
            torch.as_tensor(token_gate_init_bias, dtype=self.gate_net[-1].bias.dtype)
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

    def forward(self, fused, token):
        orig_dtype = fused.dtype
        # Run token projection in float32: extreme token values (THOR S2 ~21674)
        # overflow float16 in Conv2d(768→96), producing inf → NaN in LayerNorm.
        with torch.amp.autocast("cuda", enabled=False):
            fused_f = fused.float()
            token_f = token.float()
            main_feat = self.main_proj(self.main_pool(fused_f))
            query = self._tokens_from_map(main_feat, 0)
            token_parts = torch.split(token_f, self.token_source_ch, dim=1)
            kv_tokens = []
            for idx, (proj, token_part) in enumerate(zip(self.token_projs, token_parts), start=1):
                if self.token_calibs is not None:
                    token_part = self.token_calibs[idx - 1](token_part)
                kv_tokens.append(self._tokens_from_map(proj(token_part), idx))
            key_value = torch.cat(kv_tokens, dim=1)
            mix_tokens = self.cross_attn(query, key_value)

            b = fused_f.size(0)
            mix = mix_tokens.transpose(1, 2).reshape(
                b, self.dim, self.token_grid, self.token_grid
            )
            mix = F.interpolate(
                mix, size=fused_f.shape[-2:], mode="bilinear", align_corners=False
            )
            mix = self.mix_proj(mix).to(orig_dtype)

        gates = torch.sigmoid(self.gate_net(torch.cat([fused, mix], dim=1)))
        g_main = gates[:, 0:1]
        g_mix = gates[:, 1:2]
        return g_main * fused + g_mix * mix


class CQAWeightedTwoGateAttentionFusion(TwoGateAttentionFusion):
    """xfusion_027 token attention with source-level CQA routing.

    The router estimates contribution, quality, and complementarity scores for
    each token source before concatenating the KV memory.  This keeps the
    original one-to-many cross-attention topology, but gives the model an
    explicit source-level control surface.
    """

    def __init__(self, base_ch, token_channels, dim=96, token_source_ch=768,
                 token_grid=32, num_heads=4, dropout=0.05,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_calibration=False, token_gate_init_bias=(2.0, -2.0),
                 cqa_hidden_ch=128, cqa_temperature=1.5,
                 cqa_source_dropout=0.0, cqa_scale_strength=0.0,
                 cqa_bias_strength=0.25):
        super().__init__(
            base_ch=base_ch,
            token_channels=token_channels,
            dim=dim,
            token_source_ch=token_source_ch,
            token_grid=token_grid,
            num_heads=num_heads,
            dropout=dropout,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_calibration=token_calibration,
            token_gate_init_bias=token_gate_init_bias,
        )
        self.cqa_temperature = float(cqa_temperature)
        self.cqa_source_dropout = float(cqa_source_dropout)
        self.cqa_scale_strength = float(cqa_scale_strength)
        self.cqa_bias_strength = float(cqa_bias_strength)
        self.cqa_source_embed = nn.Parameter(
            torch.zeros(self.num_token_sources, dim)
        )
        nn.init.normal_(self.cqa_source_embed, std=0.02)

        hidden = int(cqa_hidden_ch)
        self.contrib_mlp = self._make_cqa_mlp(5 * dim, hidden, dropout)
        self.quality_mlp = self._make_cqa_mlp(5 * dim, hidden, dropout)
        self.complement_mlp = self._make_cqa_mlp(9 * dim, hidden, dropout)
        self.cqa_lambda_logits = nn.Parameter(torch.zeros(3))
        self.last_source_weights = None
        self.last_cqa_lambdas = None

    @staticmethod
    def _make_cqa_mlp(in_dim, hidden_dim, dropout):
        mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
        return mlp

    @staticmethod
    def _mean_std(feat):
        feat = feat.float()
        mean = feat.mean(dim=(2, 3))
        var = feat.var(dim=(2, 3), unbiased=False)
        return mean, var.clamp_min(0.0).sqrt()

    @staticmethod
    def _clean_descriptor(desc):
        desc = torch.nan_to_num(desc.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return desc.clamp(min=-50.0, max=50.0)

    def _descriptor(self, feat):
        mean, std = self._mean_std(feat)
        return self._clean_descriptor(torch.cat([mean, std], dim=1))

    def _quality_stats(self, feat):
        mean, std = self._mean_std(feat)
        abs_feat = feat.abs()
        abs_mean = abs_feat.mean(dim=(2, 3))
        abs_max = abs_feat.amax(dim=(2, 3))
        return self._clean_descriptor(torch.cat([mean, std, abs_mean, abs_max], dim=1))

    def _drop_mask(self, batch, device):
        if not self.training or self.cqa_source_dropout <= 0.0:
            return None
        mask = (
            torch.rand(batch, self.num_token_sources, device=device)
            < self.cqa_source_dropout
        )
        all_dropped = mask.all(dim=1)
        if all_dropped.any():
            mask[all_dropped, 0] = False
        return mask

    def _source_weights(self, token_feats, main_feat):
        token_feats = [feat.detach().float() for feat in token_feats]
        main_feat = main_feat.detach().float()
        b = main_feat.size(0)
        source_embed = self.cqa_source_embed.view(
            1, self.num_token_sources, self.dim
        ).expand(b, -1, -1)
        pixel_desc = self._descriptor(main_feat)

        desc = torch.stack(
            [self._descriptor(feat) for feat in token_feats],
            dim=1,
        )
        qstats = torch.stack(
            [self._quality_stats(feat) for feat in token_feats],
            dim=1,
        )
        pixel_desc = pixel_desc.unsqueeze(1).expand(-1, self.num_token_sources, -1)

        contrib_in = torch.cat([desc, pixel_desc, source_embed], dim=2)
        quality_in = torch.cat([qstats, source_embed], dim=2)

        if self.num_token_sources > 1:
            leave_one_out = (
                desc.sum(dim=1, keepdim=True) - desc
            ) / (self.num_token_sources - 1)
        else:
            leave_one_out = desc.new_zeros(desc.shape)
        complement_in = torch.cat(
            [
                desc,
                leave_one_out,
                (desc - leave_one_out).abs(),
                desc * leave_one_out,
                source_embed,
            ],
            dim=2,
        )
        complement_in = self._clean_descriptor(complement_in)

        contrib = 2.0 * torch.tanh(self.contrib_mlp(contrib_in).squeeze(-1))
        quality = 2.0 * torch.tanh(self.quality_mlp(quality_in).squeeze(-1))
        complement = 2.0 * torch.tanh(self.complement_mlp(complement_in).squeeze(-1))

        lambdas = torch.softmax(self.cqa_lambda_logits, dim=0)
        scores = (
            lambdas[0] * contrib
            + lambdas[1] * quality
            + lambdas[2] * complement
        )
        drop_mask = self._drop_mask(b, main_feat.device)
        if drop_mask is not None:
            scores = scores.masked_fill(drop_mask, -1.0e4)

        temperature = max(self.cqa_temperature, 1.0e-4)
        weights = torch.softmax(scores.clamp(-4.0, 4.0) / temperature, dim=1)
        return torch.nan_to_num(
            weights,
            nan=1.0 / self.num_token_sources,
            posinf=1.0 / self.num_token_sources,
            neginf=1.0 / self.num_token_sources,
        )

    def forward(self, fused, token):
        main_feat = self.main_proj(self.main_pool(fused))
        query = self._tokens_from_map(main_feat, 0)
        token_parts = torch.split(token, self.token_source_ch, dim=1)

        token_feats = []
        for idx, (proj, token_part) in enumerate(zip(self.token_projs, token_parts)):
            if self.token_calibs is not None:
                token_part = self.token_calibs[idx](token_part)
            token_feats.append(proj(token_part))

        with torch.amp.autocast("cuda", enabled=False):
            source_weights = self._source_weights(token_feats, main_feat)
        self.last_source_weights = source_weights.detach()
        self.last_cqa_lambdas = torch.softmax(
            self.cqa_lambda_logits.detach(), dim=0
        )
        kv_tokens = [
            self._tokens_from_map(feat, idx + 1)
            for idx, feat in enumerate(token_feats)
        ]

        key_value = torch.cat(kv_tokens, dim=1)
        source_bias = torch.log(
            (source_weights * self.num_token_sources).clamp_min(1.0e-4)
        )
        source_bias = self.cqa_bias_strength * source_bias.clamp(-2.0, 2.0)
        if self.cqa_scale_strength > 0.0:
            source_bias = source_bias + self.cqa_scale_strength * (
                source_weights * self.num_token_sources - 1.0
            ).clamp(-1.0, 1.0)
        key_bias = source_bias.repeat_interleave(
            self.token_grid * self.token_grid, dim=1
        ).to(query.dtype)
        mix_tokens = self.cross_attn(query, key_value, key_bias=key_bias)

        b = fused.size(0)
        mix = mix_tokens.transpose(1, 2).reshape(
            b, self.dim, self.token_grid, self.token_grid
        )
        mix = F.interpolate(
            mix, size=fused.shape[-2:], mode="bilinear", align_corners=False
        )
        mix = self.mix_proj(mix)
        gates = torch.sigmoid(self.gate_net(torch.cat([fused, mix], dim=1)))
        g_main = gates[:, 0:1]
        g_mix = gates[:, 1:2]
        return g_main * fused + g_mix * mix


class AlignedTwoGateAttentionFusion(nn.Module):
    """Position-aligned token fusion for 16x16 token grids.

    Unlike ``TwoGateAttentionFusion``, each pooled pixel query attends only to
    token features at the same spatial index. This encodes the assumption that
    one token cell corresponds to the matching pixel block in the 256x256 input.
    """

    def __init__(self, base_ch, token_channels, dim=96, token_source_ch=768,
                 num_heads=4, dropout=0.05, use_token_extractor=False,
                 token_extractor_mid_ch=None, upsample_mode="nearest",
                 token_calibration=False, token_gate_init_bias=(2.0, -2.0)):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        self.dim = int(dim)
        self.upsample_mode = upsample_mode
        self.token_calibs = (
            nn.ModuleList(ChannelCalibration(token_source_ch)
                          for _ in range(self.num_token_sources))
            if token_calibration else None
        )

        self.main_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.main_proj = nn.Conv2d(base_ch, dim, 1)
        if use_token_extractor:
            self.token_projs = nn.ModuleList(
                _make_token_extractor(token_source_ch, dim, mid_ch=token_extractor_mid_ch)
                for _ in range(self.num_token_sources)
            )
        else:
            self.token_projs = nn.ModuleList(
                nn.Conv2d(token_source_ch, dim, 1)
                for _ in range(self.num_token_sources)
            )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.modality_embed = nn.Parameter(
            torch.zeros(1 + self.num_token_sources, dim)
        )
        nn.init.normal_(self.modality_embed, std=0.02)

        self.cross_attn = _CrossAttentionBlock(
            dim, num_heads=num_heads, dropout=dropout
        )
        self.mix_proj = nn.Sequential(
            ConvGNAct(dim, base_ch, kernel_size=3),
            ConvGNAct(base_ch, base_ch, kernel_size=3),
        )
        gate_hidden = max(base_ch, 32)
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * base_ch, gate_hidden, 1, bias=False),
            nn.GroupNorm(_group_count(gate_hidden), gate_hidden),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 2, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        self.gate_net[-1].bias.data.copy_(
            torch.as_tensor(token_gate_init_bias, dtype=self.gate_net[-1].bias.dtype)
        )

    def _pos_map(self, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2)
        return self.pos_mlp(coords).transpose(1, 2).reshape(1, self.dim, h, w)

    def _tokens_from_map(self, feat, modality_idx):
        b, _, h, w = feat.shape
        feat = feat + self._pos_map(h, w, feat.device, feat.dtype)
        feat = feat + self.modality_embed[modality_idx].view(1, self.dim, 1, 1)
        return feat.flatten(2).transpose(1, 2)

    def forward(self, fused, token):
        token_h, token_w = token.shape[-2:]
        main_feat = F.adaptive_avg_pool2d(fused, (token_h, token_w))
        main_feat = self.main_proj(main_feat)
        query = self._tokens_from_map(main_feat, 0)

        token_parts = torch.split(token, self.token_source_ch, dim=1)
        kv_parts = []
        for idx, (proj, token_part) in enumerate(zip(self.token_projs, token_parts), start=1):
            if self.token_calibs is not None:
                token_part = self.token_calibs[idx - 1](token_part)
            kv_parts.append(self._tokens_from_map(proj(token_part), idx))
        key_value = torch.stack(kv_parts, dim=2)

        b, n, d = query.shape
        local_query = query.reshape(b * n, 1, d)
        local_key_value = key_value.reshape(b * n, self.num_token_sources, d)
        mix_tokens = self.cross_attn(local_query, local_key_value).reshape(b, n, d)

        mix = mix_tokens.transpose(1, 2).reshape(b, self.dim, token_h, token_w)
        if self.upsample_mode == "nearest":
            mix = F.interpolate(mix, size=fused.shape[-2:], mode="nearest")
        else:
            mix = F.interpolate(
                mix, size=fused.shape[-2:], mode=self.upsample_mode, align_corners=False
            )
        mix = self.mix_proj(mix)
        gates = torch.sigmoid(self.gate_net(torch.cat([fused, mix], dim=1)))
        g_main = gates[:, 0:1]
        g_mix = gates[:, 1:2]
        return g_main * fused + g_mix * mix


class AlignedConcatAttentionFusion(nn.Module):
    """Position-aligned local token attention followed by concat projection.

    This keeps the 046 spatial assumption but removes the token sigmoid gate,
    so the token mix cannot be trivially multiplied away before the head.
    """

    def __init__(self, base_ch, token_channels, dim=96, token_source_ch=768,
                 num_heads=4, dropout=0.05, use_token_extractor=False,
                 token_extractor_mid_ch=None, upsample_mode="nearest",
                 token_calibration=False):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        self.dim = int(dim)
        self.upsample_mode = upsample_mode
        self.token_calibs = (
            nn.ModuleList(ChannelCalibration(token_source_ch)
                          for _ in range(self.num_token_sources))
            if token_calibration else None
        )

        self.main_proj = nn.Conv2d(base_ch, dim, 1)
        if use_token_extractor:
            self.token_projs = nn.ModuleList(
                _make_token_extractor(token_source_ch, dim, mid_ch=token_extractor_mid_ch)
                for _ in range(self.num_token_sources)
            )
        else:
            self.token_projs = nn.ModuleList(
                nn.Conv2d(token_source_ch, dim, 1)
                for _ in range(self.num_token_sources)
            )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.modality_embed = nn.Parameter(
            torch.zeros(1 + self.num_token_sources, dim)
        )
        nn.init.normal_(self.modality_embed, std=0.02)

        self.cross_attn = _CrossAttentionBlock(
            dim, num_heads=num_heads, dropout=dropout
        )
        self.mix_proj = nn.Sequential(
            ConvGNAct(dim, base_ch, kernel_size=3),
            ConvGNAct(base_ch, base_ch, kernel_size=3),
        )
        self.fusion_proj = nn.Conv2d(2 * base_ch, base_ch, 1)

    def _pos_map(self, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2)
        return self.pos_mlp(coords).transpose(1, 2).reshape(1, self.dim, h, w)

    def _tokens_from_map(self, feat, modality_idx):
        b, _, h, w = feat.shape
        feat = feat + self._pos_map(h, w, feat.device, feat.dtype)
        feat = feat + self.modality_embed[modality_idx].view(1, self.dim, 1, 1)
        return feat.flatten(2).transpose(1, 2)

    def _upsample_mix(self, mix, fused):
        if self.upsample_mode == "nearest":
            return F.interpolate(mix, size=fused.shape[-2:], mode="nearest")
        return F.interpolate(
            mix, size=fused.shape[-2:], mode=self.upsample_mode, align_corners=False
        )

    def _local_mix(self, fused, token):
        token_h, token_w = token.shape[-2:]
        main_feat = F.adaptive_avg_pool2d(fused, (token_h, token_w))
        query = self._tokens_from_map(self.main_proj(main_feat), 0)

        token_parts = torch.split(token, self.token_source_ch, dim=1)
        kv_parts = []
        for idx, (proj, token_part) in enumerate(zip(self.token_projs, token_parts), start=1):
            if self.token_calibs is not None:
                token_part = self.token_calibs[idx - 1](token_part)
            kv_parts.append(self._tokens_from_map(proj(token_part), idx))
        key_value = torch.stack(kv_parts, dim=2)

        b, n, d = query.shape
        local_query = query.reshape(b * n, 1, d)
        local_key_value = key_value.reshape(b * n, self.num_token_sources, d)
        mix_tokens = self.cross_attn(local_query, local_key_value).reshape(b, n, d)

        mix = mix_tokens.transpose(1, 2).reshape(b, self.dim, token_h, token_w)
        mix = self._upsample_mix(mix, fused)
        return self.mix_proj(mix)

    def forward(self, fused, token, return_mix=False):
        mix = self._local_mix(fused, token)
        out = self.fusion_proj(torch.cat([fused, mix], dim=1))
        if return_mix:
            return out, mix
        return out


class AlignedGroupedConcatAttentionFusion(nn.Module):
    """Aligned concat fusion with separate TerraMind and THOR token mixes.

    The expected source order is TerraMind-S1, TerraMind-S2, THOR-S1, THOR-S2.
    Each group attends locally within its two sources, then both token mixes are
    concatenated with the pixel trunk.
    """

    def __init__(self, base_ch, token_channels, dim=96, token_source_ch=768,
                 num_heads=4, dropout=0.05, use_token_extractor=False,
                 token_extractor_mid_ch=None, upsample_mode="nearest",
                 token_calibration=False):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        if self.num_token_sources != 4:
            raise ValueError(
                "AlignedGroupedConcatAttentionFusion expects 4 token sources: "
                "TerraMind-S1/S2 and THOR-S1/S2"
            )
        self.dim = int(dim)
        self.upsample_mode = upsample_mode
        self.token_calibs = (
            nn.ModuleList(ChannelCalibration(token_source_ch)
                          for _ in range(self.num_token_sources))
            if token_calibration else None
        )

        self.main_proj = nn.Conv2d(base_ch, dim, 1)
        if use_token_extractor:
            self.token_projs = nn.ModuleList(
                _make_token_extractor(token_source_ch, dim, mid_ch=token_extractor_mid_ch)
                for _ in range(self.num_token_sources)
            )
        else:
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

        self.cross_attn_tm = _CrossAttentionBlock(dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_thor = _CrossAttentionBlock(dim, num_heads=num_heads, dropout=dropout)
        self.mix_proj_tm = nn.Sequential(
            ConvGNAct(dim, base_ch, kernel_size=3),
            ConvGNAct(base_ch, base_ch, kernel_size=3),
        )
        self.mix_proj_thor = nn.Sequential(
            ConvGNAct(dim, base_ch, kernel_size=3),
            ConvGNAct(base_ch, base_ch, kernel_size=3),
        )
        self.fusion_proj = nn.Conv2d(3 * base_ch, base_ch, 1)

    def _pos_map(self, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2)
        return self.pos_mlp(coords).transpose(1, 2).reshape(1, self.dim, h, w)

    def _tokens_from_map(self, feat, modality_idx):
        b, _, h, w = feat.shape
        feat = feat + self._pos_map(h, w, feat.device, feat.dtype)
        feat = feat + self.modality_embed[modality_idx].view(1, self.dim, 1, 1)
        return feat.flatten(2).transpose(1, 2)

    def _upsample_mix(self, mix, fused):
        if self.upsample_mode == "nearest":
            return F.interpolate(mix, size=fused.shape[-2:], mode="nearest")
        return F.interpolate(
            mix, size=fused.shape[-2:], mode=self.upsample_mode, align_corners=False
        )

    def _attend_group(self, query, kv_parts, cross_attn, token_h, token_w, fused, mix_proj):
        key_value = torch.stack(kv_parts, dim=2)
        b, n, d = query.shape
        local_query = query.reshape(b * n, 1, d)
        local_key_value = key_value.reshape(b * n, key_value.size(2), d)
        mix_tokens = cross_attn(local_query, local_key_value).reshape(b, n, d)
        mix = mix_tokens.transpose(1, 2).reshape(b, self.dim, token_h, token_w)
        mix = self._upsample_mix(mix, fused)
        return mix_proj(mix)

    def forward(self, fused, token):
        token_h, token_w = token.shape[-2:]
        main_feat = F.adaptive_avg_pool2d(fused, (token_h, token_w))
        query = self._tokens_from_map(self.main_proj(main_feat), 0)

        token_parts = torch.split(token, self.token_source_ch, dim=1)
        kv_parts = []
        for idx, (proj, token_part) in enumerate(zip(self.token_projs, token_parts), start=1):
            if self.token_calibs is not None:
                token_part = self.token_calibs[idx - 1](token_part)
            kv_parts.append(self._tokens_from_map(proj(token_part), idx))

        mix_tm = self._attend_group(
            query, kv_parts[:2], self.cross_attn_tm,
            token_h, token_w, fused, self.mix_proj_tm
        )
        mix_thor = self._attend_group(
            query, kv_parts[2:], self.cross_attn_thor,
            token_h, token_w, fused, self.mix_proj_thor
        )
        return self.fusion_proj(torch.cat([fused, mix_tm, mix_thor], dim=1))


class ConcatAttentionFusion(nn.Module):
    """Like TwoGateAttentionFusion but replaces the two-gate residual add with
    a plain concat-then-project, so the attention output cannot be suppressed
    by a gate initialised to near-zero."""

    def __init__(self, base_ch, token_channels, dim=96, token_source_ch=768,
                 token_grid=32, num_heads=4, dropout=0.05,
                 use_token_extractor=False, token_extractor_mid_ch=None):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        self.token_grid = int(token_grid)
        self.dim = int(dim)

        self.main_pool = nn.AdaptiveAvgPool2d((self.token_grid, self.token_grid))
        self.main_proj = nn.Conv2d(base_ch, dim, 1)
        if use_token_extractor:
            self.token_projs = nn.ModuleList(
                _make_token_extractor(token_source_ch, dim, mid_ch=token_extractor_mid_ch)
                for _ in range(self.num_token_sources)
            )
        else:
            self.token_projs = nn.ModuleList(
                nn.Conv2d(token_source_ch, dim, 1)
                for _ in range(self.num_token_sources)
            )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.modality_embed = nn.Parameter(
            torch.zeros(1 + self.num_token_sources, dim)
        )
        nn.init.normal_(self.modality_embed, std=0.02)

        self.cross_attn = _CrossAttentionBlock(
            dim, num_heads=num_heads, dropout=dropout
        )
        self.mix_proj = nn.Sequential(
            ConvGNAct(dim, base_ch, kernel_size=3),
            ConvGNAct(base_ch, base_ch, kernel_size=3),
        )
        self.fusion_proj = nn.Conv2d(2 * base_ch, base_ch, 1)

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

    def forward(self, fused, token):
        main_feat = self.main_proj(self.main_pool(fused))
        query = self._tokens_from_map(main_feat, 0)
        token_parts = torch.split(token, self.token_source_ch, dim=1)
        kv_tokens = []
        for idx, (proj, token_part) in enumerate(zip(self.token_projs, token_parts), start=1):
            kv_tokens.append(self._tokens_from_map(proj(token_part), idx))
        key_value = torch.cat(kv_tokens, dim=1)
        mix_tokens = self.cross_attn(query, key_value)

        b = fused.size(0)
        mix = mix_tokens.transpose(1, 2).reshape(
            b, self.dim, self.token_grid, self.token_grid
        )
        mix = F.interpolate(
            mix, size=fused.shape[-2:], mode="bilinear", align_corners=False
        )
        mix = self.mix_proj(mix)
        return self.fusion_proj(torch.cat([fused, mix], dim=1))


class ResidualConcatAttentionFusion(nn.Module):
    """Like ConcatAttentionFusion but uses fused + proj(cat([fused, mix])) with
    zero-init on proj, so training starts as identity and gradually absorbs
    the attention output without a sigmoid bottleneck."""

    def __init__(self, base_ch, token_channels, dim=96, token_source_ch=768,
                 token_grid=32, num_heads=4, dropout=0.05,
                 use_token_extractor=False, token_extractor_mid_ch=None):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        self.token_grid = int(token_grid)
        self.dim = int(dim)

        self.main_pool = nn.AdaptiveAvgPool2d((self.token_grid, self.token_grid))
        self.main_proj = nn.Conv2d(base_ch, dim, 1)
        if use_token_extractor:
            self.token_projs = nn.ModuleList(
                _make_token_extractor(token_source_ch, dim, mid_ch=token_extractor_mid_ch)
                for _ in range(self.num_token_sources)
            )
        else:
            self.token_projs = nn.ModuleList(
                nn.Conv2d(token_source_ch, dim, 1)
                for _ in range(self.num_token_sources)
            )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.modality_embed = nn.Parameter(
            torch.zeros(1 + self.num_token_sources, dim)
        )
        nn.init.normal_(self.modality_embed, std=0.02)

        self.cross_attn = _CrossAttentionBlock(
            dim, num_heads=num_heads, dropout=dropout
        )
        self.mix_proj = nn.Sequential(
            ConvGNAct(dim, base_ch, kernel_size=3),
            ConvGNAct(base_ch, base_ch, kernel_size=3),
        )
        self.fusion_proj = nn.Conv2d(2 * base_ch, base_ch, 1)
        nn.init.zeros_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)

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

    def forward(self, fused, token):
        main_feat = self.main_proj(self.main_pool(fused))
        query = self._tokens_from_map(main_feat, 0)
        token_parts = torch.split(token, self.token_source_ch, dim=1)
        kv_tokens = []
        for idx, (proj, token_part) in enumerate(zip(self.token_projs, token_parts), start=1):
            kv_tokens.append(self._tokens_from_map(proj(token_part), idx))
        key_value = torch.cat(kv_tokens, dim=1)
        mix_tokens = self.cross_attn(query, key_value)

        b = fused.size(0)
        mix = mix_tokens.transpose(1, 2).reshape(
            b, self.dim, self.token_grid, self.token_grid
        )
        mix = F.interpolate(
            mix, size=fused.shape[-2:], mode="bilinear", align_corners=False
        )
        mix = self.mix_proj(mix)
        return fused + self.fusion_proj(torch.cat([fused, mix], dim=1))


class GatedPixelFusionTwoGateAttentionLightUNet(nn.Module):
    """AE+Tessera gated trunk with untied two-gate token attention fusion."""

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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=32, token_calibration=False,
                 token_gate_init_bias=(2.0, -2.0)):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionTwoGateAttentionLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionTwoGateAttentionLightUNet expects AlphaEarth+Tessera "
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
        self.token_fusion = TwoGateAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_grid=token_query_grid,
            token_calibration=token_calibration,
            token_gate_init_bias=token_gate_init_bias,
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
            raise ValueError("GatedPixelFusionTwoGateAttentionLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.token_fusion(fused, token)
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        return self.head(fused, return_aux=return_aux,
                         presence_extra=presence_extra)


class GatedPixelFusionCQATwoGateAttentionLightUNet(
    GatedPixelFusionTwoGateAttentionLightUNet
):
    """xfusion_027 with contribution-quality-complementarity token routing."""

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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=32, token_calibration=False,
                 token_gate_init_bias=(2.0, -2.0),
                 cqa_source_dropout=0.0):
        super().__init__(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            alpha_channels=alpha_channels,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=base_ch,
            gate_init_bias=gate_init_bias,
            gate_mode=gate_mode,
            gate_untied=gate_untied,
            modality_dropout=modality_dropout,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            norm_kind=norm_kind,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_query_grid=token_query_grid,
            token_calibration=token_calibration,
            token_gate_init_bias=token_gate_init_bias,
        )
        self.token_fusion = CQAWeightedTwoGateAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_grid=token_query_grid,
            token_calibration=token_calibration,
            token_gate_init_bias=token_gate_init_bias,
            cqa_source_dropout=cqa_source_dropout,
        )


class GatedPixelFusionAlignedTwoGateAttentionLightUNet(
    GatedPixelFusionTwoGateAttentionLightUNet
):
    """xfusion_027 pixel path with position-aligned local token attention."""

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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=16, token_calibration=False,
                 token_gate_init_bias=(2.0, -2.0)):
        super().__init__(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            alpha_channels=alpha_channels,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=base_ch,
            gate_init_bias=gate_init_bias,
            gate_mode=gate_mode,
            gate_untied=gate_untied,
            modality_dropout=modality_dropout,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            norm_kind=norm_kind,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_query_grid=token_query_grid,
            token_calibration=token_calibration,
            token_gate_init_bias=token_gate_init_bias,
        )
        self.token_fusion = AlignedTwoGateAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_calibration=token_calibration,
            token_gate_init_bias=token_gate_init_bias,
        )


class GatedPixelFusionAlignedConcatAttentionLightUNet(
    GatedPixelFusionTwoGateAttentionLightUNet
):
    """046 pixel path with aligned local token attention and concat projection."""

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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=16, token_calibration=False,
                 **unused):
        super().__init__(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            alpha_channels=alpha_channels,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=base_ch,
            gate_init_bias=gate_init_bias,
            gate_mode=gate_mode,
            gate_untied=gate_untied,
            modality_dropout=modality_dropout,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            norm_kind=norm_kind,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_query_grid=token_query_grid,
            token_calibration=token_calibration,
        )
        self.token_fusion = AlignedConcatAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_calibration=token_calibration,
        )


class GatedPixelFusionAlignedGroupedConcatAttentionLightUNet(
    GatedPixelFusionAlignedConcatAttentionLightUNet
):
    """046 pixel path with separate aligned TerraMind and THOR token mixes."""

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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=16, token_calibration=False,
                 **unused):
        super().__init__(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            alpha_channels=alpha_channels,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=base_ch,
            gate_init_bias=gate_init_bias,
            gate_mode=gate_mode,
            gate_untied=gate_untied,
            modality_dropout=modality_dropout,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            norm_kind=norm_kind,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_query_grid=token_query_grid,
            token_calibration=token_calibration,
        )
        self.token_fusion = AlignedGroupedConcatAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_calibration=token_calibration,
        )


class GatedPixelFusionAlignedConcatAuxLightUNet(
    GatedPixelFusionAlignedConcatAttentionLightUNet
):
    """049 plus an auxiliary prediction head directly on the aligned token mix."""

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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=16, token_calibration=False,
                 token_aux_weight=0.2, **unused):
        super().__init__(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            alpha_channels=alpha_channels,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=base_ch,
            gate_init_bias=gate_init_bias,
            gate_mode=gate_mode,
            gate_untied=gate_untied,
            modality_dropout=modality_dropout,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            norm_kind=norm_kind,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_query_grid=token_query_grid,
            token_calibration=token_calibration,
        )
        self.token_aux_weight = float(token_aux_weight)
        self.token_aux_head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            presence_extra_ch=0,
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
            raise ValueError("GatedPixelFusionAlignedConcatAuxLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused, token_mix = self.token_fusion(fused, token, return_mix=True)
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        main = self.head(fused, return_aux=return_aux, presence_extra=presence_extra)
        if not return_aux:
            return main

        token_aux = self.token_aux_head(token_mix, return_aux=True)
        main["token_aux"] = token_aux
        main["token_aux_weight"] = self.token_aux_weight
        return main


# ── xfusion_030: grouped intra-model gated fusion ────────────────────────────

class TwoGateAttentionGroupedFusion(nn.Module):
    """Two-gate attention with intra-group (TerraM: S1+S2, THOR: S1+S2) rich gates.

    Each source gets a per-source extractor (768→96).  S1 and S2 within each
    foundation model are then fused with a rich untied gate before cross-
    attention.  The final cross-attention query is the main (AE+Tessera) trunk;
    KV is the two group-fused token maps (512 tokens instead of 1024).
    """

    def __init__(self, base_ch, token_channels, dim=96, token_source_ch=768,
                 token_grid=32, num_heads=4, dropout=0.05,
                 token_extractor_mid_ch=None):
        super().__init__()
        if token_channels != 4 * token_source_ch:
            raise ValueError(
                "TwoGateAttentionGroupedFusion requires exactly 4 token sources "
                f"(token_channels={token_channels} != 4*{token_source_ch})"
            )
        self.token_source_ch = int(token_source_ch)
        self.token_grid = int(token_grid)
        self.dim = int(dim)

        # per-source extractors
        self.token_projs = nn.ModuleList(
            _make_token_extractor(token_source_ch, dim, mid_ch=token_extractor_mid_ch)
            for _ in range(4)
        )

        # intra-group rich untied gates: g_a*a + g_b*b (no sum-to-1 constraint)
        self.group_gate_tm   = _build_fusion_gate(dim, mode="rich", untied=True, init_bias=4.0)
        self.group_gate_thor = _build_fusion_gate(dim, mode="rich", untied=True, init_bias=4.0)

        # main query projection
        self.main_pool = nn.AdaptiveAvgPool2d((self.token_grid, self.token_grid))
        self.main_proj = nn.Conv2d(base_ch, dim, 1)

        # positional encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # modality embed: main(0), TerraM-group(1), THOR-group(2)
        self.modality_embed = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.modality_embed, std=0.02)

        self.cross_attn = _CrossAttentionBlock(dim, num_heads=num_heads, dropout=dropout)
        self.mix_proj = nn.Sequential(
            ConvGNAct(dim, base_ch, kernel_size=3),
            ConvGNAct(base_ch, base_ch, kernel_size=3),
        )
        gate_hidden = max(base_ch, 32)
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * base_ch, gate_hidden, 1, bias=False),
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

    def forward(self, fused, token):
        parts = torch.split(token, self.token_source_ch, dim=1)   # 4 × [B, 768, h, w]
        feats = [proj(p) for proj, p in zip(self.token_projs, parts)]  # 4 × [B, 96, h, w]

        # intra-group fusion: TerraM (0,1) and THOR (2,3)
        tm_fused   = _apply_fusion_gate(self.group_gate_tm,   feats[0], feats[1], untied=True)
        thor_fused = _apply_fusion_gate(self.group_gate_thor, feats[2], feats[3], untied=True)

        # cross-attention
        main_feat = self.main_proj(self.main_pool(fused))
        query     = self._tokens_from_map(main_feat, 0)         # [B, T_q, dim]
        kv_tm     = self._tokens_from_map(tm_fused,   1)        # [B, T_k, dim]
        kv_thor   = self._tokens_from_map(thor_fused, 2)        # [B, T_k, dim]
        key_value = torch.cat([kv_tm, kv_thor], dim=1)          # [B, 2*T_k, dim]

        mix_tokens = self.cross_attn(query, key_value)
        b = fused.size(0)
        mix = mix_tokens.transpose(1, 2).reshape(b, self.dim, self.token_grid, self.token_grid)
        mix = F.interpolate(mix, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        mix = self.mix_proj(mix)

        gates = torch.sigmoid(self.gate_net(torch.cat([fused, mix], dim=1)))
        return gates[:, 0:1] * fused + gates[:, 1:2] * mix


class GatedPixelFusionTwoGateGroupedLightUNet(GatedPixelFusionTwoGateAttentionLightUNet):
    """TwoGate variant with intra-group (TerraM, THOR) gated fusion before cross-attention."""

    def __init__(self, pixel_channels, token_channels, base_ch=32,
                 token_extractor_mid_ch=None, **kwargs):
        kwargs.pop("use_token_extractor", None)
        super().__init__(pixel_channels, token_channels, base_ch=base_ch,
                         use_token_extractor=False, **kwargs)
        self.token_fusion = TwoGateAttentionGroupedFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            token_extractor_mid_ch=token_extractor_mid_ch,
        )


# ── xfusion_033: bottleneck-level token cross-attention ──────────────────────

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

        # No AdaptiveAvgPool — bottleneck is already at the right spatial scale
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

        # Project attention output back to bottleneck_ch (no upsample needed)
        self.mix_proj = nn.Sequential(
            ConvGNAct(dim, dim, kernel_size=3),
            nn.Conv2d(dim, bottleneck_ch, 1),
        )

        # Two-gate blend on bottleneck_ch; use lighter hidden dim
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

        # Project bottleneck to dim — no pool
        main_feat = self.main_proj(bottleneck)          # [B, dim, h, w]
        query = self._tokens_from_map(main_feat, 0)     # [B, h*w, dim]

        token_parts = torch.split(token, self.token_source_ch, dim=1)
        kv_tokens = [
            self._tokens_from_map(proj(tp), idx)
            for idx, (proj, tp) in enumerate(zip(self.token_projs, token_parts), start=1)
        ]
        key_value = torch.cat(kv_tokens, dim=1)         # [B, N_src*h_t*w_t, dim]

        mix_tokens = self.cross_attn(query, key_value)  # [B, h*w, dim]

        # Reshape back — no upsample, stays at bottleneck spatial size
        mix = mix_tokens.transpose(1, 2).reshape(b, self.dim, h, w)
        mix = self.mix_proj(mix)                        # [B, bottleneck_ch, h, w]

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
        # Token attention at bottleneck — 1×1 proj only (no extractor), same as 027
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

        # Encoder
        x1 = self.alpha_unet.inc(alpha)
        x2 = self.alpha_unet.down1(x1)
        x3 = self.alpha_unet.down2(x2)
        x4 = self.alpha_unet.down3(x3)     # [bottleneck_ch, 32×32]

        # Token injection at bottleneck — token signal flows through entire decoder
        x4 = self.bottleneck_token_fusion(x4, token)

        # Decoder (same skip connections as LightUNet)
        feat = self.alpha_unet.up1(x4)
        feat = torch.cat([x3, feat], dim=1)
        feat = self.alpha_unet.conv1(feat)

        feat = self.alpha_unet.up2(feat)
        feat = torch.cat([x2, feat], dim=1)
        feat = self.alpha_unet.conv2(feat)

        feat = self.alpha_unet.up3(feat)
        feat = torch.cat([x1, feat], dim=1)
        feat = self.alpha_unet.conv3(feat)  # [base_ch, 256×256]

        # Tessera gate fusion at output level (same as 027)
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


# ── xfusion_035: per-modality independent attention + zero-init gates ─────────

class PerModalityGatedFusion(nn.Module):
    """Independent cross-attention and zero-init scalar gate per token source.

    Each modality gets its own:
      - 1×1 token projection (768 → dim): controls representation form
      - _CrossAttentionBlock: independent routing, not coupled by joint softmax
      - lightweight output projection (dim → base_ch)
      - learnable scalar gate g_i, zero-initialized

    Output = fused + Σ_i  g_i × upsample(out_proj_i(cross_attn_i(Q, KV_i)))

    At init all g_i=0, so output = fused (identical to pure-pixel baseline).
    Each source independently learns how much to contribute; a globally
    unhelpful source (e.g. TM-S2 spatial noise) can converge to g_i ≈ 0
    without penalising other sources.
    """

    def __init__(self, base_ch, token_channels, dim=96,
                 token_source_ch=768, token_grid=32, num_heads=4, dropout=0.05):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        self.token_grid = int(token_grid)
        self.dim = int(dim)

        # Shared query path
        self.main_pool = nn.AdaptiveAvgPool2d((self.token_grid, self.token_grid))
        self.main_proj = nn.Conv2d(base_ch, dim, 1)

        # Shared positional encoding + per-source modality embed
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.modality_embed = nn.Parameter(
            torch.zeros(1 + self.num_token_sources, dim)
        )
        nn.init.normal_(self.modality_embed, std=0.02)

        # Per-source components
        self.token_projs = nn.ModuleList(
            nn.Conv2d(token_source_ch, dim, 1)
            for _ in range(self.num_token_sources)
        )
        self.cross_attns = nn.ModuleList(
            _CrossAttentionBlock(dim, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_token_sources)
        )
        self.out_projs = nn.ModuleList(
            nn.Sequential(
                ConvGNAct(dim, base_ch, kernel_size=3),
                nn.Conv2d(base_ch, base_ch, 1),
            )
            for _ in range(self.num_token_sources)
        )
        # Zero-init scalar gates — each source starts with zero contribution
        self.source_gates = nn.ParameterList(
            nn.Parameter(torch.zeros(1))
            for _ in range(self.num_token_sources)
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

    def _source_deltas(self, fused, token):
        b, _, h, w = fused.shape
        orig_dtype = fused.dtype

        # Run entirely in float32: some token sources (e.g. THOR S2) have
        # extreme embedding values (~21674). With 768 input channels, the
        # Conv2d(768→96) projection can produce sums up to ~600,000 in float16
        # — exceeding its max (~65504) and producing inf. That inf then enters
        # LayerNorm, where (inf − mean) / std = NaN.
        with torch.amp.autocast("cuda", enabled=False):
            fused_f = fused.float()
            token_f = token.float()

            # Shared query (pooled pixel trunk)
            main_feat = self.main_proj(self.main_pool(fused_f))
            query = self._tokens_from_map(main_feat, 0)        # [B, tg², dim]

            token_parts = torch.split(token_f, self.token_source_ch, dim=1)

            deltas = []
            for i, (tok_proj, cross_attn, out_proj, tok) in enumerate(
                zip(self.token_projs, self.cross_attns,
                    self.out_projs, token_parts)
            ):
                kv = self._tokens_from_map(tok_proj(tok), i + 1)   # [B, 256, dim]
                attn_out = cross_attn(query, kv)                    # [B, tg², dim]

                attn_spatial = attn_out.transpose(1, 2).reshape(
                    b, self.dim, self.token_grid, self.token_grid
                )
                attn_spatial = F.interpolate(
                    attn_spatial, size=(h, w), mode="bilinear", align_corners=False
                )
                delta = out_proj(attn_spatial)                      # [B, base_ch, H, W]
                deltas.append(delta.to(orig_dtype))

        return deltas

    def forward(self, fused, token):
        output = fused
        for gate, delta in zip(self.source_gates, self._source_deltas(fused, token)):
            output = output + gate * delta

        return output


class TaskConditionedSourceFusion(PerModalityGatedFusion):
    """Per-source token adapters with explicit task-source routing gates.

    Returns one feature map per task:
        F_task = fused + sum_s gate[task, source_s] * delta_s

    Gates are zero-initialized by default, so this starts as the pixel-fused
    trunk and lets each task learn which token sources are worth using.
    """

    def __init__(self, *args, task_names=("building", "tree", "water", "height"),
                 task_gate_init=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_names = tuple(task_names)
        self.task_source_gates = nn.Parameter(
            torch.full(
                (len(self.task_names), self.num_token_sources),
                float(task_gate_init),
            )
        )
        self.last_task_source_gates = None

    def forward(self, fused, token):
        deltas = self._source_deltas(fused, token)
        stacked = torch.stack(deltas, dim=1)
        gates = self.task_source_gates
        self.last_task_source_gates = gates.detach()

        task_features = {}
        for task_idx, task_name in enumerate(self.task_names):
            weights = gates[task_idx].view(1, self.num_token_sources, 1, 1, 1)
            residual = (stacked * weights).sum(dim=1)
            task_features[task_name] = fused + residual
        return task_features


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

class GatedPixelFusionPerModalityLightUNet(nn.Module):
    """027 architecture with per-modality independent attention + zero-init gates.

    Replaces the single joint TwoGateAttentionFusion with PerModalityGatedFusion:
    each token source has its own cross-attention and zero-initialized scalar gate,
    giving the model explicit per-modality quantity and form control.
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
                 presence_branch_ch=None, token_query_grid=32, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionPerModalityLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionPerModalityLightUNet expects AlphaEarth+Tessera "
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
        self.token_fusion = PerModalityGatedFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            token_grid=token_query_grid,
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
            raise ValueError("GatedPixelFusionPerModalityLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.token_fusion(fused, token)
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        return self.head(fused, return_aux=return_aux, presence_extra=presence_extra)


class GatedPixelFusionTaskRouterLightUNet(nn.Module):
    """027-style pixel trunk with explicit token-source to task gates.

    AE and Tessera are fused exactly as in xfusion_027.  Each token source then
    gets an independent attention adapter, and a small task-source gate matrix
    routes those residuals into building, tree, water, and height features
    before the multi-task head.
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
                 presence_branch_ch=None, token_query_grid=32, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionTaskRouterLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionTaskRouterLightUNet expects AlphaEarth+Tessera "
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
        self.token_fusion = TaskConditionedSourceFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            token_grid=token_query_grid,
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
            raise ValueError("GatedPixelFusionTaskRouterLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        task_features = self.token_fusion(fused, token)
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        return self.head(
            fused,
            return_aux=return_aux,
            presence_extra=presence_extra,
            task_features=task_features,
        )


# ── xfusion_037: TwoGate attention + Tessera LightUNet (replaces CompressionStem) ──

class GatedPixelFusionTwoGateAttentionUNetLightUNet(nn.Module):
    """xfusion_027 with Tessera full LightUNet replacing TesseraCompressionStem.

    Pixel path: AE → LightUNet; Tessera → ChannelCalib + LightUNet (as in
    ae_tessera_unet_gated). Rich gate fuses both at output resolution.
    Token path: unchanged from 027 — TwoGateAttentionFusion over 4 sources.
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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=32, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionTwoGateAttentionUNetLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionTwoGateAttentionUNetLightUNet expects AlphaEarth+Tessera "
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

        # Full LightUNet for Tessera (ae_tessera_unet_gated style)
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
        self.token_fusion = TwoGateAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_grid=token_query_grid,
        )

        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
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
            raise ValueError("GatedPixelFusionTwoGateAttentionUNetLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_unet.forward_features(self.tessera_entry(tessera))
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.token_fusion(fused, token)
        return self.head(fused, return_aux=return_aux)


# ── xfusion_038: PerModality zero-init gates + Tessera LightUNet ─────────────

class GatedPixelFusionPerModalityUNetLightUNet(nn.Module):
    """Tessera LightUNet (037 style) + PerModalityGatedFusion (zero-init additive gates).

    Fixes modality-ignore vs 037: replaces the competitive TwoGate blend with
    independent zero-initialized scalar gates per token source.  Each source
    starts contributing nothing (g_i=0) and independently learns its residual
    — no competition with the main trunk suppresses it.  Tessera modality
    dropout further forces the model to lean on tokens when tessera is masked.
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
                 presence_branch_ch=None, token_query_grid=32, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionPerModalityUNetLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionPerModalityUNetLightUNet expects AlphaEarth+Tessera "
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
            base_ch, mode=gate_mode, untied=self.gate_untied, init_bias=gate_init_bias
        )
        # Additive zero-init per-source gates — no competition with main trunk
        self.token_fusion = PerModalityGatedFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            token_grid=token_query_grid,
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
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
            raise ValueError("GatedPixelFusionPerModalityUNetLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_unet.forward_features(self.tessera_entry(tessera))
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.token_fusion(fused, token)
        return self.head(fused, return_aux=return_aux)


# ── xfusion_039: Bottleneck token injection + Tessera LightUNet ──────────────

class GatedPixelFusionBnAttentionUNetLightUNet(nn.Module):
    """xfusion_033 bottleneck token injection with Tessera full LightUNet.

    Token cross-attention at the AE UNet bottleneck (32×32) — token signal
    propagates through all decoder skip connections before the output-level
    Tessera gate fusion.  Tessera path uses a full LightUNet (same as 037/038)
    instead of TesseraCompressionStem.
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
            raise ValueError("GatedPixelFusionBnAttentionUNetLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionBnAttentionUNetLightUNet expects AlphaEarth+Tessera "
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

        self.tessera_entry = nn.Sequential(
            ChannelCalibration(tessera_channels),
            ConvGNAct(tessera_channels, tessera_channels, kernel_size=1, padding=0),
        )
        self.tessera_unet = LightUNet(
            tessera_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind
        )
        self.tessera_unet.head = nn.Identity()

        self.gate_conv = _build_fusion_gate(
            base_ch, mode=gate_mode, untied=self.gate_untied, init_bias=gate_init_bias
        )
        # Token cross-attention at bottleneck: token signal flows through decoder
        self.bottleneck_token_fusion = BottleneckTwoGateAttentionFusion(
            bottleneck_ch=bottleneck_ch,
            token_channels=token_channels,
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
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
            raise ValueError("GatedPixelFusionBnAttentionUNetLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        # AE encoder
        x1 = self.alpha_unet.inc(alpha)
        x2 = self.alpha_unet.down1(x1)
        x3 = self.alpha_unet.down2(x2)
        x4 = self.alpha_unet.down3(x3)

        # Token injection at bottleneck — propagates through all decoder skips
        x4 = self.bottleneck_token_fusion(x4, token)

        # AE decoder
        feat = self.alpha_unet.up1(x4)
        feat = torch.cat([x3, feat], dim=1)
        feat = self.alpha_unet.conv1(feat)
        feat = self.alpha_unet.up2(feat)
        feat = torch.cat([x2, feat], dim=1)
        feat = self.alpha_unet.conv2(feat)
        feat = self.alpha_unet.up3(feat)
        feat = torch.cat([x1, feat], dim=1)
        feat = self.alpha_unet.conv3(feat)

        # Tessera UNet gate fusion at output level
        tessera_feat = self.tessera_unet.forward_features(self.tessera_entry(tessera))
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, feat, tessera_feat, untied=self.gate_untied
        )
        return self.head(fused, return_aux=return_aux)


# ── xfusion_040: ConcatAttentionFusion + TesseraCompressionStem (027-style) ──

class GatedPixelFusionConcatAttentionLightUNet(nn.Module):
    """xfusion_027 with the two-gate token blend replaced by concat-then-project.

    Pixel path: AE → LightUNet; Tessera → TesseraCompressionStem (same as 027).
    Token path: ConcatAttentionFusion — cross-attention output is concatenated
    with the trunk feature and projected back to base_ch, removing the sigmoid
    gate that suppresses weak THOR/TerraMind signals at initialisation.
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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=32, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionConcatAttentionLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionConcatAttentionLightUNet expects AlphaEarth+Tessera "
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
        self.tessera_feature_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=base_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        self.gate_conv = _build_fusion_gate(
            base_ch, mode=gate_mode, untied=self.gate_untied, init_bias=gate_init_bias,
        )
        self.token_fusion = ConcatAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_grid=token_query_grid,
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
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
            raise ValueError("GatedPixelFusionConcatAttentionLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.token_fusion(fused, token)
        return self.head(fused, return_aux=return_aux)


# ── xfusion_041: ConcatAttentionFusion + Tessera LightUNet (037-style) ───────

class GatedPixelFusionConcatAttentionUNetLightUNet(nn.Module):
    """xfusion_037 with the two-gate token blend replaced by concat-then-project.

    Pixel path: AE → LightUNet; Tessera → ChannelCalib + LightUNet (037 style).
    Token path: ConcatAttentionFusion — no sigmoid gate on attention output.
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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=32, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionConcatAttentionUNetLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionConcatAttentionUNetLightUNet expects AlphaEarth+Tessera "
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
            base_ch, mode=gate_mode, untied=self.gate_untied, init_bias=gate_init_bias,
        )
        self.token_fusion = ConcatAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_grid=token_query_grid,
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
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
            raise ValueError("GatedPixelFusionConcatAttentionUNetLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_unet.forward_features(self.tessera_entry(tessera))
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.token_fusion(fused, token)
        return self.head(fused, return_aux=return_aux)


# ── xfusion_044: ResidualConcatAttentionFusion + TesseraCompressionStem (027-style) ──

class GatedPixelFusionResidualConcatAttentionLightUNet(nn.Module):
    """xfusion_027 with the two-gate token blend replaced by residual concat.

    Pixel path identical to 027 (AE LightUNet + TesseraCompressionStem).
    Token path: ResidualConcatAttentionFusion — fused + zero-init-proj(cat([fused, mix])).
    Starts as identity, learns to incorporate attention output without sigmoid suppression.
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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=32, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionResidualConcatAttentionLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionResidualConcatAttentionLightUNet expects AlphaEarth+Tessera "
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
        self.tessera_feature_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=base_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        self.gate_conv = _build_fusion_gate(
            base_ch, mode=gate_mode, untied=self.gate_untied, init_bias=gate_init_bias,
        )
        self.token_fusion = ResidualConcatAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_grid=token_query_grid,
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
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
            raise ValueError("GatedPixelFusionResidualConcatAttentionLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.token_fusion(fused, token)
        return self.head(fused, return_aux=return_aux)


# ── xfusion_045: ResidualConcatAttentionFusion + Tessera LightUNet (037-style) ──

class GatedPixelFusionResidualConcatAttentionUNetLightUNet(nn.Module):
    """xfusion_037 with the two-gate token blend replaced by residual concat.

    Pixel path: AE LightUNet + Tessera full LightUNet (037 style).
    Token path: ResidualConcatAttentionFusion — fused + zero-init-proj(cat([fused, mix])).
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
                 presence_branch_ch=None,
                 use_token_extractor=False, token_extractor_mid_ch=None,
                 token_query_grid=32, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("GatedPixelFusionResidualConcatAttentionUNetLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionResidualConcatAttentionUNetLightUNet expects AlphaEarth+Tessera "
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
            base_ch, mode=gate_mode, untied=self.gate_untied, init_bias=gate_init_bias,
        )
        self.token_fusion = ResidualConcatAttentionFusion(
            base_ch=base_ch,
            token_channels=token_channels,
            use_token_extractor=use_token_extractor,
            token_extractor_mid_ch=token_extractor_mid_ch,
            token_grid=token_query_grid,
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
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
            raise ValueError("GatedPixelFusionResidualConcatAttentionUNetLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_unet.forward_features(self.tessera_entry(tessera))
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.token_fusion(fused, token)
        return self.head(fused, return_aux=return_aux)


# ─────────────────────────────────────────────────────────────────────────────
#  Six-Modal Bottleneck Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class SixModalBottleneckFusion(nn.Module):
    """All 6 modalities fused via self-attention at the AE bottleneck resolution.

    Projects AE bottleneck (x4), Tessera (pooled to bottleneck grid), and all
    token sources to a shared dim, runs one TransformerBlock (self-attention +
    FFN), reads out with learned per-modality weights, then projects back to
    ae_bn_ch.  Zero-initialised output projection ensures x4_joint ≈ x4 at
    training start (residual LoRA-style stability).
    """

    def __init__(self, ae_bn_ch, tess_ch, token_channels, dim=96,
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
        self.ae_bn_ch = int(ae_bn_ch)
        n_modalities = 2 + self.num_token_sources  # AE + Tessera + token srcs

        self.proj_ae = nn.Conv2d(ae_bn_ch, dim, 1)
        self.proj_tess = nn.Conv2d(tess_ch, dim, 1)
        self.token_projs = nn.ModuleList(
            nn.Conv2d(token_source_ch, dim, 1)
            for _ in range(self.num_token_sources)
        )

        self.modality_embed = nn.Parameter(torch.zeros(n_modalities, dim))
        nn.init.normal_(self.modality_embed, std=0.02)

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.attn_block = _LatentTransformerBlock(dim, num_heads=num_heads, dropout=dropout)

        # Uniform readout at init (all zeros → softmax = 1/n)
        self.readout_logits = nn.Parameter(torch.zeros(n_modalities))

        # Zero-init: x4_joint = x4 + 0 at start → stable decoder warmup
        self.out_proj = nn.Conv2d(dim, ae_bn_ch, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _pos_tokens(self, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2)
        return self.pos_mlp(coords)

    def _to_tokens(self, feat, mod_idx):
        b, _, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # [B, H*W, dim]
        tokens = tokens + self._pos_tokens(h, w, feat.device, feat.dtype)
        return tokens + self.modality_embed[mod_idx].view(1, 1, -1)

    def forward(self, x4, tessera_bn, token):
        b, _, h, w = x4.shape
        input_dtype = x4.dtype

        # Run the entire fusion in fp32.
        # Rationale: fp16 can overflow (max ~65504) in both the projectors (384→96
        # on the raw AE bottleneck) and the 6144-token self-attention QK^T matrix.
        # AMP keeps model weights in fp32, so autocast(enabled=False) + explicit
        # .float() inputs gives a clean fp32 forward at moderate extra cost.
        with torch.amp.autocast("cuda", enabled=False):
            x4_f       = x4.float()
            tessera_f  = tessera_bn.float()
            token_f    = token.float()

            ae_tok = self._to_tokens(self.proj_ae(x4_f), 0)
            te_tok = self._to_tokens(self.proj_tess(tessera_f), 1)

            token_parts = torch.split(token_f, self.token_source_ch, dim=1)
            tok_list = [
                self._to_tokens(proj(part), 2 + i)
                for i, (proj, part) in enumerate(zip(self.token_projs, token_parts))
            ]

            # [B, n_modalities × H*W, dim]
            all_tokens = torch.cat([ae_tok, te_tok] + tok_list, dim=1)
            fused = self.attn_block(all_tokens)

            # Per-modality weighted readout
            n_tok = h * w
            chunks = fused.split(n_tok, dim=1)
            weights = torch.softmax(self.readout_logits, dim=0)
            joint = sum(w_i * c for w_i, c in zip(weights, chunks))  # [B, H*W, dim]

            joint = joint.transpose(1, 2).reshape(b, self.dim, h, w)
            result = x4_f + self.out_proj(joint)  # residual onto AE bottleneck

        return result.to(input_dtype)


class SixModalBottleneckLightUNet(nn.Module):
    """Six-modal bottleneck self-attention + AE skip decoder.

    Replaces the two-stage pixel-gate → token-correction pipeline of
    xfusion_027 with a single symmetric bottleneck fusion where all six
    modalities (AE, Tessera, TM-S1/S2, THOR-S1/S2) jointly attend at the AE
    bottleneck resolution (H/8 × W/8).  AE encoder skip connections (x1–x3)
    are preserved for full-resolution spatial reconstruction.
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None,
                 modality_dropout=0.0,
                 fusion_dim=96, fusion_num_heads=4, fusion_dropout=0.05,
                 token_source_ch=768):
        super().__init__()
        if n_classes != 4:
            raise ValueError("SixModalBottleneckLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                f"SixModalBottleneckLightUNet expects AlphaEarth+Tessera pixel "
                f"input with >{alpha_channels} channels, got {pixel_channels}"
            )

        self.supports_aux_outputs = True
        self.alpha_channels = int(alpha_channels)
        self.modality_dropout = float(modality_dropout)
        tessera_channels = pixel_channels - alpha_channels
        c4 = base_ch * 8  # AE bottleneck channels

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

        self.six_fusion = SixModalBottleneckFusion(
            ae_bn_ch=c4,
            tess_ch=base_ch,
            token_channels=token_channels,
            dim=fusion_dim,
            token_source_ch=token_source_ch,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )

        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
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
            raise ValueError("SixModalBottleneckLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        # AE encoder: keep all skip connections
        x1, x2, x3, x4 = self.alpha_unet.encode_features(alpha)

        # Tessera: compress then pool to bottleneck spatial size
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(tessera_feat, self.modality_dropout, self.training)
        tessera_bn = F.adaptive_avg_pool2d(tessera_feat, x4.shape[-2:])

        # Joint bottleneck fusion (all 6 modalities)
        x4_joint = self.six_fusion(x4, tessera_bn, token)

        # AE decoder with fused bottleneck + original skips
        full_feat = self.alpha_unet.decode_features((x1, x2, x3, x4_joint))

        return self.head(full_feat, return_aux=return_aux)


# ─────────────────────────────────────────────────────────────────────────────
#  Error-router A: frozen AE+Tessera baseline + token height correction
# ─────────────────────────────────────────────────────────────────────────────

class TokenHeightCorrectionLightUNet(nn.Module):
    """Height-only token correction on top of a protected AE+Tessera baseline.

    The baseline path intentionally reuses ``TesseraUNetGatedLightUNet`` module
    names (``alpha_unet``, ``tessera_entry``, ``tessera_unet``, ``gate_conv``,
    ``head``) so ``ae_tessera_unet_gated_v003`` checkpoints load directly with
    ``strict=False``.  TerraMind/THOR tokens can only add a bounded residual to
    channel 3 (height); channels 0-2 are copied from the baseline unchanged.
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, height_specialist_depth=0, base_ch=32,
                 gate_init_bias=4.0, gate_mode="simple", gate_untied=False,
                 modality_dropout=0.0, height_gate_source="alpha",
                 height_hidden_ch=None, height_trunk_depth=2,
                 height_independent_branches=False, height_head_kind="linear",
                 height_n_bins=64, height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, token_source_ch=768,
                 correction_hidden_ch=96, correction_context_ch=32,
                 max_delta_norm=0.20):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TokenHeightCorrectionLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "TokenHeightCorrectionLightUNet expects AlphaEarth+Tessera pixel "
                f"input with >{alpha_channels} channels, got {pixel_channels}"
            )
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"token_channels={token_channels} must be divisible by "
                f"token_source_ch={token_source_ch}"
            )

        self.supports_aux_outputs = False
        self.alpha_channels = int(alpha_channels)
        self.gate_untied = bool(gate_untied)
        self.modality_dropout = float(modality_dropout)
        self.token_source_ch = int(token_source_ch)
        self.num_token_sources = token_channels // token_source_ch
        self.max_delta_norm = float(max_delta_norm)
        tessera_channels = pixel_channels - alpha_channels

        # Baseline path: keep names checkpoint-compatible with ae_tessera_unet_gated.
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
            base_ch, mode=gate_mode, untied=self.gate_untied, init_bias=gate_init_bias
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
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

        hidden_ch = int(correction_hidden_ch)
        context_ch = int(correction_context_ch)
        self.height_correction = nn.ModuleDict({
            "source_decoders": nn.ModuleList(
                nn.Sequential(
                    ConvGNAct(token_source_ch, hidden_ch, kernel_size=1, padding=0),
                    ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
                )
                for _ in range(self.num_token_sources)
            ),
            "context_proj": ConvGNAct(4, context_ch, kernel_size=1, padding=0),
            "router": nn.Sequential(
                ConvGNAct(hidden_ch * self.num_token_sources + context_ch,
                          hidden_ch, kernel_size=3),
                ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
                nn.Conv2d(hidden_ch, 2, 1),
            ),
        })
        final = self.height_correction["router"][-1]
        nn.init.zeros_(final.bias)
        final.bias.data[1] = -6.0  # gate suppressed at init: sigmoid(-6)≈0.002, delta≈0
        for module in self._baseline_modules():
            module.eval()

    def _baseline_modules(self):
        return (
            self.alpha_unet,
            self.tessera_entry,
            self.tessera_unet,
            self.gate_conv,
            self.head,
        )

    def train(self, mode=True):
        super().train(mode)
        # Frozen baseline modules contain BatchNorm; keep their running stats
        # fixed while the token correction branch learns.
        for module in self._baseline_modules():
            module.eval()
        return self

    def _baseline_forward(self, pixel):
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
        return self.head(fused, return_aux=False)

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("TokenHeightCorrectionLightUNet expects (pixel, token) input")
        pixel, token = x

        with torch.no_grad():
            base_out = self._baseline_forward(pixel)
        base_out = base_out.detach()

        token_parts = torch.split(token, self.token_source_ch, dim=1)
        token_feats = [
            decoder(part)
            for decoder, part in zip(self.height_correction["source_decoders"], token_parts)
        ]
        token_feat = torch.cat(token_feats, dim=1)

        context = F.adaptive_avg_pool2d(base_out, token_feat.shape[-2:])
        context = self.height_correction["context_proj"](context)
        delta_gate = self.height_correction["router"](
            torch.cat([token_feat, context], dim=1)
        )
        delta_raw = delta_gate[:, 0:1]
        gate = torch.sigmoid(delta_gate[:, 1:2])
        delta = self.max_delta_norm * torch.tanh(delta_raw) * gate
        delta = F.interpolate(
            delta,
            size=base_out.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        out = torch.cat([base_out[:, :3], base_out[:, 3:4] + delta], dim=1)
        if not return_aux:
            return out
        return {
            "out": out,
            "baseline_out": base_out,
            "height_delta": delta,
            "height_delta_gate": F.interpolate(
                gate,
                size=base_out.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        }


class TokenHeightErrorRouterLightUNet(TokenHeightCorrectionLightUNet):
    """Height correction with explicit baseline-error routing supervision.

    Compared with ``TokenHeightCorrectionLightUNet``, this variant predicts two
    error masks (building-height error, vegetation-height error) and two bounded
    residuals.  The final correction is routed by both the frozen baseline's
    class probabilities and the learned error probabilities:

        height = baseline_height
               + p_building * p_error_building * delta_building
               + p_vegetation * p_error_vegetation * delta_vegetation
    """

    def __init__(self, *args, correction_hidden_ch=96, correction_context_ch=32,
                 max_delta_norm=0.20, error_threshold_m=1.0,
                 error_aux_weight=0.25, delta_sparsity_weight=0.02,
                 token_source_ch=768, **kwargs):
        super().__init__(
            *args,
            correction_hidden_ch=correction_hidden_ch,
            correction_context_ch=correction_context_ch,
            max_delta_norm=max_delta_norm,
            token_source_ch=token_source_ch,
            **kwargs,
        )
        self.supports_aux_outputs = True
        self.error_threshold_norm = float(error_threshold_m) / HEIGHT_NORM_CONSTANT
        self.error_aux_weight = float(error_aux_weight)
        self.delta_sparsity_weight = float(delta_sparsity_weight)

        hidden_ch = int(correction_hidden_ch)
        context_ch = int(correction_context_ch)
        self.height_correction = nn.ModuleDict({
            "source_decoders": nn.ModuleList(
                nn.Sequential(
                    ConvGNAct(token_source_ch, hidden_ch, kernel_size=1, padding=0),
                    ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
                )
                for _ in range(self.num_token_sources)
            ),
            # baseline out (4ch) + class uncertainty p*(1-p) (3ch)
            "context_proj": ConvGNAct(7, context_ch, kernel_size=1, padding=0),
            "router": nn.Sequential(
                ConvGNAct(hidden_ch * self.num_token_sources + context_ch,
                          hidden_ch, kernel_size=3),
                ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
                nn.Conv2d(hidden_ch, 4, 1),
            ),
        })
        final = self.height_correction["router"][-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("TokenHeightErrorRouterLightUNet expects (pixel, token) input")
        pixel, token = x

        with torch.no_grad():
            base_out = self._baseline_forward(pixel)
        base_out = base_out.detach()

        token_parts = torch.split(token, self.token_source_ch, dim=1)
        token_feats = [
            decoder(part)
            for decoder, part in zip(self.height_correction["source_decoders"], token_parts)
        ]
        token_feat = torch.cat(token_feats, dim=1)

        p_uncertainty = base_out[:, :3] * (1.0 - base_out[:, :3])
        context = torch.cat([base_out, p_uncertainty], dim=1)
        context = F.adaptive_avg_pool2d(context, token_feat.shape[-2:])
        context = self.height_correction["context_proj"](context)
        routed = self.height_correction["router"](
            torch.cat([token_feat, context], dim=1)
        )
        delta_b_raw = routed[:, 0:1]
        delta_v_raw = routed[:, 1:2]
        err_b_logit = routed[:, 2:3]
        err_v_logit = routed[:, 3:4]

        gate_b = torch.sigmoid(err_b_logit)
        gate_v = torch.sigmoid(err_v_logit)
        delta_b = self.max_delta_norm * torch.tanh(delta_b_raw) * gate_b
        delta_v = self.max_delta_norm * torch.tanh(delta_v_raw) * gate_v

        out_size = base_out.shape[-2:]
        delta_b = F.interpolate(delta_b, size=out_size, mode="bilinear", align_corners=False)
        delta_v = F.interpolate(delta_v, size=out_size, mode="bilinear", align_corners=False)
        error_logits = F.interpolate(
            torch.cat([err_b_logit, err_v_logit], dim=1),
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )
        p_b = base_out[:, 0:1]
        p_v = base_out[:, 1:2]
        height_delta = p_b * delta_b + p_v * delta_v
        out = torch.cat([base_out[:, :3], base_out[:, 3:4] + height_delta], dim=1)

        if not return_aux:
            return out
        return {
            "out": out,
            "baseline_out": base_out,
            "height_delta": height_delta,
            "height_delta_building": delta_b,
            "height_delta_vegetation": delta_v,
            "height_error_logits": error_logits,
            "height_error_threshold_norm": self.error_threshold_norm,
            "height_error_aux_weight": self.error_aux_weight,
            "delta_sparsity_weight": self.delta_sparsity_weight,
        }


# ── xfusion_057: FiLM Fusion (no UNet in pixel branch) ──────────────────────

class FiLMFusionLightUNet(nn.Module):
    """xfusion_057: pixel branch via linear projection + patch branch via gated
    mixer + FiLM conditioning. No UNet in the pixel path.

    Pixel branch: AE → GroupNorm(1,64) + Conv1×1(48ch),
                  Tessera → ChannelCalibration + Conv1×1(80ch),
                  concat → F_p (128ch, 256×256).

    Patch branch: 4 token sources (TM_S1, TM_S2, THOR_S1, THOR_S2) each
                  → Conv1×1(96ch), per-spatial softmax gate mixer at 16×16
                  → bilinear upsample → F_c (96ch, 256×256).

    FiLM: F_c → γ,β (zero-initialized), F_fused = F_p×(1+γ)+β.
    Decoder: ConvGNAct(128→96) + ConvGNAct(96→64).
    Head: MultiTaskPredictionHead(in_ch=64) — unchanged.
    """

    _AE_CH = 64
    _TES_CH = 128
    _TOKEN_SRC_CH = 768
    _N_TOKEN_SOURCES = 4
    _AE_PROJ_CH = 48
    _TES_PROJ_CH = 80
    _PIXEL_CH = 128      # = _AE_PROJ_CH + _TES_PROJ_CH
    _CTX_CH = 96
    _DEC_CH = 64
    _TOKEN_INPUT_CLAMP = 50.0
    _CTX_CLAMP = 50.0
    _GATE_LOGIT_CLAMP = 20.0
    _FILM_PARAM_CLAMP = 4.0

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64,
                 height_specialist_depth=0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None,
                 modality_dropout=0.0, token_calibration=False, **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError("FiLMFusionLightUNet assumes 4 output channels")
        expected_pixel = self._AE_CH + self._TES_CH
        if pixel_channels != expected_pixel:
            raise ValueError(
                f"FiLMFusionLightUNet expects pixel_channels={expected_pixel}, "
                f"got {pixel_channels}"
            )
        expected_token = self._TOKEN_SRC_CH * self._N_TOKEN_SOURCES
        if token_channels != expected_token:
            raise ValueError(
                f"FiLMFusionLightUNet expects token_channels={expected_token} "
                f"(4×768), got {token_channels}"
            )

        self.supports_aux_outputs = True
        self.modality_dropout = float(modality_dropout)
        self.token_calibration = bool(token_calibration)

        # Pixel branch
        self.ae_norm = nn.GroupNorm(1, self._AE_CH, affine=True)
        self.ae_proj = nn.Conv2d(self._AE_CH, self._AE_PROJ_CH, 1, bias=False)
        self.tes_calib = ChannelCalibration(self._TES_CH)
        self.tes_proj = nn.Conv2d(self._TES_CH, self._TES_PROJ_CH, 1, bias=False)

        # Patch branch (order: TM_S1, TM_S2, THOR_S1, THOR_S2)
        self.token_projs = nn.ModuleList([
            nn.Conv2d(self._TOKEN_SRC_CH, self._CTX_CH, 1, bias=False)
            for _ in range(self._N_TOKEN_SOURCES)
        ])
        self.token_calibs = (
            nn.ModuleList(ChannelCalibration(self._TOKEN_SRC_CH)
                          for _ in range(self._N_TOKEN_SOURCES))
            if self.token_calibration else None
        )
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(self._N_TOKEN_SOURCES * self._CTX_CH, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, self._N_TOKEN_SOURCES, 1),
        )

        # FiLM: zero-initialized → identity at start of training
        self.film_conv = nn.Conv2d(self._CTX_CH, self._PIXEL_CH * 2, 1)
        nn.init.zeros_(self.film_conv.weight)
        nn.init.zeros_(self.film_conv.bias)

        # Shared decoder
        self.decoder = nn.Sequential(
            ConvGNAct(self._PIXEL_CH, 96, kernel_size=3),
            ConvGNAct(96, self._DEC_CH, kernel_size=3),
        )

        # Prediction head (unchanged internals)
        self.head = MultiTaskPredictionHead(
            in_ch=self._DEC_CH,
            out_channels=n_classes,
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

    def _pixel_branch(self, ae, tes):
        a = self.ae_proj(self.ae_norm(ae))       # (B, 48, 256, 256)
        t = self.tes_proj(self.tes_calib(tes))   # (B, 80, 256, 256)
        return torch.cat([a, t], dim=1)          # (B, 128, 256, 256)

    def _patch_branch(self, token):
        # THOR token rasters can have very large raw magnitudes. Keep this
        # branch in fp32 and bound activations before they reach softmax/FiLM.
        with torch.amp.autocast("cuda", enabled=False):
            parts = token.float().split(self._TOKEN_SRC_CH, dim=1)  # 4×(B,768,16,16)
            if self.token_calibs is not None:
                parts = [
                    calib(src) for calib, src in zip(self.token_calibs, parts)
                ]
            parts = [
                src.clamp(-self._TOKEN_INPUT_CLAMP, self._TOKEN_INPUT_CLAMP)
                for src in parts
            ]
            projs = [
                proj(src).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
                for proj, src in zip(self.token_projs, parts)
            ]                                                  # 4×(B,96,16,16)
            gate_logits = self.gate_mlp(torch.cat(projs, dim=1))
            gate_logits = gate_logits.clamp(
                -self._GATE_LOGIT_CLAMP, self._GATE_LOGIT_CLAMP
            )
            gates = F.softmax(gate_logits, dim=1).unsqueeze(2)  # (B,4,1,16,16)
            stack = torch.stack(projs, dim=1)                   # (B,4,96,16,16)
            F_c_lr = (stack * gates).sum(dim=1)                 # (B,96,16,16)
            F_c = F.interpolate(
                F_c_lr, scale_factor=16, mode="bilinear", align_corners=False
            )
            return F_c.clamp(-self._CTX_CLAMP, self._CTX_CLAMP)  # (B,96,256,256)

    def _film_fuse(self, F_p, F_c):
        with torch.amp.autocast("cuda", enabled=False):
            gamma, beta = self.film_conv(F_c.float()).chunk(2, dim=1)
            gamma = gamma.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
            beta = beta.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
            fused = F_p.float() * (1.0 + gamma) + beta
            return fused.clamp(-self._CTX_CLAMP, self._CTX_CLAMP)

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("FiLMFusionLightUNet expects (pixel, token) input")
        pixel, token = x
        ae = pixel[:, :self._AE_CH]
        tes = pixel[:, self._AE_CH:]

        if self.training and self.modality_dropout > 0.0:
            keep = (
                torch.rand(token.size(0), 1, 1, 1, device=token.device)
                >= self.modality_dropout
            ).float()
            token = token * keep / max(1e-6, 1.0 - self.modality_dropout)

        F_p = self._pixel_branch(ae, tes)    # (B, 128, 256, 256)
        F_c = self._patch_branch(token)       # (B,  96, 256, 256)
        F_fused = self._film_fuse(F_p, F_c)  # (B, 128, 256, 256)
        feat = self.decoder(F_fused)          # (B,  64, 256, 256)
        return self.head(feat, return_aux=return_aux)
