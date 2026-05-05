import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, _group_count
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

    def forward(self, query, key_value):
        out, _ = self.attn(
            self.q_norm(query),
            self.kv_norm(key_value),
            self.kv_norm(key_value),
            need_weights=False,
        )
        return query + out


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
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + h
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
        gates = torch.sigmoid(self.gate_net(torch.cat([fused, mix], dim=1)))
        g_main = gates[:, 0:1]
        g_mix = gates[:, 1:2]
        return g_main * fused + g_mix * mix


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
                 use_token_extractor=False, token_extractor_mid_ch=None):
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
