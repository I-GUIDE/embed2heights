import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct
from .backbones import LightUNet
from .heads import MultiTaskPredictionHead
from .pixel_fusion import TesseraCompressionStem, TesseraIoUFusionGatedLightUNet


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


class PerPixelGatedTokenScaleResidual(nn.Module):
    """Per-pixel sigmoid-gated token residual. Model learns WHERE to borrow
    from the token modality based on the AE+Tessera target features.

    Init: gate ~ sigmoid(-4) ~ 0.018 so output starts ~= target (safe baseline).
    The gate is a 1x1 conv on the target features, output 1 channel (broadcast)."""

    def __init__(self, token_ch, target_ch, hidden_ch=None, init_bias=-4.0):
        super().__init__()
        hidden_ch = hidden_ch or min(max(target_ch, 64), 256)
        self.net = nn.Sequential(
            ConvGNAct(token_ch, hidden_ch, kernel_size=1, padding=0),
            ConvGNAct(hidden_ch, target_ch, kernel_size=3),
        )
        self.gate_conv = nn.Conv2d(target_ch, 1, kernel_size=1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, init_bias)

    def forward(self, target, token_feat):
        if token_feat.shape[-2:] != target.shape[-2:]:
            token_feat = F.interpolate(
                token_feat,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        token_proj = self.net(token_feat)
        gate = torch.sigmoid(self.gate_conv(target))
        return target + gate * token_proj


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
                 presence_branch_ch=48, per_pixel_gate=False, **unused):
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

        gate_cls = PerPixelGatedTokenScaleResidual if per_pixel_gate else GatedTokenScaleResidual
        self.bottleneck_adapter = (
            gate_cls(self._token_input_channels[32], c4)
            if "bottleneck" in self.fusion_points else None
        )
        self.decoder64_adapter = (
            gate_cls(self._token_input_channels[64], c3)
            if "decoder64" in self.fusion_points else None
        )
        self.decoder128_adapter = (
            gate_cls(self._token_input_channels[128], c2)
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


class HierarchicalGatedFusion(nn.Module):
    """User's hierarchical gated fusion idea.

    Stage 1: AE + Tessera via existing TesseraIoUFusionGatedLightUNet
      → produces (B, 4, H, W) main output (and aux dict if requested).
    Stage 2: token features (16x16, e.g. TerraMind-S2 768ch) → 1x1 channel reduce →
      ConvGNAct → bilinear upsample to (H, W) → conv → (B, 4, H, W) Stage 2 output.
    Final gate: per-pixel sigmoid gate over [stage1_out, stage2_out],
      INITIALIZED so the gate ~= 1 (Stage 1 dominant) at init. This guarantees
      we never DESTROY our known-good Stage 1; tokens can only add small corrections
      where useful.

    Aux outputs are passed through from Stage 1 (since Stage 2 is just a logit
    correction, not a deep-supervised branch).
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 token_hidden_ch=64, **stage1_kwargs):
        super().__init__()
        if n_classes != 4:
            raise ValueError("HierarchicalGatedFusion assumes 4 output channels")
        self.supports_aux_outputs = True
        # Stage 1: the proven gated AE+Tessera arch
        self.stage1 = TesseraIoUFusionGatedLightUNet(
            n_channels=pixel_channels, n_classes=n_classes, **stage1_kwargs
        )
        # Stage 2: light token branch
        # Channel reduce 768 → 64, then upsample, then 4-channel logits
        self.token_proj = ConvGNAct(token_channels, token_hidden_ch, kernel_size=1)
        self.token_conv = ConvGNAct(token_hidden_ch, token_hidden_ch, kernel_size=3)
        self.token_head = nn.Conv2d(token_hidden_ch, n_classes, kernel_size=1)
        # Final gate: 2*n_classes inputs → n_classes gate values; bias=4.0 means
        # sigmoid(4)≈0.98 at init, so Stage1 dominates initially.
        self.final_gate = nn.Conv2d(2 * n_classes, n_classes, kernel_size=1)
        nn.init.zeros_(self.final_gate.weight)
        nn.init.constant_(self.final_gate.bias, 4.0)

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("HierarchicalGatedFusion expects (pixel, token) input")
        pixel, token = x
        # Stage 1: AE+Tessera through gated UNet
        if return_aux:
            stage1_out, aux = self.stage1(pixel, return_aux=True)
        else:
            stage1_out = self.stage1(pixel, return_aux=False)
            aux = None

        # Stage 2: token branch (16x16 → up to pixel HxW)
        t = self.token_proj(token)
        t = self.token_conv(t)
        t = F.interpolate(t, size=(pixel.shape[2], pixel.shape[3]),
                          mode="bilinear", align_corners=False)
        stage2_out = self.token_head(t)

        # Per-pixel sigmoid gate: g≈1 at init → stage1 dominant.
        gate_in = torch.cat([stage1_out, stage2_out], dim=1)
        gate = torch.sigmoid(self.final_gate(gate_in))
        fused = gate * stage1_out + (1.0 - gate) * stage2_out

        if return_aux:
            return fused, aux
        return fused
