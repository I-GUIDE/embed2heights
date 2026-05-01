import math

import torch
import torch.nn as nn
import torch.nn.functional as F


HEIGHT_NORM_CONSTANT = 30.0  # mirrors core.dataset; meters / NORM = normalized height.


# ==========================================
# 1. LIGHT UNET COMPONENTS
# ==========================================

def _light_norm(num_channels, kind="bn"):
    # Pretrain→finetune transfer prefers stateless norm; "gn" replaces BN with
    # GroupNorm (8 groups, capped at the channel count). Default stays "bn".
    kind = (kind or "bn").lower()
    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    if kind == "gn":
        groups = min(8, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unknown norm_kind={kind!r}; expected 'bn' or 'gn'.")


class DoubleConv(nn.Module):
    """(convolution => [Norm] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, norm_kind="bn"):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _light_norm(out_channels, norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _light_norm(out_channels, norm_kind),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleBlock(nn.Module):
    """
    Bilinear Upsampling + Convolution.
    Smoother than PixelShuffle/TransposeConv, avoids checkerboard artifacts.
    """

    def __init__(self, in_channels, out_channels, norm_kind="bn"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = _light_norm(out_channels, norm_kind)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_ch=32, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None):
        super(LightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.supports_aux_outputs = True

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.inc = DoubleConv(n_channels, c1, norm_kind=norm_kind)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, norm_kind=norm_kind))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, norm_kind=norm_kind))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4, norm_kind=norm_kind))

        self.up1 = UpsampleBlock(c4, c3, norm_kind=norm_kind)
        self.conv1 = DoubleConv(c4, c3, norm_kind=norm_kind)

        self.up2 = UpsampleBlock(c3, c2, norm_kind=norm_kind)
        self.conv2 = DoubleConv(c3, c2, norm_kind=norm_kind)

        self.up3 = UpsampleBlock(c2, c1, norm_kind=norm_kind)
        self.conv3 = DoubleConv(c2, c1, norm_kind=norm_kind)

        self.head = MultiTaskPredictionHead(
            in_ch=c1,
            out_channels=n_classes,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
        )

    def forward_features(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        return x

    def forward(self, x, return_aux=False):
        x = self.forward_features(x)
        return self.head(x, return_aux=return_aux)


class LightUNetPlusPlus(nn.Module):
    """UNet++ (nested U-Net) variant of LightUNet.

    Same external contract as ``LightUNet``: ``forward_features(x)`` returns a
    ``base_ch``-wide full-resolution feature map, so this class can be dropped
    into ``TesseraIoUFusion`` in place of ``LightUNet``. The internal decoder
    is replaced by the dense nested skip-connection grid from
    Zhou et al. 2018 ("UNet++: A Nested U-Net Architecture..."), which fuses
    encoder features with progressively refined intermediate decoder nodes
    instead of a single skip per scale. Deep supervision is intentionally
    omitted — only the deepest decoder node ``X(0,3)`` is exposed, since the
    downstream MultiTaskPredictionHead already supplies the supervision.

    Decoder grid (depth=4 encoder, 3 decoder columns):
        X(0,0) <- X(0,1) <- X(0,2) <- X(0,3)
        X(1,0) <- X(1,1) <- X(1,2)
        X(2,0) <- X(2,1)
        X(3,0)
    """

    def __init__(self, n_channels, n_classes, base_ch=32, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.supports_aux_outputs = True

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Encoder column (j=0)
        self.x00 = DoubleConv(n_channels, c1, norm_kind=norm_kind)
        self.pool1 = nn.MaxPool2d(2)
        self.x10 = DoubleConv(c1, c2, norm_kind=norm_kind)
        self.pool2 = nn.MaxPool2d(2)
        self.x20 = DoubleConv(c2, c3, norm_kind=norm_kind)
        self.pool3 = nn.MaxPool2d(2)
        self.x30 = DoubleConv(c3, c4, norm_kind=norm_kind)

        # Decoder column j=1: each node fuses its encoder peer with the
        # upsampled encoder one level deeper.
        self.up_10_01 = UpsampleBlock(c2, c1, norm_kind=norm_kind)
        self.x01 = DoubleConv(c1 + c1, c1, norm_kind=norm_kind)
        self.up_20_11 = UpsampleBlock(c3, c2, norm_kind=norm_kind)
        self.x11 = DoubleConv(c2 + c2, c2, norm_kind=norm_kind)
        self.up_30_21 = UpsampleBlock(c4, c3, norm_kind=norm_kind)
        self.x21 = DoubleConv(c3 + c3, c3, norm_kind=norm_kind)

        # Decoder column j=2: dense skips from all earlier same-row nodes.
        self.up_11_02 = UpsampleBlock(c2, c1, norm_kind=norm_kind)
        self.x02 = DoubleConv(c1 * 2 + c1, c1, norm_kind=norm_kind)
        self.up_21_12 = UpsampleBlock(c3, c2, norm_kind=norm_kind)
        self.x12 = DoubleConv(c2 * 2 + c2, c2, norm_kind=norm_kind)

        # Decoder column j=3: final fused node, output shape matches LightUNet.
        self.up_12_03 = UpsampleBlock(c2, c1, norm_kind=norm_kind)
        self.x03 = DoubleConv(c1 * 3 + c1, c1, norm_kind=norm_kind)

        self.head = MultiTaskPredictionHead(
            in_ch=c1,
            out_channels=n_classes,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
        )

    def forward_features(self, x):
        x00 = self.x00(x)
        x10 = self.x10(self.pool1(x00))
        x20 = self.x20(self.pool2(x10))
        x30 = self.x30(self.pool3(x20))

        x01 = self.x01(torch.cat([x00, self.up_10_01(x10)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up_20_11(x20)], dim=1))
        x21 = self.x21(torch.cat([x20, self.up_30_21(x30)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, self.up_11_02(x11)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self.up_21_12(x21)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, self.up_12_03(x12)], dim=1))
        return x03

    def forward(self, x, return_aux=False):
        x = self.forward_features(x)
        return self.head(x, return_aux=return_aux)


# ==========================================
# 2. DECODER FOR VIT-TOKEN EMBEDDINGS (16x16 -> 256x256)
# ==========================================

class StandardUpsampleBlock(nn.Module):
    """
    Uses standard dense convolutions.
    Blazingly fast on Apple Silicon MPS, unlike grouped/depthwise convs.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Standard 3x3 convolution (groups=1) which the M2 GPU loves
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class EfficientDecoder256Fast(nn.Module):
    """
    High-speed, memory-safe decoder for 16x16 -> 256x256 upsampling on M2 Max.
    """

    def __init__(self, in_channels=768, out_channels=4):
        super().__init__()

        # THE SQUEEZE: 768 -> 256 at 16x16 resolution. (Prevents memory blowup)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        # PROGRESSIVE UPSAMPLING: Halving channels as resolution doubles.
        self.up1 = StandardUpsampleBlock(256, 128)  # 16x16   -> 32x32
        self.up2 = StandardUpsampleBlock(128, 64)  # 32x32   -> 64x64
        self.up3 = StandardUpsampleBlock(64, 32)  # 64x64   -> 128x128
        self.up4 = StandardUpsampleBlock(32, 16)  # 128x128 -> 256x256

        # PREDICTION HEAD
        self.head = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)


# ==========================================
# 2b. NECK + U-NET DECODER FOR VIT TOKENS
# ==========================================
#
# Rationale: last-layer ViT tokens at 16x16 have no native multi-scale
# features, so a pure upsampling decoder (EfficientDecoder256Fast) has no
# skip connections to recover fine spatial detail. The neck synthesises a
# pseudo pyramid by projecting the single source feature into independent
# per-scale representations (each scale learns its own 1x1 projection +
# refinement conv), which the U-Net-style decoder then consumes as skips.
# Inspired by the "neck" reported in the IBM ESA TerraMind challenge writeup.


class TokenPyramidNeck(nn.Module):
    """Produce a 4-level pseudo pyramid from a single 16x16 token grid.

    Each level first 1x1-projects the source at the cheap 16x16 resolution,
    then bilinear-upsamples, then applies a 3x3 conv to specialise per scale.
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


class TokenNeckDecoder(nn.Module):
    """U-Net-style decoder for 16x16 ViT tokens with a pseudo multi-scale neck.

    Flow (input: B x 768 x 16 x 16):
        neck -> {16:256, 32:128, 64:64, 128:32}
        stage 16->32:  upsample(256->128) cat neck[32] -> conv -> 128
        stage 32->64:  upsample(128->64)  cat neck[64] -> conv -> 64
        stage 64->128: upsample(64->32)   cat neck[128]-> conv -> 32
        stage 128->256: upsample(32->32) -> conv -> 32
        head: MultiTaskPredictionHead(32) -> 4ch submission tensor.
    """

    def __init__(self, n_channels=768, n_classes=4,
                 level_channels=(256, 128, 64, 32), drop=0.05):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TokenNeckDecoder assumes 4 output channels")
        self.supports_aux_outputs = True
        c16, c32, c64, c128 = level_channels

        self.neck = TokenPyramidNeck(n_channels, level_channels)

        # Each up block: bilinear x2 + 3x3 conv to halve channels smoothly.
        self.up_16_32 = UpsampleBlock(c16, c32)
        self.fuse_32 = nn.Sequential(
            ConvGNAct(c32 + c32, c32, kernel_size=3),
            ConvGNAct(c32, c32, kernel_size=3),
        )

        self.up_32_64 = UpsampleBlock(c32, c64)
        self.fuse_64 = nn.Sequential(
            ConvGNAct(c64 + c64, c64, kernel_size=3),
            ConvGNAct(c64, c64, kernel_size=3),
        )

        self.up_64_128 = UpsampleBlock(c64, c128)
        self.fuse_128 = nn.Sequential(
            ConvGNAct(c128 + c128, c128, kernel_size=3),
            ConvGNAct(c128, c128, kernel_size=3),
        )

        self.up_128_256 = UpsampleBlock(c128, c128)
        self.fuse_256 = ConvGNAct(c128, c128, kernel_size=3)

        self.head = MultiTaskPredictionHead(c128, out_channels=n_classes, drop=drop)

    def forward_features(self, x):
        pyr = self.neck(x)

        x = self.up_16_32(pyr[16])
        x = self.fuse_32(torch.cat([x, pyr[32]], dim=1))

        x = self.up_32_64(x)
        x = self.fuse_64(torch.cat([x, pyr[64]], dim=1))

        x = self.up_64_128(x)
        x = self.fuse_128(torch.cat([x, pyr[128]], dim=1))

        x = self.up_128_256(x)
        x = self.fuse_256(x)
        return x

    def forward(self, x, return_aux=False):
        x = self.forward_features(x)
        return self.head(x, return_aux=return_aux)


class TokenNeckNormDecoder(nn.Module):
    """Token neck with input standardization for high-dynamic-range tokens."""

    def __init__(self, n_channels=768, n_classes=4,
                 level_channels=(256, 128, 64, 32), drop=0.05):
        super().__init__()
        self.supports_aux_outputs = True
        self.input_norm = TokenChannelStandardize(n_channels)
        self.token_neck = TokenNeckDecoder(
            n_channels=n_channels,
            n_classes=n_classes,
            level_channels=level_channels,
            drop=drop,
        )

    def forward_features(self, x):
        return self.token_neck.forward_features(self.input_norm(x))

    def forward(self, x, return_aux=False):
        return self.token_neck(self.input_norm(x), return_aux=return_aux)


class TokenChannelStandardize(nn.Module):
    """Per-sample, per-channel spatial standardization for token grids."""

    def __init__(self, channels, eps=1e-5, clamp=8.0):
        super().__init__()
        self.eps = eps
        self.clamp = clamp
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(-2, -1), keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        if self.clamp is not None:
            x = x.clamp(-self.clamp, self.clamp)
        return x * self.weight + self.bias


class SameModelTokenFusion(nn.Module):
    """Fuse same-backbone S1/S2 token grids before the token neck.

    The input is expected to be channel-concatenated as [S1, S2]. S2 is used as
    a conservative anchor and the learned fused residual is gated in from zero.
    """

    def __init__(self, in_channels, adapter_channels=None, fused_channels=None,
                 modality_dropout=0.15, normalize_inputs=False, initial_gate=0.0):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError("SameModelTokenFusion expects concatenated S1/S2 channels")
        self.source_channels = in_channels // 2
        self.fused_channels = fused_channels or self.source_channels
        self.adapter_channels = adapter_channels or min(384, self.source_channels)
        self.modality_dropout = min(max(float(modality_dropout), 0.0), 0.49)
        self.input_norm = TokenChannelStandardize(in_channels) if normalize_inputs else nn.Identity()

        self.s1_adapter = ConvGNAct(
            self.source_channels, self.adapter_channels, kernel_size=1, padding=0
        )
        self.s2_adapter = ConvGNAct(
            self.source_channels, self.adapter_channels, kernel_size=1, padding=0
        )
        self.anchor = (
            nn.Identity()
            if self.source_channels == self.fused_channels
            else ConvGNAct(self.source_channels, self.fused_channels, kernel_size=1, padding=0)
        )
        self.fuse = nn.Sequential(
            ConvGNAct(self.adapter_channels * 2, self.fused_channels, kernel_size=1, padding=0),
            ConvGNAct(self.fused_channels, self.fused_channels, kernel_size=3),
        )
        self.gate = nn.Parameter(torch.full((1,), float(initial_gate)))

    def _apply_modality_dropout(self, s1, s2):
        if not self.training or self.modality_dropout <= 0:
            return s1, s2
        batch = s1.shape[0]
        choice = torch.rand(batch, 1, 1, 1, device=s1.device)
        drop_s1 = choice < self.modality_dropout
        drop_s2 = (choice >= self.modality_dropout) & (choice < 2 * self.modality_dropout)
        return s1.masked_fill(drop_s1, 0.0), s2.masked_fill(drop_s2, 0.0)

    def forward(self, x):
        x = self.input_norm(x)
        s1, s2 = torch.split(x, self.source_channels, dim=1)
        s1, s2 = self._apply_modality_dropout(s1, s2)
        s1 = self.s1_adapter(s1)
        s2_adapted = self.s2_adapter(s2)
        delta = self.fuse(torch.cat([s1, s2_adapted], dim=1))
        return self.anchor(s2) + self.gate * delta


class CrossModalTokenFusion(nn.Module):
    """Lightweight S1/S2 token interaction before the token neck.

    This keeps the conservative S2 residual path used by ``SameModelTokenFusion``
    but adds an explicit 16x16 token mixer: S2 queries attend to S1 keys/values
    and the result is projected back as a gated residual. That is closer to the
    early token-level multimodal interaction used by TerraMind while staying
    small enough for the downstream data regime here.
    """

    def __init__(self, in_channels, adapter_channels=None, fused_channels=None,
                 num_heads=4, modality_dropout=0.15, normalize_inputs=False,
                 initial_gate=0.0):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError("CrossModalTokenFusion expects concatenated S1/S2 channels")
        self.source_channels = in_channels // 2
        self.fused_channels = fused_channels or self.source_channels
        self.adapter_channels = adapter_channels or min(384, self.source_channels)
        if self.adapter_channels % num_heads != 0:
            for candidate in (8, 4, 2, 1):
                if self.adapter_channels % candidate == 0:
                    num_heads = candidate
                    break
        self.modality_dropout = min(max(float(modality_dropout), 0.0), 0.49)
        self.input_norm = TokenChannelStandardize(in_channels) if normalize_inputs else nn.Identity()

        self.s1_adapter = ConvGNAct(
            self.source_channels, self.adapter_channels, kernel_size=1, padding=0
        )
        self.s2_adapter = ConvGNAct(
            self.source_channels, self.adapter_channels, kernel_size=1, padding=0
        )
        self.s1_type = nn.Parameter(torch.zeros(1, self.adapter_channels, 1, 1))
        self.s2_type = nn.Parameter(torch.zeros(1, self.adapter_channels, 1, 1))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.adapter_channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.05,
        )
        self.attn_norm = nn.LayerNorm(self.adapter_channels)
        self.mix = nn.Sequential(
            ConvGNAct(self.adapter_channels * 3, self.fused_channels, kernel_size=1, padding=0),
            ConvGNAct(self.fused_channels, self.fused_channels, kernel_size=3),
        )
        self.anchor = (
            nn.Identity()
            if self.source_channels == self.fused_channels
            else ConvGNAct(self.source_channels, self.fused_channels, kernel_size=1, padding=0)
        )
        self.gate = nn.Parameter(torch.full((1,), float(initial_gate)))

    def _apply_modality_dropout(self, s1, s2):
        if not self.training or self.modality_dropout <= 0:
            return s1, s2
        batch = s1.shape[0]
        choice = torch.rand(batch, 1, 1, 1, device=s1.device)
        drop_s1 = choice < self.modality_dropout
        drop_s2 = (choice >= self.modality_dropout) & (choice < 2 * self.modality_dropout)
        return s1.masked_fill(drop_s1, 0.0), s2.masked_fill(drop_s2, 0.0)

    @staticmethod
    def _to_tokens(x):
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _to_grid(x, height, width):
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)

    def forward(self, x):
        x = self.input_norm(x)
        s1_raw, s2_raw = torch.split(x, self.source_channels, dim=1)
        s1_raw, s2_raw = self._apply_modality_dropout(s1_raw, s2_raw)
        s1 = self.s1_adapter(s1_raw) + self.s1_type
        s2 = self.s2_adapter(s2_raw) + self.s2_type

        b, c, h, w = s2.shape
        query = self._to_tokens(s2)
        key_value = self._to_tokens(s1)
        attended, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
        attended = self.attn_norm(query + attended)
        attended = self._to_grid(attended, h, w)

        delta = self.mix(torch.cat([s1, s2, attended], dim=1))
        return self.anchor(s2_raw) + self.gate * delta


class TokenFusionNeckDecoder(nn.Module):
    """S1/S2 same-model token fusion followed by TokenNeckDecoder."""

    def __init__(self, n_channels=1536, n_classes=4, level_channels=(256, 128, 64, 32),
                 drop=0.05, modality_dropout=0.15, normalize_inputs=False,
                 initial_gate=0.0):
        super().__init__()
        if n_channels % 2 != 0:
            raise ValueError("TokenFusionNeckDecoder expects S1/S2 channel concatenation")
        self.supports_aux_outputs = True
        fused_channels = n_channels // 2
        self.fusion = SameModelTokenFusion(
            n_channels,
            fused_channels=fused_channels,
            modality_dropout=modality_dropout,
            normalize_inputs=normalize_inputs,
            initial_gate=initial_gate,
        )
        self.token_neck = TokenNeckDecoder(
            n_channels=fused_channels,
            n_classes=n_classes,
            level_channels=level_channels,
            drop=drop,
        )

    def forward_features(self, x):
        return self.token_neck.forward_features(self.fusion(x))

    def forward(self, x, return_aux=False):
        return self.token_neck(self.fusion(x), return_aux=return_aux)


class TokenFusionNeckNormDecoder(TokenFusionNeckDecoder):
    """Normalized S1/S2 token fusion for sources with large scale mismatch."""

    def __init__(self, n_channels=1536, n_classes=4, level_channels=(256, 128, 64, 32),
                 drop=0.05, modality_dropout=0.15):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            level_channels=level_channels,
            drop=drop,
            modality_dropout=modality_dropout,
            normalize_inputs=True,
            initial_gate=0.1,
        )


class TokenFusionNeckCrossAttentionDecoder(nn.Module):
    """Cross-attention S1/S2 token fusion followed by TokenNeckDecoder."""

    def __init__(self, n_channels=1536, n_classes=4, level_channels=(256, 128, 64, 32),
                 drop=0.05, modality_dropout=0.15, normalize_inputs=False,
                 initial_gate=0.0):
        super().__init__()
        if n_channels % 2 != 0:
            raise ValueError("TokenFusionNeckCrossAttentionDecoder expects S1/S2 channel concatenation")
        self.supports_aux_outputs = True
        fused_channels = n_channels // 2
        self.fusion = CrossModalTokenFusion(
            n_channels,
            fused_channels=fused_channels,
            modality_dropout=modality_dropout,
            normalize_inputs=normalize_inputs,
            initial_gate=initial_gate,
        )
        self.token_neck = TokenNeckDecoder(
            n_channels=fused_channels,
            n_classes=n_classes,
            level_channels=level_channels,
            drop=drop,
        )

    def forward_features(self, x):
        return self.token_neck.forward_features(self.fusion(x))

    def forward(self, x, return_aux=False):
        return self.token_neck(self.fusion(x), return_aux=return_aux)


class TokenFusionNeckCrossAttentionNormDecoder(TokenFusionNeckCrossAttentionDecoder):
    """Normalized cross-attention fusion for high-dynamic-range token sources."""

    def __init__(self, n_channels=1536, n_classes=4, level_channels=(256, 128, 64, 32),
                 drop=0.05, modality_dropout=0.15):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            level_channels=level_channels,
            drop=drop,
            modality_dropout=modality_dropout,
            normalize_inputs=True,
            initial_gate=0.1,
        )


# ==========================================
# 3. EMBEDDING REFINER (encoder-light, decoder-heavy)
# ==========================================
#
# Design philosophy for pixel-aligned GFM embeddings (e.g. AlphaEarth 64ch @
# 256x256): the input is already a dense, semantically-rich feature map
# produced by a large pretrained foundation model. Rebuilding a deep encoder
# on top of it is wasteful (recomputes what AlphaEarth already encoded) and
# harmful (repeated downsampling throws away the pixel-level detail that this
# task — sub-pixel land cover + nDSM — needs most).
#
# Instead:
#   - Keep full 256x256 resolution end-to-end (no stride, no maxpool).
#   - Calibrate input channels (learnable per-channel affine / LayerScale).
#   - Refine with a stack of full-resolution ConvNeXt-style blocks.
#   - Use ASPP to expand receptive field without sacrificing resolution.
#   - Decouple segmentation and height heads; condition height on seg logits
#     (RMSE is scored only on pixels where the class is present, so height
#     must be aware of where each class lives).

class ChannelCalibration(nn.Module):
    """Learnable per-channel affine on the raw embedding.

    AlphaEarth channels have arbitrary scales/means that reflect the GFM's
    internal representation, not this task. A cheap affine lets the network
    renormalize them for downstream use without modifying the embedding.
    """

    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.scale + self.shift


class ConvNeXtBlock(nn.Module):
    """Full-resolution refinement block (ConvNeXt-v1 style).

    7x7 depthwise conv for local spatial context, inverted bottleneck MLP for
    channel mixing, LayerScale + residual. Cheap enough to stack many at
    256x256 without blowing up memory.
    """

    def __init__(self, dim, drop=0.0, ls_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)  # LayerNorm over channels
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(ls_init * torch.ones(1, dim, 1, 1))
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        r = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        return r + self.drop(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    Expands receptive field via parallel dilated convs + global image pooling.
    Crucially, all branches keep the input resolution — we get multi-scale
    context without a single downsample.
    """

    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18), dropout=0.1):
        super().__init__()
        branches = []
        for r in rates:
            if r == 1:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.GroupNorm(1, out_ch),
                    nn.GELU(),
                ))
            else:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                    nn.GroupNorm(1, out_ch),
                    nn.GELU(),
                ))
        self.branches = nn.ModuleList(branches)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[-2:], mode='bilinear', align_corners=False)
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))


def _group_count(channels):
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


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
                 presence_head_depth=1, presence_branch_ch=None):
        super().__init__()
        if out_channels != 4:
            raise ValueError("MultiTaskPredictionHead assumes 4 output channels")
        if height_gate_source not in {"alpha", "fused"}:
            raise ValueError("height_gate_source must be one of: alpha, fused")
        if height_head_kind not in {"linear", "softbin"}:
            raise ValueError("height_head_kind must be one of: linear, softbin")
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

        # Optional water-only bypass: run the same shared trunk weights on a
        # parallel feature that did not see the TerraMind cross-level adapter,
        # then route only the water presence branch through it. Building, tree,
        # height, fraction, FiLM, and Tessera-residual paths stay on `x`.
        bypass_h = None
        if water_bypass_x is not None and self.presence_head_kind == "split_all":
            bypass_h = self.shared(water_bypass_x)
            bypass_h = self.shared_act(bypass_h + self.shared_res(bypass_h))

        # Auxiliary soft fraction (for height gating + regression losses)
        fraction_logits = self.fraction_head(x)
        fractions = torch.sigmoid(fraction_logits)

        # Main presence classifier (submission channels 0-2). When an external
        # edge feature is provided, it learns only a zero-initialized residual
        # logit correction on top of the alpha-only logits.
        if bypass_h is not None:
            alpha_presence_logits = torch.cat([
                self.presence_head["building"](x),
                self.presence_head["tree"](x),
                self.presence_head["water"](bypass_h),
            ], dim=1)
        else:
            alpha_presence_logits = self._forward_presence_head(x)
        if presence_extra is not None:
            if self.presence_delta_head is None:
                raise ValueError("presence_extra was provided but this head has no residual branch")
            presence_delta_logits = self._forward_presence_delta(presence_extra)
            presence_logits = alpha_presence_logits + presence_delta_logits
        else:
            presence_delta_logits = None
            presence_logits = alpha_presence_logits
        presence_prob = torch.sigmoid(presence_logits)

        # FiLM conditioning uses soft fractions (fine-grained coverage signal)
        scale = self.film_scale(fractions)
        shift = self.film_shift(fractions)
        h = x * (1.0 + scale) + shift

        if self.height_independent_branches:
            h_base = self.height_base_trunk(h)
            h_building = self.height_building_trunk(h)
            h_vegetation = self.height_vegetation_trunk(h)
        else:
            h_shared = self.height_trunk(h)
            h_base = h_shared
            h_building = h_shared
            h_vegetation = h_shared

        base_logits = self.height_base_proj(h_base)
        building_logits = self.height_building_delta_proj(h_building)
        vegetation_logits = self.height_vegetation_delta_proj(h_vegetation)

        if self.height_head_kind == "softbin":
            # Softmax over K log-spaced bin centers, take expectation. Centers
            # are non-negative so each output height is non-negative without
            # softplus. Absolute (not delta) heights per class — the bin-CE
            # aux loss in losses.py forces commitment to the correct bin
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
        # `height_vegetation` are L1-trained on their class mask (losses.py),
        # so each is reliable ONLY on that class's pixels. We therefore route
        # each pixel to its relevant specialist by presence, and fall back to
        # `base_height` on background pixels.
        height_gate_logits = (
            presence_logits
            if self.height_gate_source == "fused"
            else alpha_presence_logits
        )
        height_presence_prob = torch.sigmoid(height_gate_logits)
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


def _build_fusion_gate(channels, mode="simple", untied=False, init_bias=4.0):
    """Build a gate module that sees concat(alpha, tessera) features.

    Args:
        channels: width of each modality stream (gate input is 2*channels).
        mode: "simple" (1x1 conv) or "rich" (Conv1x1 -> GN -> GELU -> Conv1x1).
        untied: if False, output a single tied gate G in (0,1), used as
                fused = G*AE + (1-G)*TES. If True, output independent gates
                G_AE and G_TES so fused = G_AE*AE + G_TES*TES (GMU 2(a)).
        init_bias: warm-start bias so at t=0 fused = AE features. Tied: a
                single +init_bias. Untied: +init_bias on AE half, -init_bias
                on TES half.
    """
    n_out = 2 * channels if untied else channels
    if mode == "simple":
        gate = nn.Conv2d(2 * channels, n_out, kernel_size=1)
        final = gate
    elif mode == "rich":
        hidden = max(channels, 32)
        gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, n_out, 1),
        )
        final = gate[-1]
    else:
        raise ValueError(f"Unknown gate mode: {mode!r}")

    nn.init.zeros_(final.weight)
    if untied:
        bias = torch.empty(n_out)
        bias[:channels].fill_(init_bias)
        bias[channels:].fill_(-init_bias)
        final.bias.data.copy_(bias)
    else:
        nn.init.constant_(final.bias, init_bias)
    return gate


def _apply_fusion_gate(gate_module, ae_feat, tes_feat, untied):
    raw = gate_module(torch.cat([ae_feat, tes_feat], dim=1))
    if untied:
        C = ae_feat.size(1)
        g_ae = torch.sigmoid(raw[:, :C])
        g_tes = torch.sigmoid(raw[:, C:])
        return g_ae * ae_feat + g_tes * tes_feat
    g = torch.sigmoid(raw)
    return g * ae_feat + (1.0 - g) * tes_feat


def _maybe_drop_modality(tes_feat, p, training):
    """Per-sample inverted dropout on the Tessera feature stream (training only)."""
    if not training or p <= 0.0:
        return tes_feat
    B = tes_feat.size(0)
    keep = (torch.rand(B, 1, 1, 1, device=tes_feat.device) >= p).float()
    return tes_feat * keep / max(1e-6, 1.0 - p)


class TesseraCompressionStem(nn.Module):
    """Strongly compress Tessera before it can influence IoU logits only.

    ``out_ch`` is the interface width seen by the presence residual head.
    ``hidden_ch`` and ``hidden_depth`` control extraction capacity without
    widening that interface.
    """

    def __init__(self, in_ch, out_ch=16, hidden_ch=None, hidden_depth=0):
        super().__init__()
        hidden_ch = hidden_ch or max(out_ch * 2, 32)
        hidden_depth = max(0, int(hidden_depth))
        self.calib = ChannelCalibration(in_ch)
        layers = [ConvGNAct(in_ch, hidden_ch, kernel_size=1, padding=0)]
        layers.extend(ConvGNAct(hidden_ch, hidden_ch, kernel_size=3) for _ in range(hidden_depth))
        layers.extend([
            ConvGNAct(hidden_ch, out_ch, kernel_size=3),
            ConvGNAct(out_ch, out_ch, kernel_size=3),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.calib(x))


class TesseraIoUFusionLightUNet(nn.Module):
    """AlphaEarth LightUNet with a Tessera residual IoU correction branch.

    Input layout is fixed by MultiPixelEmbeddingDataset when called as
    AlphaEarth primary + Tessera secondary:
      - channels [0:64] are AlphaEarth and drive the U-Net, fractions, and height
      - remaining channels are Tessera and drive only residual presence logits
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 tessera_presence_ch=16, tessera_hidden_ch=None,
                 tessera_hidden_depth=0, height_specialist_depth=0,
                 base_ch=32, height_gate_source="alpha",
                 height_hidden_ch=None, height_trunk_depth=2,
                 height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared"):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraIoUFusionLightUNet assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "TesseraIoUFusionLightUNet expects concatenated AlphaEarth+Tessera "
                f"input with >{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=tessera_presence_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
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
        )

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_stem(tessera)
        return self.head(alpha_feat, return_aux=return_aux, presence_extra=tessera_feat)


class TesseraIoUFusionGatedLightUNet(nn.Module):
    """AlphaEarth + Tessera fused at the **trunk feature level** via a learned gate.

    Distinct from TesseraIoUFusionLightUNet, which adds Tessera only as a
    residual correction on the presence logits. Here Tessera is promoted to
    a peer feature stream (base_ch wide) that fuses into the trunk features
    just before the head, so fractions / FiLM / height all benefit from
    Tessera, not just presence.

    Flow:
        AE  -> alpha_unet.forward_features -> alpha_feat (B, base_ch, H, W)
        TES -> tessera_feature_stem        -> tessera_feat (B, base_ch, H, W)
        gate = sigmoid(gate_conv(concat(alpha_feat, tessera_feat)))  in (0,1)
        fused = gate * alpha_feat + (1 - gate) * tessera_feat   (tied)
            or   G_AE * alpha_feat + G_TES * tessera_feat       (untied)

    Warm-start: gate's final conv is zero-init with a large positive bias so
    sigmoid(b) ~ 1 at step 0, fused ~ alpha_feat. Identical t=0 behavior to
    AE-only baseline; any divergence is learned from data.

    The presence-logit residual path is also preserved (set
    tessera_presence_ch>0 to keep it). It is orthogonal to the trunk fusion
    and the two compose: trunk fusion improves shared features for all heads
    (incl. height); the presence residual is a lightweight correction
    directly on the IoU output.
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 tessera_presence_ch=0, tessera_hidden_ch=None,
                 tessera_hidden_depth=0, height_specialist_depth=0,
                 base_ch=32, gate_init_bias=4.0,
                 gate_mode="simple", gate_untied=False,
                 modality_dropout=0.0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared"):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraIoUFusionGatedLightUNet assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "TesseraIoUFusionGatedLightUNet expects concatenated AlphaEarth+Tessera "
                f"input with >{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.gate_untied = bool(gate_untied)
        self.modality_dropout = float(modality_dropout)
        tessera_channels = n_channels - alpha_channels

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch,
                                    norm_kind=norm_kind)
        self.alpha_unet.head = nn.Identity()
        # Tessera stem outputs base_ch (peer width with alpha_feat) so the
        # gate can mix the two streams at the same width.
        self.tessera_feature_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=base_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        self.gate_conv = _build_fusion_gate(
            base_ch, mode=gate_mode, untied=self.gate_untied,
            init_bias=gate_init_bias,
        )

        # Optional small presence-logit residual on top (composes with gate).
        self.tessera_presence_ch = int(tessera_presence_ch)
        if self.tessera_presence_ch > 0:
            self.presence_extra_proj = nn.Conv2d(base_ch, self.tessera_presence_ch, 1)
        else:
            self.presence_extra_proj = None

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
        )

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )

        fused = _apply_fusion_gate(self.gate_conv, alpha_feat, tessera_feat,
                                   untied=self.gate_untied)

        presence_extra = (self.presence_extra_proj(tessera_feat)
                          if self.presence_extra_proj is not None else None)
        return self.head(fused, return_aux=return_aux,
                         presence_extra=presence_extra)


class TesseraIoUFusionUNetPlusPlus(nn.Module):
    """TesseraIoUFusion variant with a UNet++ AlphaEarth backbone.

    Drop-in replacement for ``TesseraIoUFusionLightUNet``: the only change is
    swapping ``LightUNet`` for ``LightUNetPlusPlus``. Input layout, residual
    Tessera presence branch, and head are unchanged, so all existing presets
    (presence_centered loss, specialist depth, height boosts, etc.) carry
    over without modification.
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 tessera_presence_ch=16, tessera_hidden_ch=None,
                 tessera_hidden_depth=0, height_specialist_depth=0,
                 base_ch=32, height_gate_source="alpha",
                 height_hidden_ch=None, height_trunk_depth=2,
                 height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared"):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraIoUFusionUNetPlusPlus assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "TesseraIoUFusionUNetPlusPlus expects concatenated AlphaEarth+Tessera "
                f"input with >{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        self.alpha_unet = LightUNetPlusPlus(alpha_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=tessera_presence_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
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
        )

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_stem(tessera)
        return self.head(alpha_feat, return_aux=return_aux, presence_extra=tessera_feat)


class TokenSharedAdapter(nn.Module):
    """Project token-neck features into the AlphaEarth shared feature space."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, out_ch, kernel_size=1, padding=0),
            ConvGNAct(out_ch, out_ch, kernel_size=3),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.gate * self.net(x)


class TesseraTokenSharedProbeLightUNet(nn.Module):
    """Q-style AlphaEarth+Tessera fusion with one token source injected into shared features.

    Input is a tuple/list ``(pixel, token)``:
      - pixel channels [0:64] are AlphaEarth, remaining channels are Tessera
      - token is TerraMind/THOR at 16x16 and is decoded by TokenNeckDecoder

    The token adapter is gated at zero, so the model starts from the Q family
    and learns whether shared token features improve the downstream tasks.
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, tessera_presence_ch=16,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 height_gate_source="fused", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraTokenSharedProbeLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "TesseraTokenSharedProbeLightUNet expects AlphaEarth+Tessera pixel "
                f"input with >{alpha_channels} channels, got {pixel_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = pixel_channels - alpha_channels

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=tessera_presence_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        self.token_neck = TokenNeckDecoder(n_channels=token_channels, n_classes=n_classes)
        self.token_neck.head = nn.Identity()
        self.token_adapter = TokenSharedAdapter(in_ch=32, out_ch=base_ch)
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
        )

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("TesseraTokenSharedProbeLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        token_feat = self.token_neck.forward_features(token)
        fused_feat = alpha_feat + self.token_adapter(token_feat)
        tessera_feat = self.tessera_stem(tessera)
        return self.head(fused_feat, return_aux=return_aux, presence_extra=tessera_feat)


class TesseraTokenFusionSharedProbeLightUNet(nn.Module):
    """Q-style AlphaEarth+Tessera fusion with same-model S1/S2 token fusion.

    Input is a tuple/list ``(pixel, token)``:
      - pixel channels [0:64] are AlphaEarth, remaining channels are Tessera
      - token is [S1, S2] channel-concatenated at 16x16

    S1/S2 are fused before the token neck, then injected into the AlphaEarth
    shared feature map through the same zero-gated adapter used by the
    single-token shared probe.
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, tessera_presence_ch=16,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 height_gate_source="fused", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, normalize_tokens=False):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraTokenFusionSharedProbeLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "TesseraTokenFusionSharedProbeLightUNet expects AlphaEarth+Tessera pixel "
                f"input with >{alpha_channels} channels, got {pixel_channels}"
            )
        if token_channels % 2 != 0:
            raise ValueError(
                "TesseraTokenFusionSharedProbeLightUNet expects [S1, S2] token "
                f"channel concatenation, got {token_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = pixel_channels - alpha_channels
        fused_token_channels = token_channels // 2

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=tessera_presence_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        self.token_fusion = SameModelTokenFusion(
            token_channels,
            fused_channels=fused_token_channels,
            normalize_inputs=normalize_tokens,
            initial_gate=0.1 if normalize_tokens else 0.0,
        )
        self.token_neck = TokenNeckDecoder(
            n_channels=fused_token_channels,
            n_classes=n_classes,
        )
        self.token_neck.head = nn.Identity()
        self.token_adapter = TokenSharedAdapter(in_ch=32, out_ch=base_ch)
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
        )

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("TesseraTokenFusionSharedProbeLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        token = self.token_fusion(token)
        token_feat = self.token_neck.forward_features(token)
        fused_feat = alpha_feat + self.token_adapter(token_feat)
        tessera_feat = self.tessera_stem(tessera)
        return self.head(fused_feat, return_aux=return_aux, presence_extra=tessera_feat)


class TokenPyramidProvider(nn.Module):
    """Build a token feature pyramid, optionally fusing same-model S1/S2 first."""

    def __init__(self, token_channels, fusion_kind="single", normalize_tokens=False,
                 level_channels=(256, 128, 64, 32)):
        super().__init__()
        self.fusion_kind = fusion_kind
        if fusion_kind == "single":
            self.token_norm = TokenChannelStandardize(token_channels) if normalize_tokens else nn.Identity()
            self.fusion = nn.Identity()
            fused_channels = token_channels
        elif fusion_kind == "s1s2_gated":
            if token_channels % 2 != 0:
                raise ValueError("s1s2_gated token pyramid expects concatenated S1/S2 channels")
            fused_channels = token_channels // 2
            self.token_norm = nn.Identity()
            self.fusion = SameModelTokenFusion(
                token_channels,
                fused_channels=fused_channels,
                normalize_inputs=normalize_tokens,
                initial_gate=0.1 if normalize_tokens else 0.0,
            )
        elif fusion_kind == "s1s2_xattn":
            if token_channels % 2 != 0:
                raise ValueError("s1s2_xattn token pyramid expects concatenated S1/S2 channels")
            fused_channels = token_channels // 2
            self.token_norm = nn.Identity()
            self.fusion = CrossModalTokenFusion(
                token_channels,
                fused_channels=fused_channels,
                normalize_inputs=normalize_tokens,
                initial_gate=0.1 if normalize_tokens else 0.0,
            )
        else:
            raise ValueError(f"Unknown token fusion kind: {fusion_kind}")
        self.neck = TokenPyramidNeck(fused_channels, level_channels=level_channels)

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


class SigmoidFusionTokenAdapter(nn.Module):
    """Spatial sigmoid-gate fusion of a token feature into one U-Net scale.

    Drop-in for ``GatedTokenScaleResidual`` at the same call site
    ``forward(target, token_feat)``. Replaces the zero-init scalar
    ``target + g * proj(token)`` with the trunk-level gate used by
    ``TesseraIoUFusionGatedLightUNet``:

        token_proj = proj(token)
        fused      = sigmoid(gate(cat(target, token_proj))) * target +
                     (1 - sigmoid(...)) * token_proj    # tied
        or  G_T * target + G_K * token_proj             # untied (GMU 2(a))

    Warm-start (``init_bias=4.0``) → sigmoid≈1 at step 0, fused == target,
    matching the zero-scalar t=0 behavior of GatedTokenScaleResidual; any
    divergence is learned from data.
    """

    def __init__(self, token_ch, target_ch, hidden_ch=None,
                 gate_mode="simple", gate_untied=False, gate_init_bias=4.0,
                 hadamard_residual=False):
        super().__init__()
        hidden_ch = hidden_ch or min(max(target_ch, 64), 256)
        self.proj = nn.Sequential(
            ConvGNAct(token_ch, hidden_ch, kernel_size=1, padding=0),
            ConvGNAct(hidden_ch, target_ch, kernel_size=3),
        )
        self.gate_untied = bool(gate_untied)
        self.gate_conv = _build_fusion_gate(
            target_ch, mode=gate_mode, untied=self.gate_untied,
            init_bias=gate_init_bias,
        )
        # BiFusion-style Hadamard residual: (W1*proj_TM) ⊙ (W2*target).
        # Validated by tools/diagnostic_ae_tm_spatial_alignment.py: at decoder64
        # AE trunk and TM-projected feature share +0.173 spatial-bonus R^2 over
        # channel-only pairing, so per-position elementwise interaction has
        # signal beyond what the sigmoid gate (a scalar mixer) can reach.
        # lambda init=0 keeps t=0 behavior identical to the sigmoid-gate-only
        # baseline; the Hadamard branch only contributes if it earns gradient.
        self.hadamard_residual = bool(hadamard_residual)
        if self.hadamard_residual:
            self.hada_W1 = nn.Conv2d(target_ch, target_ch, kernel_size=1, bias=False)
            self.hada_W2 = nn.Conv2d(target_ch, target_ch, kernel_size=1, bias=False)
            self.hada_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, target, token_feat):
        if token_feat.shape[-2:] != target.shape[-2:]:
            token_feat = F.interpolate(
                token_feat, size=target.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        token_proj = self.proj(token_feat)
        fused = _apply_fusion_gate(self.gate_conv, target, token_proj,
                                   untied=self.gate_untied)
        if self.hadamard_residual:
            hada = self.hada_W1(token_proj) * self.hada_W2(target)
            fused = fused + self.hada_lambda * hada
        return fused


class _DPTResidualConvUnit(nn.Module):
    """ResidualConvUnit (Ranftl 2021): pre-act 2x (3x3 conv + GN) + skip residual.

    Uses GroupNorm + GELU for consistency with the rest of the project; the
    paper uses BN + ReLU. Skip is identity.
    """

    def __init__(self, channels):
        super().__init__()
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, x):
        out = self.norm1(self.conv1(self.act1(x)))
        out = self.norm2(self.conv2(self.act2(out)))
        return x + out


class CompactDPTDecoder(nn.Module):
    """Tiny DPT-style decoder over a single ViT token grid (B, D, 16, 16).

    Mirrors the *Reassemble + RefineNet Fusion* design from
    Ranftl et al. 2021 (Dense Prediction Transformers), but compressed for
    small-data regimes (~2k patches): a single fixed stage width ``stage_ch``,
    two RCUs per fusion block, stage-to-stage top-down propagation. Output is
    a dense feature map at full resolution (256x256) with ``out_ch`` channels,
    ready to feed ``MultiTaskPredictionHead``.

    Stages are at scales {32, 64, 128, 256}, each reassembled from the same
    16x16 token grid by 1x1 projection + bilinear upsample + 3x3 conv. The
    fusion block at scale s receives:

        feat_s = RCU(RCU(upsample(prev_feat) + reassembled_s))
        out_s  = ConvGNAct(feat_s)

    The first stage (s=32) has prev_feat=None so it sees only its
    reassembled feature.

    Param count (default stage_ch=128, out_ch=48): ~3.7M.
    """

    def __init__(self, token_ch=768, stage_ch=128, out_ch=48,
                 scales=(32, 64, 128, 256)):
        super().__init__()
        if stage_ch <= 0 or out_ch <= 0:
            raise ValueError("stage_ch and out_ch must be positive")
        self.scales = tuple(scales)
        self.proj = nn.Conv2d(token_ch, stage_ch, kernel_size=1)
        self.reassemble_conv = nn.ModuleList([
            ConvGNAct(stage_ch, stage_ch, kernel_size=3) for _ in self.scales
        ])
        self.rcu1 = nn.ModuleList([
            _DPTResidualConvUnit(stage_ch) for _ in self.scales
        ])
        self.rcu2 = nn.ModuleList([
            _DPTResidualConvUnit(stage_ch) for _ in self.scales
        ])
        self.fuse_out = nn.ModuleList([
            ConvGNAct(stage_ch, stage_ch, kernel_size=3) for _ in self.scales
        ])
        self.head_proj = ConvGNAct(stage_ch, out_ch, kernel_size=3)

    def _reassemble(self, proj, sz, conv):
        up = F.interpolate(proj, size=(sz, sz), mode="bilinear", align_corners=False)
        return conv(up)

    def forward(self, token):  # (B, D, 16, 16)
        proj = self.proj(token)
        feat = None
        for i, sz in enumerate(self.scales):
            reassembled = self._reassemble(proj, sz, self.reassemble_conv[i])
            if feat is None:
                feat = reassembled
            else:
                feat = F.interpolate(feat, size=(sz, sz), mode="bilinear",
                                     align_corners=False)
                feat = feat + reassembled
            feat = self.rcu1[i](feat)
            feat = self.rcu2[i](feat)
            feat = self.fuse_out[i](feat)
        return self.head_proj(feat)


class CompactDPTTokenOnly(nn.Module):
    """Pure-token model: tokens-only DPT-compact decoder + multi-task head.

    Accepts the same ``(pixel, token)`` input tuple as the cross-level fusion
    models for pipeline compatibility, but **discards the pixel half** and
    decodes only from the token grid. Used to measure the ceiling of a
    foundation-model token source (e.g. TerraMind/THOR-S2) on its own,
    without confounding contributions from AlphaEarth or Tessera.

    The head is the standard ``MultiTaskPredictionHead`` so we can compare
    apples-to-apples with xfusion baselines (presence_3way_deep, FiLM,
    height specialist, etc.).
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 stage_ch=128, base_ch=48,
                 height_specialist_depth=0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None):
        super().__init__()
        if n_classes != 4:
            raise ValueError("CompactDPTTokenOnly assumes 4 output channels")
        self.supports_aux_outputs = True
        self.pixel_channels = pixel_channels  # ignored, kept for kwargs symmetry
        self.decoder = CompactDPTDecoder(
            token_ch=token_channels, stage_ch=stage_ch, out_ch=base_ch,
        )
        self.head = MultiTaskPredictionHead(
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
            raise ValueError(
                "CompactDPTTokenOnly expects (pixel, token) input tuple "
                "(pixel is ignored)"
            )
        _, token = x
        feat = self.decoder(token)
        return self.head(feat, return_aux=return_aux)


class TokenDPTBackbone(nn.Module):
    """DPT-style multi-scale token decoder for cross-level AE fusion.

    Drop-in replacement for the *output side* of ``TokenPyramidProvider``:
    same forward contract ``token -> {scale: feat at scale}`` for the scales
    in ``scales``, but with two structural upgrades borrowed from
    Ranftl et al. 2021 (Dense Prediction Transformers):

      1. **Stage-to-stage top-down propagation** — the token feature at the
         coarsest scale is RCU-refined, then upsampled and *added* to the
         next-finer reassembled feature before that level's RCU. This lets
         token semantics at 32x32 flow into the 64x64 and 128x128 outputs
         (independent ``TokenPyramidProvider`` levels do not share signal).
      2. **RCU residual refinement** — each stage runs two
         ``_DPTResidualConvUnit`` blocks (4 conv + 2 skip residuals total)
         instead of the single 3x3 conv used by the pyramid neck.

    Output channels are fixed at ``stage_ch`` for every scale, regardless of
    pyramid level (the pyramid neck instead uses the wider-channel-thinner-
    spatial trade with c32=128, c64=64, c128=32).

    Set ``normalize_tokens=True`` for token sources whose raw values are far
    from unit scale (e.g. THOR-S2 lands in +-2e4); a per-channel
    standardization runs before projection.
    """

    def __init__(self, token_ch=768, stage_ch=128, scales=(32, 64, 128),
                 normalize_tokens=False):
        super().__init__()
        if not scales:
            raise ValueError("TokenDPTBackbone requires at least one scale")
        self.scales = tuple(sorted(set(scales)))  # ensure ascending order
        self.token_norm = (
            TokenChannelStandardize(token_ch) if normalize_tokens else nn.Identity()
        )
        self.proj = nn.Conv2d(token_ch, stage_ch, kernel_size=1)
        self.reassemble_conv = nn.ModuleList([
            ConvGNAct(stage_ch, stage_ch, kernel_size=3) for _ in self.scales
        ])
        self.rcu1 = nn.ModuleList([
            _DPTResidualConvUnit(stage_ch) for _ in self.scales
        ])
        self.rcu2 = nn.ModuleList([
            _DPTResidualConvUnit(stage_ch) for _ in self.scales
        ])
        self.fuse_out = nn.ModuleList([
            ConvGNAct(stage_ch, stage_ch, kernel_size=3) for _ in self.scales
        ])

    def forward(self, token):
        token = self.token_norm(token)
        proj = self.proj(token)
        outputs = {}
        feat = None
        for i, sz in enumerate(self.scales):
            up = F.interpolate(proj, size=(sz, sz), mode="bilinear",
                               align_corners=False)
            reassembled = self.reassemble_conv[i](up)
            if feat is None:
                feat = reassembled
            else:
                feat = F.interpolate(feat, size=(sz, sz), mode="bilinear",
                                     align_corners=False)
                feat = feat + reassembled
            feat = self.rcu1[i](feat)
            feat = self.rcu2[i](feat)
            outputs[sz] = self.fuse_out[i](feat)
        return outputs


class FiLMTokenModulator(nn.Module):
    """Cross-level AE/token fusion via FiLM modulation (Perez et al. 2018).

    Drop-in replacement for ``GatedTokenScaleResidual`` and
    ``SigmoidFusionTokenAdapter`` at AE U-Net cross-level fusion points.
    Same call contract ``forward(target, token_feat)``.

    Mechanism (offset form, used in adapters / FiLM-tuned NeRFs):

        token_proj = MLP(token_feat)                 # B, hidden_ch
        gamma, beta = split(film_conv(token_proj))   # B, target_ch each
        fused = target * (1 + gamma) + beta

    The final ``film_conv`` is zero-initialized (weight=0, bias=0), so at
    step 0 ``gamma = beta = 0`` and ``fused == target`` exactly. Identical
    t=0 behavior to ``SigmoidFusionTokenAdapter`` with bias=4.0 and the
    same warm-start guarantee as the existing scalar gate.

    Conceptual difference vs sigmoid gate:
      - Sigmoid gate: ``fused = G * target + (1 - G) * proj(token)`` -- token
        replaces target where the gate opens. Tends to keep ``G ~ 1`` when
        token is uninformative (xfusion gate diagnostic showed mean
        ``(1-G) ~ 0.05``).
      - FiLM: ``fused = target * (1 + g) + b`` -- token never replaces
        target, only modulates it (multiplicative scale + additive bias).
        Even when token is mostly uninformative, the model is encouraged
        to use it as a gentle conditioning rather than mute it.
    """

    def __init__(self, token_ch, target_ch, hidden_ch=None):
        super().__init__()
        hidden_ch = hidden_ch or min(max(target_ch, 64), 256)
        self.proj = nn.Sequential(
            ConvGNAct(token_ch, hidden_ch, kernel_size=1, padding=0),
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
        )
        self.film_conv = nn.Conv2d(hidden_ch, 2 * target_ch, kernel_size=1)
        nn.init.zeros_(self.film_conv.weight)
        nn.init.zeros_(self.film_conv.bias)

    def forward(self, target, token_feat):
        if token_feat.shape[-2:] != target.shape[-2:]:
            token_feat = F.interpolate(
                token_feat, size=target.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        h = self.proj(token_feat)
        gamma_beta = self.film_conv(h)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return target * (1.0 + gamma) + beta


class TesseraTokenCrossLevelFusionLightUNet(nn.Module):
    """Q-style AlphaEarth+Tessera with conservative cross-level token injection.

    Unlike the shared-probe model, token features are not added only at the
    final full-resolution shared map. A token pyramid is aligned to selected
    AlphaEarth U-Net stages and injected as zero-gated residuals, so the model
    starts exactly on the Q-family path and can opt into token signal only at
    the requested levels.
    """

    _TOKEN_LEVEL_CHANNELS = {32: 128, 64: 64, 128: 32}

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, tessera_presence_ch=16,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 height_gate_source="fused", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, token_fusion_kind="single",
                 fusion_points=("bottleneck",), normalize_tokens=False,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None,
                 token_gate_kind="scalar",
                 token_backbone_kind="pyramid", token_stage_ch=128,
                 gate_mode="simple", gate_untied=False, gate_init_bias=4.0,
                 tessera_trunk_gate=False,
                 tessera_trunk_hidden_ch=None, tessera_trunk_hidden_depth=0,
                 modality_dropout=0.0, token_hadamard_residual=False):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraTokenCrossLevelFusionLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "TesseraTokenCrossLevelFusionLightUNet expects AlphaEarth+Tessera pixel "
                f"input with >{alpha_channels} channels, got {pixel_channels}"
            )
        allowed_points = {"bottleneck", "decoder64", "decoder128"}
        unknown = set(fusion_points) - allowed_points
        if unknown:
            raise ValueError(f"Unknown cross-level fusion point(s): {sorted(unknown)}")
        if token_gate_kind not in {"scalar", "sigmoid", "film"}:
            raise ValueError(
                f"token_gate_kind must be 'scalar', 'sigmoid' or 'film', got {token_gate_kind!r}"
            )
        if token_backbone_kind not in {"pyramid", "dpt"}:
            raise ValueError(
                f"token_backbone_kind must be 'pyramid' or 'dpt', got {token_backbone_kind!r}"
            )
        if token_backbone_kind == "dpt" and token_fusion_kind != "single":
            # TokenDPTBackbone consumes a single (already fused) token grid;
            # cross-modal pre-fusion (s1s2_gated/s1s2_xattn) is pyramid-only.
            raise ValueError(
                "token_backbone_kind='dpt' currently requires token_fusion_kind='single'"
            )
        if token_hadamard_residual and token_gate_kind != "sigmoid":
            raise ValueError(
                "token_hadamard_residual=True requires token_gate_kind='sigmoid' "
                f"(only SigmoidFusionTokenAdapter supports it), got {token_gate_kind!r}"
            )

        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.fusion_points = tuple(fusion_points)
        self.token_gate_kind = token_gate_kind
        self.token_backbone_kind = token_backbone_kind
        self.token_hadamard_residual = bool(token_hadamard_residual)
        self.tessera_trunk_gate = bool(tessera_trunk_gate)
        self.gate_untied = bool(gate_untied)
        self.modality_dropout = float(modality_dropout)
        tessera_channels = pixel_channels - alpha_channels
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=tessera_presence_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        ) if tessera_presence_ch > 0 else None

        # --- Token side: either the legacy pyramid neck or a DPT backbone ---
        # Both expose a forward() returning {scale: token_feat_at_scale}.
        # `_token_input_channels[scale]` is the channel count seen by the
        # downstream cross-level adapter and is what _make_adapter consumes.
        if token_backbone_kind == "dpt":
            scale_lookup = {"bottleneck": 32, "decoder64": 64, "decoder128": 128}
            dpt_scales = tuple(sorted({scale_lookup[p] for p in self.fusion_points}))
            self.token_pyramid = None
            self.token_dpt = TokenDPTBackbone(
                token_ch=token_channels, stage_ch=token_stage_ch,
                scales=dpt_scales, normalize_tokens=normalize_tokens,
            )
            self._token_input_channels = {sz: token_stage_ch for sz in dpt_scales}
        else:
            self.token_dpt = None
            self.token_pyramid = TokenPyramidProvider(
                token_channels,
                fusion_kind=token_fusion_kind,
                normalize_tokens=normalize_tokens,
            )
            self._token_input_channels = dict(self._TOKEN_LEVEL_CHANNELS)

        # Optional trunk-level Tessera gate (a la TesseraIoUFusionGatedLightUNet).
        # Mixes a base_ch-wide Tessera feature into the AlphaEarth trunk feature
        # at the end of the decoder, just before the head.
        if self.tessera_trunk_gate:
            self.tessera_trunk_stem = TesseraCompressionStem(
                tessera_channels,
                out_ch=base_ch,
                hidden_ch=tessera_trunk_hidden_ch,
                hidden_depth=tessera_trunk_hidden_depth,
            )
            self.trunk_gate_conv = _build_fusion_gate(
                base_ch, mode=gate_mode, untied=self.gate_untied,
                init_bias=gate_init_bias,
            )
        else:
            self.tessera_trunk_stem = None
            self.trunk_gate_conv = None

        def _make_adapter(token_ch, target_ch):
            if self.token_gate_kind == "film":
                return FiLMTokenModulator(token_ch, target_ch)
            if self.token_gate_kind == "sigmoid":
                return SigmoidFusionTokenAdapter(
                    token_ch, target_ch, gate_mode=gate_mode,
                    gate_untied=self.gate_untied, gate_init_bias=gate_init_bias,
                    hadamard_residual=self.token_hadamard_residual,
                )
            return GatedTokenScaleResidual(token_ch, target_ch)

        self.bottleneck_adapter = (
            _make_adapter(self._token_input_channels[32], c4)
            if "bottleneck" in self.fusion_points else None
        )
        self.decoder64_adapter = (
            _make_adapter(self._token_input_channels[64], c3)
            if "decoder64" in self.fusion_points else None
        )
        self.decoder128_adapter = (
            _make_adapter(self._token_input_channels[128], c2)
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
        token_pyr = (
            self.token_dpt(token)
            if self.token_dpt is not None
            else self.token_pyramid(token)
        )

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

        if self.tessera_trunk_stem is not None:
            trunk_tes = self.tessera_trunk_stem(tessera)
            trunk_tes = _maybe_drop_modality(
                trunk_tes, self.modality_dropout, self.training
            )
            feat = _apply_fusion_gate(
                self.trunk_gate_conv, feat, trunk_tes, untied=self.gate_untied
            )

        presence_extra = (
            self.tessera_stem(tessera) if self.tessera_stem is not None else None
        )
        return self.head(feat, return_aux=return_aux, presence_extra=presence_extra)


class TesseraTokenCrossLevelFusionWaterBypassLightUNet(TesseraTokenCrossLevelFusionLightUNet):
    """Cross-level S2 fusion with a water-only decoder bypass.

    Identical to ``TesseraTokenCrossLevelFusionLightUNet`` except that the
    water presence head consumes a parallel decoder feature that has not been
    perturbed by the decoder64 token adapter. Building, tree, height, fraction,
    FiLM, and the Tessera presence residual all continue to consume the
    TerraMind-augmented main path. The bypass decoder shares all weights with
    the main decoder (up2/conv2/up3/conv3); the only structural difference is
    that the adapter residual is skipped on the water branch.

    Requires ``decoder64`` in ``fusion_points`` and ``presence_head_kind ==
    'split_all'`` so the head can route the water branch independently.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "decoder64" not in self.fusion_points:
            raise ValueError(
                "Water bypass requires decoder64 in fusion_points (the bypass "
                "is taken right before the decoder64 adapter)."
            )
        if self.head.presence_head_kind != "split_all":
            raise ValueError(
                "Water bypass requires presence_head_kind='split_all' so the "
                "water branch can be routed off the bypass feature."
            )

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "TesseraTokenCrossLevelFusionWaterBypassLightUNet expects (pixel, token) input"
            )
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]
        token_pyr = (
            self.token_dpt(token)
            if self.token_dpt is not None
            else self.token_pyramid(token)
        )

        x1 = self.alpha_unet.inc(alpha)
        x2 = self.alpha_unet.down1(x1)
        x3 = self.alpha_unet.down2(x2)
        x4 = self.alpha_unet.down3(x3)
        if self.bottleneck_adapter is not None:
            x4 = self.bottleneck_adapter(x4, token_pyr[32])

        feat = self.alpha_unet.up1(x4)
        feat = torch.cat([x3, feat], dim=1)
        feat = self.alpha_unet.conv1(feat)

        # Snapshot the pre-adapter decoder64 feature for the water bypass.
        bypass_feat = feat

        # Main path: apply the decoder64 token residual.
        feat = self.decoder64_adapter(feat, token_pyr[64])

        feat = self.alpha_unet.up2(feat)
        feat = torch.cat([x2, feat], dim=1)
        feat = self.alpha_unet.conv2(feat)
        if self.decoder128_adapter is not None:
            feat = self.decoder128_adapter(feat, token_pyr[128])
        feat = self.alpha_unet.up3(feat)
        feat = torch.cat([x1, feat], dim=1)
        feat = self.alpha_unet.conv3(feat)

        # Bypass path uses the same decoder weights but never sees the adapter.
        bypass_feat = self.alpha_unet.up2(bypass_feat)
        bypass_feat = torch.cat([x2, bypass_feat], dim=1)
        bypass_feat = self.alpha_unet.conv2(bypass_feat)
        bypass_feat = self.alpha_unet.up3(bypass_feat)
        bypass_feat = torch.cat([x1, bypass_feat], dim=1)
        bypass_feat = self.alpha_unet.conv3(bypass_feat)

        tessera_feat = self.tessera_stem(tessera)
        return self.head(
            feat,
            return_aux=return_aux,
            presence_extra=tessera_feat,
            water_bypass_x=bypass_feat,
        )


class TokenDecoder64HeadResidual(nn.Module):
    """Bounded decoder64 token residual for selected presence channels and height.

    The residual is deliberately applied after the Q-family head rather than to
    the shared feature trunk. This lets ablations decide exactly whether
    TerraMind S2 is allowed to touch water.
    """

    _PRESENCE_CHANNELS = {
        "nonwater": (0, 1),
        "all": (0, 1, 2),
        "water": (2,),
        "none": (),
    }

    def __init__(self, token_ch=64, hidden_ch=96, presence_mode="nonwater",
                 height_residual=True, max_presence_delta=4.0,
                 max_height_delta_m=10.0):
        super().__init__()
        if presence_mode not in self._PRESENCE_CHANNELS:
            raise ValueError(
                f"presence_mode must be one of {sorted(self._PRESENCE_CHANNELS)}, "
                f"got {presence_mode!r}"
            )
        self.presence_mode = presence_mode
        self.presence_channels = self._PRESENCE_CHANNELS[presence_mode]
        self.height_residual = bool(height_residual)
        self.max_presence_delta = float(max_presence_delta)
        self.max_height_delta_norm = float(max_height_delta_m) / HEIGHT_NORM_CONSTANT

        self.trunk = nn.Sequential(
            ConvGNAct(token_ch, hidden_ch, kernel_size=1, padding=0),
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
        )
        if self.presence_channels:
            self.presence_proj = nn.Conv2d(hidden_ch, len(self.presence_channels), 1)
            self.presence_gate = nn.Parameter(torch.zeros(1, len(self.presence_channels), 1, 1))
        else:
            self.presence_proj = None
            self.presence_gate = None
        if self.height_residual:
            self.height_proj = nn.Conv2d(hidden_ch, 2, 1)
            self.height_gate = nn.Parameter(torch.zeros(1, 2, 1, 1))
        else:
            self.height_proj = None
            self.height_gate = None

    def forward(self, token_feat, target_size):
        if token_feat.shape[-2:] != target_size:
            token_feat = F.interpolate(
                token_feat,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        feat = self.trunk(token_feat)
        b, _, h, w = feat.shape

        presence_delta = feat.new_zeros(b, 3, h, w)
        if self.presence_proj is not None:
            raw = (
                self.presence_gate
                * torch.tanh(self.presence_proj(feat))
                * self.max_presence_delta
            )
            parts = [feat.new_zeros(b, 1, h, w) for _ in range(3)]
            for raw_idx, channel_idx in enumerate(self.presence_channels):
                parts[channel_idx] = raw[:, raw_idx:raw_idx + 1]
            presence_delta = torch.cat(parts, dim=1)

        height_delta = feat.new_zeros(b, 2, h, w)
        if self.height_proj is not None:
            height_delta = (
                self.height_gate
                * torch.tanh(self.height_proj(feat))
                * self.max_height_delta_norm
            )
        return presence_delta, height_delta


class TesseraTokenDecoder64HeadResidualLightUNet(nn.Module):
    """Q-style model with TerraMind S2 routed into selected output heads only."""

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, tessera_presence_ch=16,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 height_gate_source="fused", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, presence_mode="nonwater",
                 height_residual=True):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraTokenDecoder64HeadResidualLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "TesseraTokenDecoder64HeadResidualLightUNet expects AlphaEarth+Tessera "
                f"pixel input with >{alpha_channels} channels, got {pixel_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.height_gate_source = height_gate_source
        tessera_channels = pixel_channels - alpha_channels

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=tessera_presence_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
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
        )
        self.token_pyramid = TokenPyramidProvider(
            token_channels,
            fusion_kind="single",
            normalize_tokens=False,
        )
        self.token_residual = TokenDecoder64HeadResidual(
            token_ch=64,
            presence_mode=presence_mode,
            height_residual=height_residual,
        )

    @staticmethod
    def _reroute_height(aux, presence_logits, height_building, height_vegetation,
                        height_gate_source):
        height_gate_logits = (
            presence_logits
            if height_gate_source == "fused"
            else aux["alpha_presence_logits"]
        )
        height_presence_prob = torch.sigmoid(height_gate_logits)
        p_b = height_presence_prob[:, 0:1, :, :]
        p_v = height_presence_prob[:, 1:2, :, :]
        p_fg = 1.0 - (1.0 - p_b) * (1.0 - p_v)
        denom = p_b + p_v + 1e-6
        w_b = p_b / denom
        w_v = p_v / denom
        h_fg = w_b * height_building + w_v * height_vegetation
        return p_fg * h_fg + (1.0 - p_fg) * aux["height_base"]

    def _apply_token_residual(self, aux, token_feat):
        target_size = aux["presence_logits"].shape[-2:]
        presence_delta, height_delta = self.token_residual(token_feat, target_size)

        presence_logits = aux["presence_logits"] + presence_delta
        presence_prob = torch.sigmoid(presence_logits)
        height_building = torch.clamp(
            aux["height_building"] + height_delta[:, 0:1],
            min=0.0,
        )
        height_vegetation = torch.clamp(
            aux["height_vegetation"] + height_delta[:, 1:2],
            min=0.0,
        )
        height = self._reroute_height(
            aux,
            presence_logits,
            height_building,
            height_vegetation,
            self.height_gate_source,
        )
        out = torch.cat([presence_prob, height], dim=1)

        updated = dict(aux)
        updated["out"] = out
        updated["presence_logits"] = presence_logits
        updated["presence_prob"] = presence_prob
        updated["height_building"] = height_building
        updated["height_vegetation"] = height_vegetation
        updated["token_presence_delta_logits"] = presence_delta
        updated["token_height_delta"] = height_delta
        return updated

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("TesseraTokenDecoder64HeadResidualLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_stem(tessera)
        aux = self.head(alpha_feat, return_aux=True, presence_extra=tessera_feat)
        token_pyr = self.token_pyramid(token)
        aux = self._apply_token_residual(aux, token_pyr[64])
        return aux if return_aux else aux["out"]


class HeightSpecialistTokenResidual(nn.Module):
    """Predict bounded building/vegetation height corrections from token features."""

    def __init__(self, in_ch=32, hidden_ch=64, max_delta_m=10.0):
        super().__init__()
        self.max_delta_norm = float(max_delta_m) / HEIGHT_NORM_CONSTANT
        self.net = nn.Sequential(
            ConvGNAct(in_ch, hidden_ch, kernel_size=3),
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
            nn.Conv2d(hidden_ch, 2, kernel_size=1),
        )
        self.gate = nn.Parameter(torch.zeros(1, 2, 1, 1))

    def forward(self, x):
        return self.gate * torch.tanh(self.net(x)) * self.max_delta_norm


class TesseraTokenHeightResidualProbeLightUNet(nn.Module):
    """Q-style model where fused TerraMind tokens correct only height specialists.

    Presence logits, including water, stay on the AlphaEarth+Tessera Q path.
    Token features only add bounded corrections to building/vegetation height
    specialists before the existing presence-gated height routing.
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64, tessera_presence_ch=16,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32,
                 height_gate_source="fused", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, token_fusion_kind="neck"):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraTokenHeightResidualProbeLightUNet assumes 4 output channels")
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "TesseraTokenHeightResidualProbeLightUNet expects AlphaEarth+Tessera pixel "
                f"input with >{alpha_channels} channels, got {pixel_channels}"
            )
        if token_channels % 2 != 0:
            raise ValueError(
                "TesseraTokenHeightResidualProbeLightUNet expects [S1, S2] token "
                f"channel concatenation, got {token_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.height_gate_source = height_gate_source
        tessera_channels = pixel_channels - alpha_channels

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=tessera_presence_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        if token_fusion_kind == "xattn":
            self.token_path = TokenFusionNeckCrossAttentionDecoder(
                n_channels=token_channels,
                n_classes=n_classes,
            )
        elif token_fusion_kind == "neck":
            self.token_path = TokenFusionNeckDecoder(
                n_channels=token_channels,
                n_classes=n_classes,
            )
        else:
            raise ValueError(f"Unknown token_fusion_kind: {token_fusion_kind}")
        self.token_path.token_neck.head = nn.Identity()
        self.height_residual = HeightSpecialistTokenResidual(in_ch=32)
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
        )

    @staticmethod
    def _reroute_height(aux, height_building, height_vegetation, height_gate_source):
        height_gate_logits = (
            aux["presence_logits"]
            if height_gate_source == "fused"
            else aux["alpha_presence_logits"]
        )
        height_presence_prob = torch.sigmoid(height_gate_logits)
        p_b = height_presence_prob[:, 0:1, :, :]
        p_v = height_presence_prob[:, 1:2, :, :]
        p_fg = 1.0 - (1.0 - p_b) * (1.0 - p_v)
        denom = p_b + p_v + 1e-6
        w_b = p_b / denom
        w_v = p_v / denom
        h_fg = w_b * height_building + w_v * height_vegetation
        return p_fg * h_fg + (1.0 - p_fg) * aux["height_base"]

    def _apply_token_height_residual(self, aux, token_feat):
        correction = self.height_residual(token_feat)
        height_building = torch.clamp(aux["height_building"] + correction[:, 0:1], min=0.0)
        height_vegetation = torch.clamp(aux["height_vegetation"] + correction[:, 1:2], min=0.0)
        height = self._reroute_height(
            aux,
            height_building,
            height_vegetation,
            self.height_gate_source,
        )
        out = torch.cat([aux["presence_prob"], height], dim=1)

        updated = dict(aux)
        updated["out"] = out
        updated["height_building"] = height_building
        updated["height_vegetation"] = height_vegetation
        updated["token_height_correction"] = correction
        return updated

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("TesseraTokenHeightResidualProbeLightUNet expects (pixel, token) input")
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels, :, :]
        tessera = pixel[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_stem(tessera)
        aux = self.head(alpha_feat, return_aux=True, presence_extra=tessera_feat)
        token_feat = self.token_path.forward_features(token)
        aux = self._apply_token_height_residual(aux, token_feat)
        return aux if return_aux else aux["out"]


class EmbeddingRefiner(nn.Module):
    """Encoder-light, decoder-heavy model for pixel-aligned GFM embeddings.

    Flow (all at 256x256):
        input  -> calibrate -> 1x1 stem -> N/2 ConvNeXt blocks
               -> ASPP (multi-scale context) -> fuse
               -> N/2 ConvNeXt blocks
               -> seg head (3 ch: building/veg/water)
               -> height head (1 ch), conditioned on seg logits
               -> concat -> (B, 4, H, W)
    """

    def __init__(self, n_channels, n_classes=4, dim=96, n_blocks=6,
                 aspp_dim=128, drop=0.05):
        super().__init__()
        if n_classes != 4:
            raise ValueError("EmbeddingRefiner assumes 4 output channels: building, veg, water, height")
        self.supports_aux_outputs = True

        self.calib = ChannelCalibration(n_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, dim, 1, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU(),
        )

        half = n_blocks // 2
        self.refine_early = nn.Sequential(*[ConvNeXtBlock(dim, drop=drop) for _ in range(half)])
        self.aspp = ASPP(dim, aspp_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim + aspp_dim, dim, 1, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU(),
        )
        self.refine_late = nn.Sequential(*[ConvNeXtBlock(dim, drop=drop) for _ in range(n_blocks - half)])

        self.head = MultiTaskPredictionHead(dim, out_channels=n_classes, drop=drop)

    def forward(self, x, return_aux=False):
        x = self.calib(x)
        x = self.stem(x)
        x = self.refine_early(x)
        ctx = self.aspp(x)
        x = self.fuse(torch.cat([x, ctx], dim=1))
        x = self.refine_late(x)

        return self.head(x, return_aux=return_aux)


# ==========================================
# 4. HRNET FOR PIXEL-ALIGNED EMBEDDINGS
# ==========================================

class HRBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, kernel_size=3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
        )
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(_group_count(out_ch), out_ch),
            )
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


def _make_branch(channels, num_blocks):
    return nn.Sequential(*[HRBasicBlock(channels, channels) for _ in range(num_blocks)])


def _make_transition(in_channels, out_channels):
    layers = []
    num_in = len(in_channels)
    for i, out_ch in enumerate(out_channels):
        if i < num_in:
            if in_channels[i] == out_ch:
                layers.append(nn.Identity())
            else:
                layers.append(ConvGNAct(in_channels[i], out_ch, kernel_size=3))
        else:
            ops = []
            in_ch = in_channels[-1]
            for k in range(i + 1 - num_in):
                ops.append(ConvGNAct(in_ch, out_ch, kernel_size=3, stride=2))
                in_ch = out_ch
            layers.append(nn.Sequential(*ops))
    return nn.ModuleList(layers)


class HRModule(nn.Module):
    def __init__(self, channels, num_blocks=2):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList([_make_branch(ch, num_blocks) for ch in channels])
        self.fuse_layers = nn.ModuleList()

        for i, out_ch in enumerate(channels):
            transforms = nn.ModuleList()
            for j, in_ch in enumerate(channels):
                if i == j:
                    transforms.append(nn.Identity())
                elif j > i:
                    transforms.append(ConvGNAct(in_ch, out_ch, kernel_size=1, padding=0))
                else:
                    ops = []
                    cur_ch = in_ch
                    for k in range(i - j):
                        next_ch = out_ch if k == i - j - 1 else cur_ch
                        ops.append(ConvGNAct(cur_ch, next_ch, kernel_size=3, stride=2))
                        cur_ch = next_ch
                    transforms.append(nn.Sequential(*ops))
            self.fuse_layers.append(transforms)
        self.act = nn.GELU()

    def forward(self, xs):
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        fused = []
        for i, transforms in enumerate(self.fuse_layers):
            target_size = xs[i].shape[-2:]
            y = None
            for j, transform in enumerate(transforms):
                z = transform(xs[j])
                if j > i:
                    z = F.interpolate(z, size=target_size, mode='bilinear', align_corners=False)
                y = z if y is None else y + z
            fused.append(self.act(y))
        return fused


class HRStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_modules, num_blocks):
        super().__init__()
        self.transition = _make_transition(in_channels, out_channels)
        self.hr_modules = nn.ModuleList([
            HRModule(out_channels, num_blocks=num_blocks)
            for _ in range(num_modules)
        ])

    def forward(self, xs):
        transitioned = []
        for i, transition in enumerate(self.transition):
            source = xs[i] if i < len(xs) else xs[-1]
            transitioned.append(transition(source))
        xs = transitioned
        for module in self.hr_modules:
            xs = module(xs)
        return xs


class HRNetEmbedding(nn.Module):
    """HRNet-style high-resolution backbone for 256x256 GFM embeddings."""

    def __init__(self, n_channels, n_classes=4, width=18, drop=0.05):
        super().__init__()
        if n_classes != 4:
            raise ValueError("HRNetEmbedding assumes 4 output channels")
        self.supports_aux_outputs = True

        widths = [width, width * 2, width * 4, width * 8]
        self.calib = ChannelCalibration(n_channels)
        self.stem = nn.Sequential(
            ConvGNAct(n_channels, widths[0], kernel_size=3),
            HRBasicBlock(widths[0], widths[0]),
            HRBasicBlock(widths[0], widths[0]),
        )

        self.stage2 = HRStage([widths[0]], widths[:2], num_modules=1, num_blocks=2)
        self.stage3 = HRStage(widths[:2], widths[:3], num_modules=2, num_blocks=2)
        self.stage4 = HRStage(widths[:3], widths[:4], num_modules=2, num_blocks=2)

        fused_ch = sum(widths)
        head_ch = max(96, width * 4)
        self.fuse = nn.Sequential(
            ConvGNAct(fused_ch, head_ch, kernel_size=1, padding=0),
            ConvGNAct(head_ch, head_ch, kernel_size=3),
        )
        self.head = MultiTaskPredictionHead(head_ch, out_channels=n_classes, drop=drop)

    def forward(self, x, return_aux=False):
        x = self.calib(x)
        x = self.stem(x)
        feats = self.stage2([x])
        feats = self.stage3(feats)
        feats = self.stage4(feats)

        full_size = feats[0].shape[-2:]
        feats = [
            feat if feat.shape[-2:] == full_size
            else F.interpolate(feat, size=full_size, mode='bilinear', align_corners=False)
            for feat in feats
        ]
        x = self.fuse(torch.cat(feats, dim=1))
        return self.head(x, return_aux=return_aux)


# ==========================================
# 5. MODEL BUILDER
# ==========================================

def infer_model_type(n_channels):
    """
    Pick a sensible default architecture from input channel count.

    High-channel embeddings (e.g. 768-dim ViT tokens from TerraMind/THOR) live
    on a coarse 16x16 grid and need the upsampling decoder. Lower-channel,
    pixel-aligned embeddings (AlphaEarth 64, Tessera 128) are treated as
    already-encoded features and go through the EmbeddingRefiner.
    """
    if n_channels >= 512:
        return "decoder_residual"
    return "embedding_refiner"


def build_model(model_type, n_channels, n_classes, tessera_presence_ch=16,
                tessera_hidden_ch=None, tessera_hidden_depth=0,
                height_specialist_depth=0, lightunet_base_ch=32,
                height_gate_source="alpha", height_hidden_ch=None,
                height_trunk_depth=2, height_independent_branches=False,
                height_head_kind="linear", height_n_bins=64,
                height_bin_max_m=80.0, lightunet_norm_kind="bn",
                gate_mode="simple", gate_untied=False, gate_init_bias=4.0,
                modality_dropout=0.0):
    selected = model_type.lower()

    if selected == "auto":
        selected = infer_model_type(n_channels)
    if selected == "lightunet":
        return LightUNet(n_channels, n_classes, base_ch=lightunet_base_ch,
                         norm_kind=lightunet_norm_kind), selected
    if selected in {"lightunet_presence_2plus1", "lightunet_presence_3way",
                    "lightunet_presence_shared3"}:
        if selected.endswith("_2plus1"):
            preset_presence_kind = "split_water"
        elif selected.endswith("_shared3"):
            preset_presence_kind = "shared_split_all"
        else:
            preset_presence_kind = "split_all"
        return LightUNet(
            n_channels,
            n_classes,
            base_ch=lightunet_base_ch,
            norm_kind=lightunet_norm_kind,
            presence_head_kind=preset_presence_kind,
        ), selected
    if selected in ("lightunet_pp", "lightunet_unetpp"):
        return LightUNetPlusPlus(n_channels, n_classes, base_ch=lightunet_base_ch,
                                 norm_kind=lightunet_norm_kind), selected
    if selected == "decoder":
        selected = "decoder_residual"
    if selected == "decoder_residual":
        return EfficientDecoder256Fast(in_channels=n_channels, out_channels=n_classes), selected
    if selected == "token_neck":
        return TokenNeckDecoder(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "token_neck_norm":
        return TokenNeckNormDecoder(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "token_fusion_neck":
        return TokenFusionNeckDecoder(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "token_fusion_neck_norm":
        return TokenFusionNeckNormDecoder(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "token_fusion_neck_xattn":
        return TokenFusionNeckCrossAttentionDecoder(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "token_fusion_neck_xattn_norm":
        return TokenFusionNeckCrossAttentionNormDecoder(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "embedding_refiner":
        return EmbeddingRefiner(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "hrnet_w18":
        return HRNetEmbedding(n_channels=n_channels, n_classes=n_classes, width=18), selected
    if selected == "hrnet_w32":
        return HRNetEmbedding(n_channels=n_channels, n_classes=n_classes, width=32), selected
    if selected == "tessera_iou_fusion":
        return TesseraIoUFusionLightUNet(
            n_channels=n_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            norm_kind=lightunet_norm_kind,
        ), selected
    if selected in {"tessera_iou_fusion_presence_2plus1",
                    "tessera_iou_fusion_presence_3way",
                    "tessera_iou_fusion_presence_shared3"}:
        if selected.endswith("_2plus1"):
            preset_presence_kind = "split_water"
        elif selected.endswith("_shared3"):
            preset_presence_kind = "shared_split_all"
        else:
            preset_presence_kind = "split_all"
        return TesseraIoUFusionLightUNet(
            n_channels=n_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            norm_kind=lightunet_norm_kind,
            presence_head_kind=preset_presence_kind,
        ), selected
    if selected in {
        "tessera_iou_fusion_gated",
        "tessera_iou_fusion_gated_presence_2plus1",
        "tessera_iou_fusion_gated_presence_3way",
        "tessera_iou_fusion_gated_presence_shared3",
    }:
        if selected.endswith("_2plus1"):
            preset_presence_kind = "split_water"
        elif selected.endswith("_shared3"):
            preset_presence_kind = "shared_split_all"
        elif selected.endswith("_3way"):
            preset_presence_kind = "split_all"
        else:
            preset_presence_kind = "shared"
        return TesseraIoUFusionGatedLightUNet(
            n_channels=n_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
            gate_mode=gate_mode,
            gate_untied=gate_untied,
            gate_init_bias=gate_init_bias,
            modality_dropout=modality_dropout,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            norm_kind=lightunet_norm_kind,
            presence_head_kind=preset_presence_kind,
        ), selected
    if selected == "tessera_iou_fusion_unetpp":
        return TesseraIoUFusionUNetPlusPlus(
            n_channels=n_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            norm_kind=lightunet_norm_kind,
        ), selected
    if selected == "tessera_token_shared_probe":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "tessera_token_shared_probe expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return TesseraTokenSharedProbeLightUNet(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
        ), selected
    if selected in {"tessera_token_fusion_shared_probe", "tessera_token_fusion_shared_probe_norm"}:
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                f"{selected} expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return TesseraTokenFusionSharedProbeLightUNet(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            normalize_tokens=selected.endswith("_norm"),
        ), selected
    if selected in {"tessera_token_height_residual_probe", "tessera_token_xattn_height_residual_probe"}:
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                f"{selected} expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return TesseraTokenHeightResidualProbeLightUNet(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            token_fusion_kind="xattn" if "xattn" in selected else "neck",
        ), selected
    head_residual_configs = {
        "tessera_token_s2_nonwater_residual_decoder64": {
            "presence_mode": "nonwater",
            "height_residual": True,
        },
        "tessera_token_s2_all_residual_decoder64": {
            "presence_mode": "all",
            "height_residual": True,
        },
        "tessera_token_s2_water_residual_decoder64": {
            "presence_mode": "water",
            "height_residual": False,
        },
    }
    if selected in head_residual_configs:
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                f"{selected} expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        cfg = head_residual_configs[selected]
        return TesseraTokenDecoder64HeadResidualLightUNet(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_mode=cfg["presence_mode"],
            height_residual=cfg["height_residual"],
        ), selected
    crosslevel_configs = {
        "tessera_token_crosslevel_s2_bottleneck": {
            "token_fusion_kind": "single",
            "fusion_points": ("bottleneck",),
        },
        "tessera_token_crosslevel_s2_decoder64": {
            "token_fusion_kind": "single",
            "fusion_points": ("decoder64",),
        },
        "tessera_token_crosslevel_s2_decoder64_presence_2plus1": {
            "token_fusion_kind": "single",
            "fusion_points": ("decoder64",),
            "presence_head_kind": "split_water",
        },
        "tessera_token_crosslevel_s2_decoder64_presence_3way": {
            "token_fusion_kind": "single",
            "fusion_points": ("decoder64",),
            "presence_head_kind": "split_all",
        },
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep": {
            "token_fusion_kind": "single",
            "fusion_points": ("decoder64",),
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
        },
        "tessera_token_crosslevel_s2_bottleneck_decoder64_presence_3way_deep": {
            "token_fusion_kind": "single",
            "fusion_points": ("bottleneck", "decoder64"),
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
        },
        "tessera_token_crosslevel_s2_decoder64_decoder128_presence_3way_deep": {
            "token_fusion_kind": "single",
            "fusion_points": ("decoder64", "decoder128"),
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
        },
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_water_bypass": {
            "token_fusion_kind": "single",
            "fusion_points": ("decoder64",),
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
            "water_bypass": True,
        },
        # xfusion_019 + sigmoid-gate fusion at decoder64 (Task A): replaces the
        # zero-init scalar token residual with a spatial sigmoid gate that mixes
        # AlphaEarth trunk and TerraMind-S2 token features at base_ch*4 width.
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated": {
            "token_fusion_kind": "single",
            "fusion_points": ("decoder64",),
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
            "token_gate_kind": "sigmoid",
        },
        # Task A + Tessera trunk-level sigmoid gate (Task B): on top of the
        # TerraMind sigmoid gate at decoder64, also gate-fuses a base_ch-wide
        # Tessera feature stream into the AlphaEarth trunk at the end of the
        # decoder, mirroring TesseraIoUFusionGatedLightUNet. The original
        # tessera_presence_ch residual still composes orthogonally.
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated_tessera_gated": {
            "token_fusion_kind": "single",
            "fusion_points": ("decoder64",),
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
            "token_gate_kind": "sigmoid",
            "tessera_trunk_gate": True,
        },
        # xfusion_026 + BiFusion-style Hadamard residual at the decoder64 TM
        # adapter. Adds (W1*proj_TM) ⊙ (W2*AE_trunk) with a zero-init scalar
        # so t=0 == xfusion_026. Targets the +0.173 spatial-bonus R^2 found by
        # tools/diagnostic_ae_tm_spatial_alignment.py: per-position interaction
        # has signal beyond what the sigmoid gate (a scalar mixer) can reach.
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated_hada_tessera_gated": {
            "token_fusion_kind": "single",
            "fusion_points": ("decoder64",),
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
            "token_gate_kind": "sigmoid",
            "tessera_trunk_gate": True,
            "token_hadamard_residual": True,
        },
        # Layered TerraMind injection via sigmoid gates at all 3 spatial
        # scales of the AlphaEarth U-Net (32x32 bottleneck, 64x64 decoder,
        # 128x128 decoder). Each level gets its own SigmoidFusionTokenAdapter
        # (independent proj + tied sigmoid mix). Pair with --tessera-presence-ch 0
        # to isolate "AE + multi-level TerraMind gates" without Tessera.
        "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated": {
            "token_fusion_kind": "single",
            "fusion_points": ("bottleneck", "decoder64", "decoder128"),
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
            "token_gate_kind": "sigmoid",
        },
        # Same as above but routes the token grid through TokenChannelStandardize
        # before pyramid generation. Required for token sources whose channel
        # statistics are very far from unit scale (e.g. THOR-S2, whose raw
        # values land in the +-2e4 range and immediately overflow fp16 AMP).
        "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated_norm": {
            "token_fusion_kind": "single",
            "fusion_points": ("bottleneck", "decoder64", "decoder128"),
            "normalize_tokens": True,
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
            "token_gate_kind": "sigmoid",
        },
        # xfusion_032: AE U-Net + DPT-style token backbone + FiLM modulation
        # at all 3 fusion points, normalize_tokens on (works for THOR-S2).
        # No Tessera path. Each AE level is conditioned by the token via
        # FiLM (scale + shift, zero-init) instead of the sigmoid mix used
        # by the *_terramind_gated variants. Token features themselves are
        # produced by TokenDPTBackbone with stage-to-stage RCU propagation
        # (constant 128-ch stage width, scales {32,64,128}), replacing the
        # independent TokenPyramidProvider neck.
        "tessera_token_crosslevel_s2_dpt_film_3way_deep_norm": {
            "token_fusion_kind": "single",
            "fusion_points": ("bottleneck", "decoder64", "decoder128"),
            "normalize_tokens": True,
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
            "token_gate_kind": "film",
            "token_backbone_kind": "dpt",
            "token_stage_ch": 128,
        },
        # Same multi-level TerraMind sigmoid gates as above, but ALSO:
        #   - Tessera trunk-level sigmoid gate at the end of the AE decoder
        #     (TesseraIoUFusionGatedLightUNet style, base_ch wide)
        #   - keep the 16ch Tessera presence residual (compose orthogonally)
        # i.e. xfusion_027 + xfusion_026's Tessera paths. Pair with
        # --tessera-presence-ch 16 to enable the residual.
        "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated_tessera_gated": {
            "token_fusion_kind": "single",
            "fusion_points": ("bottleneck", "decoder64", "decoder128"),
            "presence_head_kind": "split_all",
            "presence_head_depth": 2,
            "presence_branch_ch": 48,
            "token_gate_kind": "sigmoid",
            "tessera_trunk_gate": True,
        },
        "tessera_token_crosslevel_xattn_bottleneck": {
            "token_fusion_kind": "s1s2_xattn",
            "fusion_points": ("bottleneck",),
        },
        "tessera_token_crosslevel_xattn_decoder64": {
            "token_fusion_kind": "s1s2_xattn",
            "fusion_points": ("decoder64",),
        },
        "tessera_token_crosslevel_xattn_bottleneck_decoder64": {
            "token_fusion_kind": "s1s2_xattn",
            "fusion_points": ("bottleneck", "decoder64"),
        },
    }
    if selected in crosslevel_configs:
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                f"{selected} expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        cfg = crosslevel_configs[selected]
        ModelCls = (
            TesseraTokenCrossLevelFusionWaterBypassLightUNet
            if cfg.get("water_bypass") else TesseraTokenCrossLevelFusionLightUNet
        )
        return ModelCls(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            token_fusion_kind=cfg["token_fusion_kind"],
            fusion_points=cfg["fusion_points"],
            normalize_tokens=cfg.get("normalize_tokens", False),
            presence_head_kind=cfg.get("presence_head_kind", "shared"),
            presence_head_depth=cfg.get("presence_head_depth", 1),
            presence_branch_ch=cfg.get("presence_branch_ch"),
            token_gate_kind=cfg.get("token_gate_kind", "scalar"),
            token_backbone_kind=cfg.get("token_backbone_kind", "pyramid"),
            token_stage_ch=cfg.get("token_stage_ch", 128),
            gate_mode=gate_mode,
            gate_untied=gate_untied,
            gate_init_bias=gate_init_bias,
            tessera_trunk_gate=cfg.get("tessera_trunk_gate", False),
            tessera_trunk_hidden_ch=tessera_hidden_ch,
            tessera_trunk_hidden_depth=tessera_hidden_depth,
            modality_dropout=modality_dropout,
            token_hadamard_residual=cfg.get("token_hadamard_residual", False),
        ), selected

    if selected in {"dpt_compact_token_only_3way_deep",
                    "dpt_compact_token_only"}:
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                f"{selected} expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        if selected == "dpt_compact_token_only_3way_deep":
            preset_presence_kind = "split_all"
            preset_presence_depth = 2
            preset_presence_branch_ch = 48
        else:
            preset_presence_kind = "shared"
            preset_presence_depth = 1
            preset_presence_branch_ch = None
        return CompactDPTTokenOnly(
            pixel_channels=pixel_channels,
            token_channels=token_channels,
            n_classes=n_classes,
            stage_ch=128,
            base_ch=lightunet_base_ch,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=preset_presence_kind,
            presence_head_depth=preset_presence_depth,
            presence_branch_ch=preset_presence_branch_ch,
        ), selected

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Use one of: auto, lightunet, lightunet_presence_2plus1, lightunet_presence_3way, "
        "lightunet_presence_shared3, "
        "lightunet_pp, decoder_residual, token_neck, token_neck_norm, "
        "token_fusion_neck, token_fusion_neck_norm, token_fusion_neck_xattn, "
        "token_fusion_neck_xattn_norm, embedding_refiner, hrnet_w18, hrnet_w32, tessera_iou_fusion, "
        "tessera_iou_fusion_unetpp, tessera_iou_fusion_presence_2plus1, "
        "tessera_iou_fusion_presence_3way, tessera_iou_fusion_presence_shared3, "
        "tessera_iou_fusion_gated, tessera_iou_fusion_gated_presence_2plus1, "
        "tessera_iou_fusion_gated_presence_3way, "
        "tessera_iou_fusion_gated_presence_shared3, "
        "tessera_token_shared_probe, tessera_token_fusion_shared_probe, "
        "tessera_token_fusion_shared_probe_norm, tessera_token_height_residual_probe, "
        "tessera_token_xattn_height_residual_probe, tessera_token_s2_nonwater_residual_decoder64, "
        "tessera_token_s2_all_residual_decoder64, tessera_token_s2_water_residual_decoder64, "
        "tessera_token_crosslevel_s2_bottleneck, "
        "tessera_token_crosslevel_s2_decoder64, "
        "tessera_token_crosslevel_s2_decoder64_presence_2plus1, "
        "tessera_token_crosslevel_s2_decoder64_presence_3way, "
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep, "
        "tessera_token_crosslevel_s2_bottleneck_decoder64_presence_3way_deep, "
        "tessera_token_crosslevel_s2_decoder64_decoder128_presence_3way_deep, "
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_water_bypass, "
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated, "
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated_tessera_gated, "
        "tessera_token_crosslevel_s2_decoder64_presence_3way_deep_terramind_gated_hada_tessera_gated, "
        "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated, "
        "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated_norm, "
        "tessera_token_crosslevel_s2_bottleneck_decoder64_decoder128_presence_3way_deep_terramind_gated_tessera_gated, "
        "tessera_token_crosslevel_s2_dpt_film_3way_deep_norm, "
        "dpt_compact_token_only, dpt_compact_token_only_3way_deep, "
        "tessera_token_crosslevel_xattn_bottleneck, "
        "tessera_token_crosslevel_xattn_decoder64, tessera_token_crosslevel_xattn_bottleneck_decoder64"
    )
