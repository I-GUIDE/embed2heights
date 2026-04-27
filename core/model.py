import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels):
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


# ==========================================
# 1. LIGHT UNET COMPONENTS
# ==========================================

class DoubleConv(nn.Module):
    """(conv => GroupNorm => GELU) * 2.

    GroupNorm + GELU instead of BatchNorm + ReLU — batch-size-independent
    (important at bs=32), mixed-precision stable (no running stats), and
    matches the GN+GELU convention already used by ConvGNAct / the v35 head.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleBlock(nn.Module):
    """Bilinear upsample + 3x3 conv + GroupNorm + GELU.

    Smoother than PixelShuffle/TransposeConv (avoids checkerboard artifacts);
    GN+GELU matches the rest of the codebase and trains stably in bf16.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_ch=32):
        super(LightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.supports_aux_outputs = True

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.inc = DoubleConv(n_channels, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4))

        self.up1 = UpsampleBlock(c4, c3)
        self.conv1 = DoubleConv(c4, c3)

        self.up2 = UpsampleBlock(c3, c2)
        self.conv2 = DoubleConv(c3, c2)

        self.up3 = UpsampleBlock(c2, c1)
        self.conv3 = DoubleConv(c2, c1)

        self.head = MultiTaskPredictionHead(in_ch=c1, out_channels=n_classes)

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

    def forward(self, x, return_aux=False):
        pyr = self.neck(x)

        x = self.up_16_32(pyr[16])
        x = self.fuse_32(torch.cat([x, pyr[32]], dim=1))

        x = self.up_32_64(x)
        x = self.fuse_64(torch.cat([x, pyr[64]], dim=1))

        x = self.up_64_128(x)
        x = self.fuse_128(torch.cat([x, pyr[128]], dim=1))

        x = self.up_128_256(x)
        x = self.fuse_256(x)

        return self.head(x, return_aux=return_aux)


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
                 presence_extra_ch=0, height_specialist_depth=0):
        super().__init__()
        if out_channels != 4:
            raise ValueError("MultiTaskPredictionHead assumes 4 output channels")
        hidden_ch = hidden_ch or min(160, max(64, in_ch // 2))
        self._hidden_ch = hidden_ch
        self.presence_extra_ch = presence_extra_ch
        self.height_specialist_depth = int(height_specialist_depth)

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
        self.presence_head = nn.Sequential(
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
            nn.Conv2d(hidden_ch, 3, 1),
        )
        if presence_extra_ch > 0:
            self.presence_delta_head = nn.Conv2d(presence_extra_ch, 3, 1)
            nn.init.zeros_(self.presence_delta_head.weight)
            nn.init.zeros_(self.presence_delta_head.bias)
        else:
            self.presence_delta_head = None

        # --- FiLM conditioning: soft fractions modulate height features ---
        self.film_scale = nn.Conv2d(3, hidden_ch, 1)
        self.film_shift = nn.Conv2d(3, hidden_ch, 1)

        # --- Shared height trunk + 3 lightweight output projections ---
        self.height_trunk = nn.Sequential(
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
        )
        def _specialist_head(depth):
            # depth=0 preserves the original single 1x1 projection.
            if depth <= 0:
                return nn.Conv2d(hidden_ch, 1, 1)
            layers = [ConvGNAct(hidden_ch, hidden_ch, kernel_size=3) for _ in range(depth)]
            layers.append(nn.Conv2d(hidden_ch, 1, 1))
            return nn.Sequential(*layers)

        self.height_base_proj = nn.Conv2d(hidden_ch, 1, 1)
        self.height_building_delta_proj = _specialist_head(self.height_specialist_depth)
        self.height_vegetation_delta_proj = _specialist_head(self.height_specialist_depth)

    def forward(self, x, return_aux=False, presence_extra=None):
        # Shared trunk with residual
        x = self.shared(x)
        x = self.shared_act(x + self.shared_res(x))

        # Auxiliary soft fraction (for height gating + regression losses)
        fraction_logits = self.fraction_head(x)
        fractions = torch.sigmoid(fraction_logits)

        # Main presence classifier (submission channels 0-2). When an external
        # edge feature is provided, it learns only a zero-initialized residual
        # logit correction on top of the alpha-only logits.
        alpha_presence_logits = self.presence_head(x)
        if presence_extra is not None:
            if self.presence_delta_head is None:
                raise ValueError("presence_extra was provided but this head has no residual branch")
            presence_delta_logits = self.presence_delta_head(presence_extra)
            presence_logits = alpha_presence_logits + presence_delta_logits
        else:
            presence_delta_logits = None
            presence_logits = alpha_presence_logits
        presence_prob = torch.sigmoid(presence_logits)

        # FiLM conditioning uses soft fractions (fine-grained coverage signal)
        scale = self.film_scale(fractions)
        shift = self.film_shift(fractions)
        h = x * (1.0 + scale) + shift

        # Shared height trunk -> 3 lightweight projections
        h = self.height_trunk(h)
        base_height = F.softplus(self.height_base_proj(h), threshold=20.0)
        # Deltas are non-negative: buildings/vegetation only add height
        building_delta = F.softplus(self.height_building_delta_proj(h), threshold=20.0)
        vegetation_delta = F.softplus(self.height_vegetation_delta_proj(h), threshold=20.0)

        # Absolute class heights (also used as specialists for the submission)
        building_height = base_height + building_delta
        vegetation_height = base_height + vegetation_delta

        # Presence-gated specialist selection for the single submitted height.
        # Rationale: leaderboard's per-class RMSE masks pixels by `gt_class > 0`,
        # which matches the presence head's supervision. `height_building` /
        # `height_vegetation` are L1-trained on their class mask (losses.py),
        # so each is reliable ONLY on that class's pixels. We therefore route
        # each pixel to its relevant specialist by presence, and fall back to
        # `base_height` on background pixels.
        height_presence_prob = torch.sigmoid(alpha_presence_logits)
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
        return {
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
                 base_ch=32):
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
        )

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_stem(tessera)
        return self.head(alpha_feat, return_aux=return_aux, presence_extra=tessera_feat)


class TesseraIoUFusionGatedLightUNet(nn.Module):
    """AlphaEarth + Tessera fused at the **feature level** via a learned gate.

    Distinct from TesseraIoUFusionLightUNet, which adds Tessera only as a
    residual correction on the *presence logits*. Here Tessera is promoted
    to a peer feature stream that fuses into the trunk features just before
    the head.

    Flow:
        AE  ─► alpha_unet.forward_features ─► alpha_feat (B, base_ch, H, W)
        TES ─► tessera_stem (out_ch=base_ch) ─► tessera_feat (B, base_ch, H, W)

        gate = sigmoid(gate_conv(concat(alpha_feat, tessera_feat)))   ∈ (0,1)
        fused = gate * alpha_feat + (1 - gate) * tessera_feat

    Initialization mirrors the existing zero-init residual pattern in this
    codebase (presence_delta_head, height deltas via softplus): the gate's
    final conv is zero-init with a large positive bias so sigmoid(b) ≈ 1
    at step 0 → fused ≈ alpha_feat. Tessera contributes nothing initially
    and learns to seep in as a residual correction. Identical t=0 behavior
    to the AE-only baseline; any divergence is learned from data.

    The presence-logit residual path is *also* preserved (set
    presence_extra_ch>0 to keep it). It is orthogonal to the trunk-level
    fusion and the two compose: trunk fusion improves shared features for
    all heads (incl. height); the presence residual is a lightweight
    correction directly on the IoU output.
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 tessera_presence_ch=0, tessera_hidden_ch=None,
                 tessera_hidden_depth=0, height_specialist_depth=0,
                 base_ch=32, gate_init_bias=4.0):
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
        tessera_channels = n_channels - alpha_channels

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch)
        self.alpha_unet.head = nn.Identity()
        # Tessera stem outputs base_ch (peer width with alpha_feat) for
        # feature-level fusion. Optionally a smaller presence_extra branch
        # is also produced if tessera_presence_ch > 0.
        self.tessera_feature_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=base_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        # Spatial gate: 1x1 conv on concat(alpha, tessera), sigmoid.
        # Zero-init weights + large positive bias → starts as gate≈1
        # (alpha-only) and learns Tessera contribution as a residual.
        self.gate_conv = nn.Conv2d(2 * base_ch, base_ch, kernel_size=1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, gate_init_bias)

        # Optional small presence-logit residual on top (composes with gate).
        self.tessera_presence_ch = int(tessera_presence_ch)
        if self.tessera_presence_ch > 0:
            # Reuse the same stem signature, just project from the (already
            # computed) base_ch feature down to a small width for the
            # presence residual branch.
            self.presence_extra_proj = nn.Conv2d(base_ch, self.tessera_presence_ch, 1)
        else:
            self.presence_extra_proj = None

        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            presence_extra_ch=self.tessera_presence_ch,
            height_specialist_depth=height_specialist_depth,
        )

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)

        gate = torch.sigmoid(self.gate_conv(torch.cat([alpha_feat, tessera_feat], dim=1)))
        fused = gate * alpha_feat + (1.0 - gate) * tessera_feat

        presence_extra = (self.presence_extra_proj(tessera_feat)
                          if self.presence_extra_proj is not None else None)
        return self.head(fused, return_aux=return_aux,
                         presence_extra=presence_extra)


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
                fusion_mode="residual_presence"):
    selected = model_type.lower()

    if selected == "auto":
        selected = infer_model_type(n_channels)
    if selected == "lightunet":
        return LightUNet(n_channels, n_classes, base_ch=lightunet_base_ch), selected
    if selected == "decoder":
        selected = "decoder_residual"
    if selected == "decoder_residual":
        return EfficientDecoder256Fast(in_channels=n_channels, out_channels=n_classes), selected
    if selected == "token_neck":
        return TokenNeckDecoder(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "embedding_refiner":
        return EmbeddingRefiner(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "hrnet_w18":
        return HRNetEmbedding(n_channels=n_channels, n_classes=n_classes, width=18), selected
    if selected == "hrnet_w32":
        return HRNetEmbedding(n_channels=n_channels, n_classes=n_classes, width=32), selected
    if selected == "tessera_iou_fusion":
        if fusion_mode == "gated_feature":
            return TesseraIoUFusionGatedLightUNet(
                n_channels=n_channels,
                n_classes=n_classes,
                tessera_presence_ch=tessera_presence_ch,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
            ), selected
        return TesseraIoUFusionLightUNet(
            n_channels=n_channels,
            n_classes=n_classes,
            tessera_presence_ch=tessera_presence_ch,
            tessera_hidden_ch=tessera_hidden_ch,
            tessera_hidden_depth=tessera_hidden_depth,
            height_specialist_depth=height_specialist_depth,
            base_ch=lightunet_base_ch,
        ), selected

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Use one of: auto, lightunet, decoder_residual, token_neck, embedding_refiner, "
        "hrnet_w18, hrnet_w32, tessera_iou_fusion"
    )
