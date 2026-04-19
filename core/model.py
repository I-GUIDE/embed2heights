import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. LIGHT UNET COMPONENTS
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleBlock(nn.Module):
    """
    Bilinear Upsampling + Convolution.
    Smoother than PixelShuffle/TransposeConv, avoids checkerboard artifacts.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.supports_aux_outputs = True

        # Architecture: Light version (32->64->128->256)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = UpsampleBlock(256, 128)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = UpsampleBlock(128, 64)
        self.conv2 = DoubleConv(128, 64)

        self.up3 = UpsampleBlock(64, 32)
        self.conv3 = DoubleConv(64, 32)

        self.head = MultiTaskPredictionHead(in_ch=32, out_channels=n_classes)

    def forward(self, x, return_aux=False):
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

    def __init__(self, in_ch, out_channels=4, hidden_ch=None, drop=0.05):
        super().__init__()
        if out_channels != 4:
            raise ValueError("MultiTaskPredictionHead assumes 4 output channels")
        hidden_ch = hidden_ch or min(160, max(64, in_ch // 2))
        self._hidden_ch = hidden_ch

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

        # --- FiLM conditioning: soft fractions modulate height features ---
        self.film_scale = nn.Conv2d(3, hidden_ch, 1)
        self.film_shift = nn.Conv2d(3, hidden_ch, 1)

        # --- Shared height trunk + 3 lightweight output projections ---
        self.height_trunk = nn.Sequential(
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
            ConvGNAct(hidden_ch, hidden_ch, kernel_size=3),
        )
        self.height_base_proj = nn.Conv2d(hidden_ch, 1, 1)
        self.height_building_delta_proj = nn.Conv2d(hidden_ch, 1, 1)
        self.height_vegetation_delta_proj = nn.Conv2d(hidden_ch, 1, 1)

    def forward(self, x, return_aux=False):
        # Shared trunk with residual
        x = self.shared(x)
        x = self.shared_act(x + self.shared_res(x))

        # Auxiliary soft fraction (for height gating + regression losses)
        fraction_logits = self.fraction_head(x)
        fractions = torch.sigmoid(fraction_logits)

        # Main presence classifier (submission channels 0-2)
        presence_logits = self.presence_head(x)
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
        p_b = presence_prob[:, 0:1, :, :]
        p_v = presence_prob[:, 1:2, :, :]
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
            "height_base": base_height,
            "height_building": building_height,
            "height_vegetation": vegetation_height,
        }


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


def build_model(model_type, n_channels, n_classes):
    selected = model_type.lower()

    if selected == "auto":
        selected = infer_model_type(n_channels)
    if selected == "lightunet":
        return LightUNet(n_channels, n_classes), selected
    if selected == "decoder":
        selected = "decoder_residual"
    if selected == "decoder_residual":
        return EfficientDecoder256Fast(in_channels=n_channels, out_channels=n_classes), selected
    if selected == "embedding_refiner":
        return EmbeddingRefiner(n_channels=n_channels, n_classes=n_classes), selected
    if selected == "hrnet_w18":
        return HRNetEmbedding(n_channels=n_channels, n_classes=n_classes, width=18), selected
    if selected == "hrnet_w32":
        return HRNetEmbedding(n_channels=n_channels, n_classes=n_classes, width=32), selected

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Use one of: auto, lightunet, decoder_residual, embedding_refiner, hrnet_w18, hrnet_w32"
    )
