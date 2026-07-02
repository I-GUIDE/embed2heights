import torch
import torch.nn as nn
import torch.nn.functional as F


# --- shared nn primitives (used by backbones, heads, and the fusion model) ---

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


class TransformerBottleneck(nn.Module):
    """TransUNet-style global self-attention at the encoder bottleneck.

    Flattens the lowest-resolution feature (B,C,H,W -> B,HW,C), adds a learned
    positional embedding, runs a few pre-norm transformer encoder layers, and
    folds the result back in as a ZERO-INIT residual (gamma=0 at start) so the
    whole module is the identity at init -> the model is baseline-identical and
    only LEARNS to add global context (the one thing the CNN lacks; targets the
    height/region generalization the U-Net++ nesting already started)."""

    def __init__(self, channels, n_layers=2, n_heads=6, mlp_ratio=4, max_tokens=1024):
        super().__init__()
        while channels % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, channels))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads, dim_feedforward=channels * mlp_ratio,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.gamma = nn.Parameter(torch.zeros(1))   # zero-init -> identity at start

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)            # B, HW, C
        t = t + self.pos[:, : h * w, :]
        t = self.transformer(t)
        out = t.transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * out


def _light_norm(num_channels, kind="bn"):
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
    def __init__(self, in_channels, out_channels, norm_kind="bn"):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _light_norm(out_channels, norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _light_norm(out_channels, norm_kind),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_kind="bn"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = _light_norm(out_channels, norm_kind)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class LightUNet(nn.Module):
    def __init__(self, n_channels, base_ch=32, norm_kind="bn"):
        super().__init__()
        self.n_channels = n_channels
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

    def forward(self, x):
        return self.forward_features(x)


class LightUNet3Plus(nn.Module):
    """UNet 3+ (full-scale skip connections) variant — drop-in for LightUNet.

    Same 4-level encoder (X^i_0, i=0..3) as LightUNet/LightUNetPP. The decoder
    differs: instead of U-Net++'s dense *nested* grid (progressive same/adjacent
    -scale refinement), EVERY decoder node aggregates ALL scales at once —
    finer-resolution encoder maps (max-pooled down), the same-scale encoder map,
    and coarser decoder maps (bilinearly up). Each scale is projected to a unified
    `cat = base_ch` channels and the 4 are concatenated + fused, so the node
    output stays `base_ch` -> `forward_features(x)` returns a base_ch map exactly
    like the other backbones (true drop-in; the hybrid model's own head consumes
    it). NO deep supervision and NO classification-guided module (both off: DS was
    refuted here, CGM is an organ-segmentation FP trick irrelevant to us) — this
    isolates the pure full-scale-skip topology vs U-Net++'s nested topology.
    """

    def __init__(self, n_channels, base_ch=32, norm_kind="bn"):
        super().__init__()
        self.n_channels = n_channels
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.supports_aux_outputs = True

        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        cat = base_ch                      # UNet3+ unified per-scale channel width

        # Encoder (column 0), shared downsampler factory (maxpool default).
        self.x0_0 = DoubleConv(n_channels, c[0], norm_kind=norm_kind)
        self.x1_0 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c[0], c[1], norm_kind=norm_kind))
        self.x2_0 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c[1], c[2], norm_kind=norm_kind))
        self.x3_0 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c[2], c[3], norm_kind=norm_kind))

        def cu(cin):                       # single 3x3 conv->norm->relu, project to cat
            return nn.Sequential(
                nn.Conv2d(cin, cat, kernel_size=3, padding=1),
                _light_norm(cat, norm_kind), nn.ReLU(inplace=True),
            )

        # Decoder node d3 (1/4 res): e0(full,/4) e1(1/2,/2) e2(same) e3(coarser,x2)
        self.d3_from_e0 = cu(c[0]); self.d3_from_e1 = cu(c[1])
        self.d3_from_e2 = cu(c[2]); self.d3_from_e3 = cu(c[3])
        self.d3_fuse = DoubleConv(4 * cat, cat, norm_kind=norm_kind)
        # Decoder node d2 (1/2 res): e0(/2) e1(same) d3(x2) e3(coarsest,x4)
        self.d2_from_e0 = cu(c[0]); self.d2_from_e1 = cu(c[1])
        self.d2_from_d3 = cu(cat);  self.d2_from_e3 = cu(c[3])
        self.d2_fuse = DoubleConv(4 * cat, cat, norm_kind=norm_kind)
        # Decoder node d1 (full res): e0(same) d2(x2) d3(x4) e3(x8)
        self.d1_from_e0 = cu(c[0]); self.d1_from_d2 = cu(cat)
        self.d1_from_d3 = cu(cat);  self.d1_from_e3 = cu(c[3])
        self.d1_fuse = DoubleConv(4 * cat, cat, norm_kind=norm_kind)

    @staticmethod
    def _down(x, k):
        return F.max_pool2d(x, kernel_size=k, stride=k)

    @staticmethod
    def _up(x, k):
        return F.interpolate(x, scale_factor=k, mode="bilinear", align_corners=True)

    def forward_features(self, x):
        e0 = self.x0_0(x)                  # full res, c0
        e1 = self.x1_0(e0)                 # 1/2,     c1
        e2 = self.x2_0(e1)                 # 1/4,     c2
        e3 = self.x3_0(e2)                 # 1/8,     c3 (bottleneck)

        d3 = self.d3_fuse(torch.cat([
            self.d3_from_e0(self._down(e0, 4)),
            self.d3_from_e1(self._down(e1, 2)),
            self.d3_from_e2(e2),
            self.d3_from_e3(self._up(e3, 2)),
        ], dim=1))
        d2 = self.d2_fuse(torch.cat([
            self.d2_from_e0(self._down(e0, 2)),
            self.d2_from_e1(e1),
            self.d2_from_d3(self._up(d3, 2)),
            self.d2_from_e3(self._up(e3, 4)),
        ], dim=1))
        d1 = self.d1_fuse(torch.cat([
            self.d1_from_e0(e0),
            self.d1_from_d2(self._up(d2, 2)),
            self.d1_from_d3(self._up(d3, 4)),
            self.d1_from_e3(self._up(e3, 8)),
        ], dim=1))
        return d1

    def forward(self, x):
        return self.forward_features(x)


class LightUNetPP(nn.Module):
    """U-Net++ (nested) variant of LightUNet.

    Same 4-level encoder (X^i_0 for i=0..3) and the same final-feature channel
    width (base_ch) as LightUNet, so it is a drop-in replacement for any
    backbone that consumes `forward_features(x)`. Decoder is densely nested:
    every node X^i_j with j>0 fuses (j previous same-level outputs) with an
    upsampled X^(i+1)_(j-1). The forward feature is X^0_3.
    """

    def __init__(self, n_channels, base_ch=32, norm_kind="bn",
                 bottleneck_attn=False,
                 bottleneck_attn_layers=2, bottleneck_attn_heads=6):
        super().__init__()
        self.n_channels = n_channels
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.supports_aux_outputs = True
        # TransUNet bottleneck: global self-attention on X^3_0 (lowest res).
        self.bottleneck_attn = bool(bottleneck_attn)

        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # Column 0 = encoder (X^i_0).
        self.x0_0 = DoubleConv(n_channels, c[0], norm_kind=norm_kind)
        self.x1_0 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c[0], c[1], norm_kind=norm_kind))
        self.x2_0 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c[1], c[2], norm_kind=norm_kind))
        self.x3_0 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c[2], c[3], norm_kind=norm_kind))

        # Up-projections feeding each X^i_j (j>=1): X^(i+1)_(j-1) -> c[i].
        self.up0_1 = UpsampleBlock(c[1], c[0], norm_kind=norm_kind)
        self.up1_1 = UpsampleBlock(c[2], c[1], norm_kind=norm_kind)
        self.up2_1 = UpsampleBlock(c[3], c[2], norm_kind=norm_kind)
        self.up0_2 = UpsampleBlock(c[1], c[0], norm_kind=norm_kind)
        self.up1_2 = UpsampleBlock(c[2], c[1], norm_kind=norm_kind)
        self.up0_3 = UpsampleBlock(c[1], c[0], norm_kind=norm_kind)

        # Nested decoder convs. Input ch for X^i_j = (j+1)*c[i].
        self.x0_1 = DoubleConv(2 * c[0], c[0], norm_kind=norm_kind)
        self.x1_1 = DoubleConv(2 * c[1], c[1], norm_kind=norm_kind)
        self.x2_1 = DoubleConv(2 * c[2], c[2], norm_kind=norm_kind)
        self.x0_2 = DoubleConv(3 * c[0], c[0], norm_kind=norm_kind)
        self.x1_2 = DoubleConv(3 * c[1], c[1], norm_kind=norm_kind)
        self.x0_3 = DoubleConv(4 * c[0], c[0], norm_kind=norm_kind)

        if self.bottleneck_attn:
            self.bottleneck = TransformerBottleneck(
                c[3], n_layers=int(bottleneck_attn_layers),
                n_heads=int(bottleneck_attn_heads),
            )

    def forward_features(self, x):
        x00 = self.x0_0(x)
        x10 = self.x1_0(x00)
        x20 = self.x2_0(x10)
        x30 = self.x3_0(x20)
        if self.bottleneck_attn:
            x30 = self.bottleneck(x30)              # global self-attention bottleneck

        x01 = self.x0_1(torch.cat([x00, self.up0_1(x10)], dim=1))
        x11 = self.x1_1(torch.cat([x10, self.up1_1(x20)], dim=1))
        x21 = self.x2_1(torch.cat([x20, self.up2_1(x30)], dim=1))

        x02 = self.x0_2(torch.cat([x00, x01, self.up0_2(x11)], dim=1))
        x12 = self.x1_2(torch.cat([x10, x11, self.up1_2(x21)], dim=1))

        x03 = self.x0_3(torch.cat([x00, x01, x02, self.up0_3(x12)], dim=1))
        return x03

    def forward(self, x):
        return self.forward_features(x)


def build_pixel_backbone(kind, in_channels, base_ch, norm_kind,
                         bottleneck_attn=False):
    """Pick the pixel backbone variant: 'unet' (LightUNet), 'unetpp' (LightUNetPP),
    or 'unet3plus' (LightUNet3Plus)."""
    k = (kind or "unet").lower()
    if k in ("unet", "lightunet"):
        return LightUNet(in_channels, base_ch=base_ch, norm_kind=norm_kind)
    if k in ("unetpp", "unet++", "lightunetpp"):
        return LightUNetPP(in_channels, base_ch=base_ch, norm_kind=norm_kind,
                           bottleneck_attn=bottleneck_attn)
    if k in ("unet3plus", "unet3+", "unet+++", "lightunet3plus"):
        # Full-scale skip connections (no DS, no CGM); pure topology vs U-Net++.
        return LightUNet3Plus(in_channels, base_ch=base_ch, norm_kind=norm_kind)
    raise ValueError(
        f"Unknown pixel_backbone_kind={kind!r}; expected 'unet', 'unetpp', or 'unet3plus'."
    )
