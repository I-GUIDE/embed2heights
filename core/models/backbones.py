import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import MultiTaskPredictionHead


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


class WaveletDownsample(nn.Module):
    """Haar-DWT downsample: a channel-preserving drop-in for ``nn.MaxPool2d(2)``.

    MaxPool halves H,W and throws the high-frequency content away. The Haar DWT
    also halves H,W but decomposes each channel into 4 subbands — LL (local
    average) plus LH/HL/HH (vertical/horizontal/diagonal *edge* detail). We keep
    all four (concat -> 4*C) and project back to C with a 1x1 conv, so the
    encoder can *use* the boundary high-frequencies maxpool would have aliased
    away (the WaveCNet anti-aliasing / shift-stability motivation). The proj is
    initialised to copy LL only (== average pooling), so the network warm-starts
    near the old behaviour and learns to read the edge subbands from there.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.proj = nn.Conv2d(4 * channels, channels, kernel_size=1)
        # Warm start: output = LL subband (≈ average pool); edge subbands off.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            for i in range(channels):
                self.proj.weight[i, i, 0, 0] = 1.0

    @staticmethod
    def _haar(x):
        # Pad odd spatial dims so the 2x2 decimation is exact.
        if x.shape[-1] % 2:
            x = F.pad(x, (0, 1))
        if x.shape[-2] % 2:
            x = F.pad(x, (0, 0, 0, 1))
        xe, xo = x[:, :, 0::2, :], x[:, :, 1::2, :]
        ee, eo = xe[:, :, :, 0::2], xe[:, :, :, 1::2]
        oe, oo = xo[:, :, :, 0::2], xo[:, :, :, 1::2]
        ll = (ee + eo + oe + oo) * 0.5
        lh = (ee + eo - oe - oo) * 0.5
        hl = (ee - eo + oe - oo) * 0.5
        hh = (ee - eo - oe + oo) * 0.5
        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, x):
        return self.proj(self._haar(x))


def _make_downsample(kind, in_ch):
    """'maxpool' (default LightUNet) or 'wavelet' (Haar-DWT, edge-preserving)."""
    if (kind or "maxpool").lower() == "wavelet":
        return WaveletDownsample(in_ch)
    return nn.MaxPool2d(2)


class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_ch=32, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, down_kind="maxpool"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.down_kind = down_kind
        self.supports_aux_outputs = True

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.inc = DoubleConv(n_channels, c1, norm_kind=norm_kind)
        self.down1 = nn.Sequential(_make_downsample(down_kind, c1), DoubleConv(c1, c2, norm_kind=norm_kind))
        self.down2 = nn.Sequential(_make_downsample(down_kind, c2), DoubleConv(c2, c3, norm_kind=norm_kind))
        self.down3 = nn.Sequential(_make_downsample(down_kind, c3), DoubleConv(c3, c4, norm_kind=norm_kind))

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

    def __init__(self, n_channels, n_classes, base_ch=32, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, down_kind="maxpool"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.down_kind = down_kind
        self.supports_aux_outputs = True

        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        cat = base_ch                      # UNet3+ unified per-scale channel width

        # Encoder (column 0), shared downsampler factory (maxpool default).
        self.x0_0 = DoubleConv(n_channels, c[0], norm_kind=norm_kind)
        self.x1_0 = nn.Sequential(_make_downsample(down_kind, c[0]), DoubleConv(c[0], c[1], norm_kind=norm_kind))
        self.x2_0 = nn.Sequential(_make_downsample(down_kind, c[1]), DoubleConv(c[1], c[2], norm_kind=norm_kind))
        self.x3_0 = nn.Sequential(_make_downsample(down_kind, c[2]), DoubleConv(c[2], c[3], norm_kind=norm_kind))

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

        self.head = MultiTaskPredictionHead(
            in_ch=cat,
            out_channels=n_classes,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
        )

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

    def forward(self, x, return_aux=False):
        x = self.forward_features(x)
        return self.head(x, return_aux=return_aux)


class LightUNetPP(nn.Module):
    """U-Net++ (nested) variant of LightUNet.

    Same 4-level encoder (X^i_0 for i=0..3) and the same final-feature channel
    width (base_ch) as LightUNet, so it is a drop-in replacement for any
    backbone that consumes `forward_features(x)`. Decoder is densely nested:
    every node X^i_j with j>0 fuses (j previous same-level outputs) with an
    upsampled X^(i+1)_(j-1). The forward feature is X^0_3.
    """

    def __init__(self, n_channels, n_classes, base_ch=32, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, down_kind="maxpool",
                 deep_supervision=False, bottleneck_attn=False,
                 bottleneck_attn_layers=2, bottleneck_attn_heads=6):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.down_kind = down_kind
        self.supports_aux_outputs = True
        # TransUNet bottleneck: global self-attention on X^3_0 (lowest res).
        self.bottleneck_attn = bool(bottleneck_attn)
        # Deep supervision: lightweight 1x1 heads on the shallower nested decoder
        # nodes X^0_1, X^0_2 emitting (3 presence + 1 height) channels, supervised
        # in the loss to give those nodes direct gradient (U-Net++ paper's
        # generalization mechanism). Training-only; inference uses X^0_3 head.
        self.deep_supervision = bool(deep_supervision)
        self._ds_outputs = None

        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # Column 0 = encoder. The 3 spatial downsamplers use the shared factory
        # so U-Net++ can ALSO take the Haar-DWT edge-preserving pooling (combo
        # with wavelet); down_kind="maxpool" keeps the original behaviour.
        self.x0_0 = DoubleConv(n_channels, c[0], norm_kind=norm_kind)
        self.x1_0 = nn.Sequential(_make_downsample(down_kind, c[0]), DoubleConv(c[0], c[1], norm_kind=norm_kind))
        self.x2_0 = nn.Sequential(_make_downsample(down_kind, c[1]), DoubleConv(c[1], c[2], norm_kind=norm_kind))
        self.x3_0 = nn.Sequential(_make_downsample(down_kind, c[2]), DoubleConv(c[2], c[3], norm_kind=norm_kind))

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

        self.head = MultiTaskPredictionHead(
            in_ch=c[0],
            out_channels=n_classes,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
        )

        if self.deep_supervision:
            # one lightweight head per supervised nested node (X^0_1, X^0_2).
            self.ds_head_1 = nn.Conv2d(c[0], n_classes, 1)
            self.ds_head_2 = nn.Conv2d(c[0], n_classes, 1)

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
        # Stash deep-supervision aux maps (3 presence + 1 height) on the shallow
        # nested nodes for the loss to read; cleared to None when off.
        self._ds_outputs = (
            [self.ds_head_1(x01), self.ds_head_2(x02)]
            if self.deep_supervision else None
        )
        return x03

    def forward(self, x, return_aux=False):
        x = self.forward_features(x)
        return self.head(x, return_aux=return_aux)


class _ResBlock(nn.Module):
    """3x3 residual block, fixed channel count (HRNet basic block)."""

    def __init__(self, ch, norm_kind="bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = _light_norm(ch, norm_kind)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = _light_norm(ch, norm_kind)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class _HRFuse(nn.Module):
    """Multi-resolution exchange unit (HRNet's signature cross-branch fusion).

    Given parallel branch features at channels ``chs`` (branch i runs at
    resolution 1/2**i), every branch is rebuilt as the sum of contributions from
    all branches: higher-res -> lower-res via strided 3x3 convs, lower-res ->
    higher-res via 1x1 conv + bilinear upsample, same-res via identity.
    """

    def __init__(self, chs, norm_kind="bn"):
        super().__init__()
        self.n = len(chs)
        self.fuse = nn.ModuleList()
        for i in range(self.n):
            row = nn.ModuleList()
            for j in range(self.n):
                if j == i:
                    row.append(None)
                elif j > i:
                    # lower-res branch j -> branch i: match channels, upsample in forward.
                    row.append(nn.Sequential(
                        nn.Conv2d(chs[j], chs[i], 1, bias=False),
                        _light_norm(chs[i], norm_kind),
                    ))
                else:
                    # higher-res branch j -> branch i: chain of stride-2 3x3 convs.
                    convs = []
                    cur = chs[j]
                    for k in range(i - j):
                        last = k == (i - j - 1)
                        out_c = chs[i] if last else cur
                        convs.append(nn.Conv2d(cur, out_c, 3, stride=2, padding=1, bias=False))
                        convs.append(_light_norm(out_c, norm_kind))
                        if not last:
                            convs.append(nn.ReLU(inplace=True))
                        cur = out_c
                    row.append(nn.Sequential(*convs))
            self.fuse.append(row)
        self.act = nn.ReLU(inplace=True)

    def forward(self, xs):
        out = []
        for i in range(self.n):
            acc = None
            for j in range(self.n):
                if j == i:
                    contrib = xs[i]
                elif j > i:
                    contrib = self.fuse[i][j](xs[j])
                    contrib = F.interpolate(
                        contrib, size=xs[i].shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                else:
                    contrib = self.fuse[i][j](xs[j])
                acc = contrib if acc is None else acc + contrib
            out.append(self.act(acc))
        return out


class HRNet(nn.Module):
    """Lightweight HRNetV2 backbone, drop-in for LightUNet.

    Maintains ``num_branches`` parallel streams from full resolution down to
    1/2**(num_branches-1), repeatedly exchanging information across resolutions
    instead of an encoder/decoder bottleneck. Branch i has ``base_ch * 2**i``
    channels. ``forward_features`` returns ``base_ch`` channels at the input
    resolution (every branch projected to base_ch, upsampled, concatenated, then
    fused with a 1x1 conv), matching the LightUNet contract so it plugs into the
    fusion models unchanged.
    """

    def __init__(self, n_channels, n_classes, base_ch=32, norm_kind="bn",
                 num_branches=4, blocks_per_branch=2,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.num_branches = num_branches
        self.supports_aux_outputs = True

        chs = [base_ch * (2 ** i) for i in range(num_branches)]
        self.chs = chs

        # Stem: keep full resolution, lift to base_ch (parallels LightUNet.inc).
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, base_ch, 3, padding=1, bias=False),
            _light_norm(base_ch, norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            _light_norm(base_ch, norm_kind),
            nn.ReLU(inplace=True),
        )

        # Each stage spawns one new (lower-res) branch then exchanges across all.
        self.transitions = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()
        self.stage_fuse = nn.ModuleList()
        for s in range(1, num_branches):
            self.transitions.append(nn.Sequential(
                nn.Conv2d(chs[s - 1], chs[s], 3, stride=2, padding=1, bias=False),
                _light_norm(chs[s], norm_kind),
                nn.ReLU(inplace=True),
            ))
            n_active = s + 1
            self.stage_blocks.append(nn.ModuleList([
                nn.Sequential(*[_ResBlock(chs[b], norm_kind) for _ in range(blocks_per_branch)])
                for b in range(n_active)
            ]))
            self.stage_fuse.append(_HRFuse(chs[:n_active], norm_kind))

        # HRNetV2 head: per-branch project -> upsample -> concat -> 1x1 fuse.
        self.branch_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i], base_ch, 1, bias=False),
                _light_norm(base_ch, norm_kind),
                nn.ReLU(inplace=True),
            )
            for i in range(num_branches)
        ])
        self.last_layer = nn.Sequential(
            nn.Conv2d(base_ch * num_branches, base_ch, 1, bias=False),
            _light_norm(base_ch, norm_kind),
            nn.ReLU(inplace=True),
        )

        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
        )

    def forward_features(self, x):
        branches = [self.stem(x)]
        for s in range(self.num_branches - 1):
            branches.append(self.transitions[s](branches[-1]))
            blocks = self.stage_blocks[s]
            branches = [blocks[b](branches[b]) for b in range(len(branches))]
            branches = self.stage_fuse[s](branches)

        size = branches[0].shape[-2:]
        feats = []
        for i, b in enumerate(branches):
            p = self.branch_proj[i](b)
            if i > 0:
                p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            feats.append(p)
        return self.last_layer(torch.cat(feats, dim=1))

    def forward(self, x, return_aux=False):
        x = self.forward_features(x)
        return self.head(x, return_aux=return_aux)
