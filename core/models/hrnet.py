"""HRNet-style high-resolution backbone (Stage E).

A compact HRNet (HRNetV2 design, ~W18-small) used as a drop-in replacement for
LightUNet's feature extractor: same `forward_features(x) -> (B, base_ch, H, W)`
contract, so it plugs into the existing gated-fusion model and the
MultiTaskPredictionHead (and thus composes with Stages C/B/D unchanged).

Why HRNet here: unlike the U-Net encoder-decoder that funnels through a
1/8-resolution bottleneck, HRNet keeps a high-resolution branch alive through
the whole network and repeatedly fuses it with lower-resolution branches. That
preserves the fine spatial detail that small, dense building footprints need —
directly targeting building IoU (Stage E).

This is intentionally a *small* HRNet (one module per stage, BasicBlock depth
2, configurable width) to stay trainable at 256x256 within the project's GPU
budget. Stem downsamples to 1/4; the final fused high-res feature is upsampled
back to full resolution with the selectable upsampler (Stage A: carafe /
dysample / pixelshuffle / bilinear).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import UpsampleBlock
from .blocks import ConvGNAct, _group_count


class BasicBlock(nn.Module):
    """Residual 3x3 x2 block with GroupNorm (stateless, AMP-friendly)."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_group_count(ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_group_count(ch), ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + r)


class HRFusion(nn.Module):
    """Multi-resolution fusion: every output branch is the sum of resized
    contributions from all input branches (the core HRNet exchange unit).
    """

    def __init__(self, channels):
        super().__init__()
        self.n = len(channels)
        self.channels = channels
        # cells[i][j] maps input branch j -> output branch i resolution/width.
        self.cells = nn.ModuleList()
        for i in range(self.n):
            row = nn.ModuleList()
            for j in range(self.n):
                if j == i:
                    row.append(nn.Identity())
                elif j > i:
                    # lower-res, more channels -> 1x1 to c[i], upsample in forward
                    row.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.GroupNorm(_group_count(channels[i]), channels[i]),
                    ))
                else:
                    # higher-res, fewer channels -> (i-j) stride-2 3x3 convs
                    seq = []
                    cin = channels[j]
                    for k in range(i - j):
                        cout = channels[i] if k == i - j - 1 else cin
                        seq.append(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False))
                        seq.append(nn.GroupNorm(_group_count(cout), cout))
                        if k < i - j - 1:
                            seq.append(nn.ReLU(inplace=True))
                        cin = cout
                    row.append(nn.Sequential(*seq))
            self.cells.append(row)
        self.act = nn.ReLU(inplace=True)

    def forward(self, xs):
        outs = []
        for i in range(self.n):
            acc = None
            for j in range(self.n):
                y = self.cells[i][j](xs[j])
                if j > i:
                    y = F.interpolate(y, size=xs[i].shape[-2:],
                                      mode="bilinear", align_corners=False)
                acc = y if acc is None else acc + y
            outs.append(self.act(acc))
        return outs


class HRStage(nn.Module):
    """One HRNet stage: per-branch BasicBlocks followed by a fusion."""

    def __init__(self, channels, depth=2):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(*[BasicBlock(c) for _ in range(depth)]) for c in channels
        )
        self.fuse = HRFusion(channels)

    def forward(self, xs):
        xs = [b(x) for b, x in zip(self.branches, xs)]
        return self.fuse(xs)


class HRNetBackbone(nn.Module):
    """Compact HRNet feature extractor. forward_features -> (B, base_ch, H, W).

    Interface mirrors LightUNet so it is a drop-in in the gated-fusion model:
    exposes `n_channels`, `base_ch`, `forward_features`, and a settable `head`.
    """

    def __init__(self, n_channels, n_classes=4, base_ch=48, width=18,
                 upsample_kind="bilinear", norm_kind="gn", stage_depth=2,
                 **unused):
        super().__init__()
        self.n_channels = n_channels
        self.base_ch = base_ch
        self.width = width
        self.supports_aux_outputs = True
        self.head = nn.Identity()  # placeholder; the fusion model owns the head

        c = [width, width * 2, width * 4, width * 8]

        # Stem: two stride-2 convs -> 1/4 resolution.
        self.stem = nn.Sequential(
            ConvGNAct(n_channels, width, kernel_size=3, stride=2),
            ConvGNAct(width, width, kernel_size=3, stride=2),
        )
        self.layer1 = nn.Sequential(BasicBlock(width), BasicBlock(width))

        # Transitions create each new lower-resolution branch from the last one.
        self.trans2 = nn.Sequential(
            nn.Conv2d(c[0], c[1], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(c[1]), c[1]), nn.ReLU(inplace=True),
        )
        self.stage2 = HRStage(c[:2], depth=stage_depth)
        self.trans3 = nn.Sequential(
            nn.Conv2d(c[1], c[2], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(c[2]), c[2]), nn.ReLU(inplace=True),
        )
        self.stage3 = HRStage(c[:3], depth=stage_depth)
        self.trans4 = nn.Sequential(
            nn.Conv2d(c[2], c[3], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(c[3]), c[3]), nn.ReLU(inplace=True),
        )
        self.stage4 = HRStage(c, depth=stage_depth)

        # Fuse all branches onto the highest-res (1/4) branch, project to base_ch.
        fused_ch = sum(c)
        self.fuse_head = ConvGNAct(fused_ch, base_ch, kernel_size=1)
        # Two x2 upsamples (1/4 -> 1/2 -> 1/1) with the selectable upsampler.
        self.up1 = UpsampleBlock(base_ch, base_ch, norm_kind=norm_kind,
                                 upsample_kind=upsample_kind)
        self.up2 = UpsampleBlock(base_ch, base_ch, norm_kind=norm_kind,
                                 upsample_kind=upsample_kind)
        self.refine = ConvGNAct(base_ch, base_ch, kernel_size=3)

    def forward_features(self, x):
        h, w = x.shape[-2:]
        x = self.stem(x)
        x = self.layer1(x)

        b1 = x
        xs = [b1, self.trans2(b1)]
        xs = self.stage2(xs)
        xs = xs + [self.trans3(xs[-1])]
        xs = self.stage3(xs)
        xs = xs + [self.trans4(xs[-1])]
        xs = self.stage4(xs)

        # Upsample all branches to the highest-res branch and concatenate.
        top = xs[0].shape[-2:]
        feats = [xs[0]] + [
            F.interpolate(b, size=top, mode="bilinear", align_corners=False)
            for b in xs[1:]
        ]
        y = self.fuse_head(torch.cat(feats, dim=1))
        y = self.up1(y)
        y = self.up2(y)
        # Guard against odd sizes: match exact input resolution.
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return self.refine(y)

    def forward(self, x, return_aux=False):
        feats = self.forward_features(x)
        return self.head(feats, return_aux=return_aux)
