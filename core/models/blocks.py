import torch
import torch.nn as nn
import torch.nn.functional as F

from core.data.datasets import HEIGHT_NORM_CONSTANT


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
