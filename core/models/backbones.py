import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import MultiTaskPredictionHead


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
                 presence_branch_ch=None, down_kind="maxpool"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.down_kind = down_kind
        self.supports_aux_outputs = True

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

    def forward_features(self, x):
        x00 = self.x0_0(x)
        x10 = self.x1_0(x00)
        x20 = self.x2_0(x10)
        x30 = self.x3_0(x20)

        x01 = self.x0_1(torch.cat([x00, self.up0_1(x10)], dim=1))
        x11 = self.x1_1(torch.cat([x10, self.up1_1(x20)], dim=1))
        x21 = self.x2_1(torch.cat([x20, self.up2_1(x30)], dim=1))

        x02 = self.x0_2(torch.cat([x00, x01, self.up0_2(x11)], dim=1))
        x12 = self.x1_2(torch.cat([x10, x11, self.up1_2(x21)], dim=1))

        x03 = self.x0_3(torch.cat([x00, x01, x02, self.up0_3(x12)], dim=1))
        return x03

    def forward(self, x, return_aux=False):
        x = self.forward_features(x)
        return self.head(x, return_aux=return_aux)
