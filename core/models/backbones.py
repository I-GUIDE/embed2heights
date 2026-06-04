import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import MultiTaskPredictionHead


UPSAMPLE_KINDS = ("bilinear", "pixelshuffle", "carafe", "dysample")


class CARAFEUpsample(nn.Module):
    """Content-Aware ReAssembly of FEatures (CARAFE, ICCV'19), scale_factor=2.

    Predicts a per-location k x k reassembly kernel from a compressed feature,
    then reassembles the input by a content-aware weighted average over each
    pixel's k x k neighborhood. Sharper, content-adaptive edges than bilinear —
    directly targets small/dense building boundaries (and the height
    discontinuities that ride on them) without changing the channel count.
    """

    def __init__(self, channels, scale=2, k_up=5, k_enc=3, compressed=64):
        super().__init__()
        self.scale = int(scale)
        self.k_up = int(k_up)
        compressed = min(compressed, channels)
        self.comp = nn.Conv2d(channels, compressed, 1)
        # Kernel predictor: one (up*k_up)^2 weight per output location.
        self.enc = nn.Conv2d(
            compressed,
            (self.scale * self.k_up) ** 2,
            kernel_size=k_enc,
            padding=k_enc // 2,
        )
        self.shuffle = nn.PixelShuffle(self.scale)

    def forward(self, x):
        b, c, h, w = x.shape
        # Predict normalized reassembly kernels at the upsampled resolution.
        kernels = self.enc(self.comp(x))           # (B, (s*k)^2, H, W)
        kernels = self.shuffle(kernels)            # (B, k^2, sH, sW)
        kernels = F.softmax(kernels, dim=1)

        # Unfold the input into k_up x k_up neighborhoods, upsample to target.
        x_unfold = F.unfold(x, kernel_size=self.k_up, padding=self.k_up // 2)
        x_unfold = x_unfold.view(b, c, self.k_up * self.k_up, h, w)
        x_unfold = F.interpolate(
            x_unfold.view(b, c * self.k_up * self.k_up, h, w),
            scale_factor=self.scale, mode="nearest",
        ).view(b, c, self.k_up * self.k_up, h * self.scale, w * self.scale)

        out = (x_unfold * kernels.unsqueeze(1)).sum(dim=2)
        return out


class DySampleUpsample(nn.Module):
    """DySample (ICCV'23) point-sampling dynamic upsampler, scale_factor=2.

    Learns per-location sampling offsets and grid-samples the input — almost
    parameter-free and a stronger boundary reconstructor than bilinear. Uses
    the "lp" (linear + pixelshuffle) style generator with offsets initialized
    near zero so it starts close to nearest/bilinear behavior.
    """

    def __init__(self, channels, scale=2, groups=4):
        super().__init__()
        self.scale = int(scale)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.groups = groups
        self.offset = nn.Conv2d(channels, 2 * groups * self.scale ** 2, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.shuffle = nn.PixelShuffle(self.scale)
        # Static bilinear-init base grid offset spread (per DySample paper).
        self.register_buffer(
            "init_pos", self._init_pos(), persistent=False
        )

    def _init_pos(self):
        h = (torch.arange(self.scale) - (self.scale - 1) / 2.0) / self.scale
        grid = torch.stack(torch.meshgrid(h, h, indexing="ij"), dim=-1)
        return grid.reshape(1, -1, 1, 1).flip(1)  # (1, 2*s^2, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.scale
        offset = self.offset(x)                                  # (B, 2*g*s^2, H, W)
        offset = offset.view(b, self.groups, 2 * s * s, h, w)
        offset = offset + self.init_pos.view(1, 1, 2 * s * s, 1, 1)
        offset = offset.reshape(b * self.groups, 2 * s * s, h, w)
        offset = self.shuffle(offset)                            # (B*g, 2, sH, sW)

        # Build sampling grid in normalized [-1, 1] coords.
        oh, ow = h * s, w * s
        base_y = torch.linspace(-1, 1, oh, device=x.device)
        base_x = torch.linspace(-1, 1, ow, device=x.device)
        gy, gx = torch.meshgrid(base_y, base_x, indexing="ij")
        base = torch.stack((gx, gy), dim=0).unsqueeze(0)         # (1, 2, oH, oW)
        norm = torch.tensor([2.0 / max(ow - 1, 1), 2.0 / max(oh - 1, 1)],
                            device=x.device).view(1, 2, 1, 1)
        grid = base + offset * norm                              # (B*g, 2, oH, oW)
        grid = grid.permute(0, 2, 3, 1)

        xg = x.view(b * self.groups, c // self.groups, h, w)
        out = F.grid_sample(xg, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
        return out.view(b, c, oh, ow)


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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _light_norm(out_channels, norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _light_norm(out_channels, norm_kind),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleBlock(nn.Module):
    """2x upsample + 3x3 conv. The upsampler is selectable via `upsample_kind`.

    - "bilinear"     : nn.Upsample(bilinear) — legacy default, unchanged.
    - "pixelshuffle" : learned sub-pixel conv (1x1 -> PixelShuffle).
    - "carafe"       : content-aware reassembly (CARAFE).
    - "dysample"     : dynamic point sampling (DySample).

    Non-bilinear kinds reconstruct sharper boundaries, which lifts building IoU
    and the height RMSE at building/vegetation edges. Default stays "bilinear"
    so existing checkpoints/configs reproduce bit-for-bit.
    """

    def __init__(self, in_channels, out_channels, norm_kind="bn",
                 upsample_kind="bilinear"):
        super().__init__()
        upsample_kind = (upsample_kind or "bilinear").lower()
        if upsample_kind not in UPSAMPLE_KINDS:
            raise ValueError(
                f"Unknown upsample_kind={upsample_kind!r}; expected one of {UPSAMPLE_KINDS}."
            )
        self.upsample_kind = upsample_kind
        if upsample_kind == "bilinear":
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        elif upsample_kind == "pixelshuffle":
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=1),
                nn.PixelShuffle(2),
            )
        elif upsample_kind == "carafe":
            self.upsample = CARAFEUpsample(in_channels, scale=2)
        else:  # dysample
            self.upsample = DySampleUpsample(in_channels, scale=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = _light_norm(out_channels, norm_kind)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_ch=32, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, height_norm_stats=None,
                 upsample_kind="bilinear"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.upsample_kind = upsample_kind
        self.supports_aux_outputs = True

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.inc = DoubleConv(n_channels, c1, norm_kind=norm_kind)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, norm_kind=norm_kind))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, norm_kind=norm_kind))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4, norm_kind=norm_kind))

        self.up1 = UpsampleBlock(c4, c3, norm_kind=norm_kind, upsample_kind=upsample_kind)
        self.conv1 = DoubleConv(c4, c3, norm_kind=norm_kind)

        self.up2 = UpsampleBlock(c3, c2, norm_kind=norm_kind, upsample_kind=upsample_kind)
        self.conv2 = DoubleConv(c3, c2, norm_kind=norm_kind)

        self.up3 = UpsampleBlock(c2, c1, norm_kind=norm_kind, upsample_kind=upsample_kind)
        self.conv3 = DoubleConv(c2, c1, norm_kind=norm_kind)

        self.head = MultiTaskPredictionHead(
            in_ch=c1,
            out_channels=n_classes,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            height_norm_stats=height_norm_stats,
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
