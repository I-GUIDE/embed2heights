import torch
import torch.nn as nn

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


class SEBlock(nn.Module):
    """Squeeze-Excitation channel attention. Lightweight, well-proven on segmentation."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CoordAttention(nn.Module):
    """Coordinate Attention (Hou et al. 2021, CVPR).

    Decomposes the 2D pool into separate H and W pools, then mixes them through
    a shared 1x1 conv before splitting back into per-axis attention maps. Unlike
    SE, this encodes positional information into channel attention — relevant
    here because regions (KE etc.) are spatially correlated within tiles.
    """
    def __init__(self, channels, reduction=32, min_hidden=8):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(min_hidden, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2).contiguous()
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2).contiguous()
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_kind="bn", use_se=False, use_coord_attn=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _light_norm(out_channels, norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _light_norm(out_channels, norm_kind),
            nn.ReLU(inplace=True),
        ]
        if use_se:
            layers.append(SEBlock(out_channels))
        if use_coord_attn:
            layers.append(CoordAttention(out_channels))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_kind="bn"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
                 presence_branch_ch=None, use_se=False, use_coord_attn=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.use_se = bool(use_se)
        self.use_coord_attn = bool(use_coord_attn)
        self.supports_aux_outputs = True

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        dc_kw = dict(norm_kind=norm_kind, use_se=self.use_se, use_coord_attn=self.use_coord_attn)
        self.inc = DoubleConv(n_channels, c1, **dc_kw)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, **dc_kw))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, **dc_kw))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4, **dc_kw))

        self.up1 = UpsampleBlock(c4, c3, norm_kind=norm_kind)
        self.conv1 = DoubleConv(c4, c3, **dc_kw)

        self.up2 = UpsampleBlock(c3, c2, norm_kind=norm_kind)
        self.conv2 = DoubleConv(c3, c2, **dc_kw)

        self.up3 = UpsampleBlock(c2, c1, norm_kind=norm_kind)
        self.conv3 = DoubleConv(c2, c1, **dc_kw)

        self.head = MultiTaskPredictionHead(
            in_ch=c1,
            out_channels=n_classes,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
        )

    def forward_encoder(self, x):
        """Returns (bottleneck, skip_connections) for external bottleneck fusion."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4, (x1, x2, x3)

    def forward_decoder(self, x4, skips):
        """Runs decoder from bottleneck + skip connections."""
        x1, x2, x3 = skips
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

    def forward_features(self, x):
        x4, skips = self.forward_encoder(x)
        return self.forward_decoder(x4, skips)

    def forward(self, x, return_aux=False):
        x = self.forward_features(x)
        return self.head(x, return_aux=return_aux)
