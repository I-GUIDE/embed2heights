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


class MixStyle(nn.Module):
    """MixStyle: Zhou et al. 2021. Mix feature statistics across the batch.

    During training, computes per-sample (mean, std) of features, then linearly
    interpolates each sample's stats with another sample's stats (drawn from a
    Beta distribution). This creates 'novel' feature distributions, training
    the downstream layers to be robust to style/distribution shifts.

    Inserted after the first DoubleConv. Disabled at eval. Direct attack on
    the OOF→LB domain gap (test set has different per-region statistics).
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-4):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        # Use larger eps to prevent fp16/AMP underflow on low-variance features.
        self.eps = float(eps)
        self._beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x):
        if not self.training:
            return x
        if torch.rand(1).item() > self.p:
            return x
        # Compute mixstyle in fp32 to avoid AMP fp16 numerical issues (variance
        # of low-variance feature maps can underflow to 0 in fp16, blowing up
        # the normalization step).
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x.float()
            b = x_fp32.shape[0]
            mu = x_fp32.mean(dim=[2, 3], keepdim=True)
            var = x_fp32.var(dim=[2, 3], keepdim=True, unbiased=False)
            sig = (var + self.eps).sqrt()
            # Clamp sig to a meaningful minimum even after eps to be safe
            sig = sig.clamp(min=1e-3)
            x_norm = (x_fp32 - mu) / sig
            perm = torch.randperm(b, device=x.device)
            lam = self._beta.sample((b, 1, 1, 1)).to(x.device, dtype=torch.float32)
            mu_mix = lam * mu + (1.0 - lam) * mu[perm]
            sig_mix = lam * sig + (1.0 - lam) * sig[perm]
            out = x_norm * sig_mix + mu_mix
        return out.to(x.dtype)


class BottleneckSelfAttention(nn.Module):
    """Single Transformer block on the UNet bottleneck features (32×32 at 256 input).

    Provides global context that the convolutional receptive field cannot reach
    in 3 downsamples. Lighter than a full ViT trunk: one MSA + MLP, ~0.9-1.5M
    params at base_ch=48 (bottleneck dim 384). Skipped when input is below
    a minimum spatial size to avoid using on tiny patches.
    """
    def __init__(self, channels, n_heads=4, mlp_ratio=1.0, min_size=4):
        super().__init__()
        self.channels = channels
        self.min_size = int(min_size)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        if h < self.min_size or w < self.min_size:
            return x
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        n1 = self.norm1(x_seq)
        attn_out, _ = self.attn(n1, n1, n1, need_weights=False)
        x_seq = x_seq + attn_out
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        return x_seq.transpose(1, 2).reshape(b, c, h, w)


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
                 presence_branch_ch=None, use_se=False, use_coord_attn=False,
                 use_bottleneck_attn=False, use_mixstyle=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.use_se = bool(use_se)
        self.use_coord_attn = bool(use_coord_attn)
        self.use_bottleneck_attn = bool(use_bottleneck_attn)
        self.use_mixstyle = bool(use_mixstyle)
        self.supports_aux_outputs = True

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        dc_kw = dict(norm_kind=norm_kind, use_se=self.use_se, use_coord_attn=self.use_coord_attn)
        self.inc = DoubleConv(n_channels, c1, **dc_kw)
        self.mixstyle = MixStyle() if self.use_mixstyle else None
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, **dc_kw))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, **dc_kw))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4, **dc_kw))
        self.bottleneck_attn = BottleneckSelfAttention(c4) if self.use_bottleneck_attn else None

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
        if self.mixstyle is not None:
            x1 = self.mixstyle(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if self.bottleneck_attn is not None:
            x4 = self.bottleneck_attn(x4)
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
