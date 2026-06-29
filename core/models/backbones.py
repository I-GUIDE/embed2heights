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


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block (MSA + MLP). Building block for stacks."""
    def __init__(self, channels, n_heads=4, mlp_ratio=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x_seq):
        n1 = self.norm1(x_seq)
        attn_out, _ = self.attn(n1, n1, n1, need_weights=False)
        x_seq = x_seq + attn_out
        return x_seq + self.mlp(self.norm2(x_seq))


class BottleneckSelfAttention(nn.Module):
    """Stack of Transformer blocks on the UNet bottleneck features (32×32 at 256 input).

    Provides global context that the convolutional receptive field cannot reach
    in 3 downsamples. depth=1 is the proven botattn config; depth>1 stacks
    more attention to test whether shallow attention was the bottleneck.
    """
    def __init__(self, channels, n_heads=4, mlp_ratio=1.0, min_size=4, depth=1):
        super().__init__()
        self.channels = channels
        self.min_size = int(min_size)
        self.depth = int(depth)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(channels, n_heads=n_heads, mlp_ratio=mlp_ratio)
             for _ in range(self.depth)]
        )

    def forward(self, x):
        b, c, h, w = x.shape
        if h < self.min_size or w < self.min_size:
            return x
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        for blk in self.blocks:
            x_seq = blk(x_seq)
        return x_seq.transpose(1, 2).reshape(b, c, h, w)


class AttentionGate(nn.Module):
    """Attention U-Net skip-connection gate (Oktay et al. 2018).

    g: gating signal from coarser scale (upsampled decoder feature).
    x: skip-connection feature from encoder, same spatial resolution as g.
    Output: x weighted by a sigmoid mask = sigmoid(conv1x1(ReLU(W_g(g) + W_x(x)))).
    Mask is learned, initialized to ~0.5 (psi bias=0), so the gate starts as
    a near-identity copy of the skip and can only sharpen as training proceeds.
    """
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        inter = inter_channels or max(skip_channels // 2, 8)
        self.W_g = nn.Conv2d(gate_channels, inter, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(skip_channels, inter, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter, 1, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.psi.bias)

    def forward(self, g, x):
        a = self.act(self.W_g(g) + self.W_x(x))
        mask = torch.sigmoid(self.psi(a))
        return x * mask


class DoubleConv(nn.Module):
    """Two 3x3 conv blocks with norm + activation. Optional residual skip
    connection (ResU-Net style) and GELU activation (modern, used in
    ConvNeXt/SegFormer). use_modern=True turns both on for the conv block.
    """
    def __init__(self, in_channels, out_channels, norm_kind="bn",
                 use_se=False, use_coord_attn=False, use_modern=False, dilation=1):
        super().__init__()
        self.use_modern = bool(use_modern)
        # dilation>1 grows receptive field without downsampling (atrous). padding
        # = dilation keeps the spatial size identical for a 3x3 kernel.
        d = int(dilation)
        pad = d
        Act = nn.GELU if self.use_modern else (lambda: nn.ReLU(inplace=True))
        if self.use_modern:
            # Pre-norm-style residual block: conv-norm-act-conv-norm + skip → act
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=pad, dilation=d, bias=False),
                _light_norm(out_channels, norm_kind),
                Act(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=pad, dilation=d, bias=False),
                _light_norm(out_channels, norm_kind),
            )
            self.skip = (nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                         if in_channels != out_channels else nn.Identity())
            self.final_act = Act()
            attn_layers = []
            if use_se:
                attn_layers.append(SEBlock(out_channels))
            if use_coord_attn:
                attn_layers.append(CoordAttention(out_channels))
            self.attn = nn.Sequential(*attn_layers) if attn_layers else nn.Identity()
            self.double_conv = None
        else:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=pad, dilation=d, bias=False),
                _light_norm(out_channels, norm_kind),
                Act(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=pad, dilation=d, bias=False),
                _light_norm(out_channels, norm_kind),
                Act(),
            ]
            if use_se:
                layers.append(SEBlock(out_channels))
            if use_coord_attn:
                layers.append(CoordAttention(out_channels))
            self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_modern:
            h = self.conv1(x)
            h = self.conv2(h)
            h = h + self.skip(x)
            return self.attn(self.final_act(h))
        return self.double_conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_kind="bn", use_modern=False,
                 sharp=False):
        super().__init__()
        self.sharp = bool(sharp)
        if self.sharp:
            # Sub-pixel (PixelShuffle) upsampling: a learned 2x upsample that
            # produces sharper class boundaries than bilinear interpolation,
            # which blurs edges. conv -> 4*out_ch, then PixelShuffle(2).
            self.upsample = None
            self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3,
                                  padding=1, bias=False)
            self.shuffle = nn.PixelShuffle(2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.shuffle = None
        self.bn = _light_norm(out_channels, norm_kind)
        self.act = nn.GELU() if use_modern else nn.ReLU(inplace=True)

    def forward(self, x):
        if self.sharp:
            x = self.shuffle(self.conv(x))
        else:
            x = self.conv(self.upsample(x))
        x = self.bn(x)
        return self.act(x)


class _HRNetLite(nn.Module):
    """Compact 2-resolution HRNet-style encoder. A full-res stream is maintained
    the whole way through (so small buildings are never lost to a bottleneck),
    alongside a 1/2-res context stream, with bidirectional fusion between them.
    Kept deliberately small (2 streams, 2 stages) for the ~1619-tile data budget.
    Produces a full-res `c1` feature map for the gated wrapper/head.
    """
    def __init__(self, c1, base_ch, dc_kw):
        super().__init__()
        c2 = base_ch * 2
        self.to_low = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, **dc_kw))
        self.h1 = DoubleConv(c1, c1, **dc_kw)
        self.l1 = DoubleConv(c2, c2, **dc_kw)
        self.h2 = DoubleConv(c1, c1, **dc_kw)
        self.l2 = DoubleConv(c2, c2, **dc_kw)
        self.low_to_high1 = nn.Conv2d(c2, c1, kernel_size=1, bias=False)
        self.high_to_low1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c1, c2, kernel_size=1, bias=False))
        self.low_to_high2 = nn.Conv2d(c2, c1, kernel_size=1, bias=False)
        self.high_to_low2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c1, c2, kernel_size=1, bias=False))
        self.final_low_to_high = nn.Conv2d(c2, c1, kernel_size=1, bias=False)
        self.fuse = DoubleConv(c1 + c1, c1, **dc_kw)

    @staticmethod
    def _up(t, ref):
        return F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, xh):
        xl = self.to_low(xh)
        h = self.h1(xh); l = self.l1(xl)
        h = h + self._up(self.low_to_high1(l), h)
        l = l + self.high_to_low1(h)
        h = self.h2(h); l = self.l2(l)
        h = h + self._up(self.low_to_high2(l), h)
        l = l + self.high_to_low2(h)
        out = self.fuse(torch.cat([h, self._up(self.final_low_to_high(l), h)], dim=1))
        return out


class HaarDownsample(nn.Module):
    """2x downsample via Haar wavelet transform, then a learned 1x1 projection.

    Unlike MaxPool2d (which discards high-frequency detail), the Haar DWT splits
    each channel into LL/LH/HL/HH subbands at half resolution; the 3 high-freq
    bands carry edge/boundary information, which a 1x1 conv mixes back in — so
    boundary info stays "transitive" through the downsample instead of being
    pooled away. Drop-in for nn.MaxPool2d(2): in_ch -> out_ch at H/2, W/2.
    """

    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        # 4 fixed orthonormal Haar 2x2 filters: LL, LH, HL, HH.
        f = 0.5 * torch.tensor([
            [[1., 1.], [1., 1.]],
            [[1., 1.], [-1., -1.]],
            [[1., -1.], [1., -1.]],
            [[1., -1.], [-1., 1.]],
        ], dtype=torch.float32)                       # (4, 2, 2)
        # depthwise (groups=in_ch): each input channel -> its own 4 bands.
        w = f.unsqueeze(1).repeat(in_ch, 1, 1, 1)     # (4*in_ch, 1, 2, 2)
        self.register_buffer("haar", w, persistent=False)
        self.in_ch = int(in_ch)
        self.proj = nn.Conv2d(4 * in_ch, out_ch, kernel_size=1)
        # Safe init: start as plain average-pooling (the LL/low-pass band, scaled
        # to preserve the DC level) with the high-freq bands zeroed, so at init
        # this is a known-good downsample identical-in-spirit to MaxPool. The net
        # then *learns* to mix in the LH/HL/HH boundary detail — same "start as
        # baseline, learn to add" pattern as the zero-init token-fusion/detail gate.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if out_ch == in_ch:
            with torch.no_grad():
                for c in range(in_ch):
                    # band layout per channel: [c*4+0]=LL, +1=LH, +2=HL, +3=HH.
                    # 0.5 * LL = mean of the 2x2 patch (Haar LL filter is 0.5-scaled).
                    self.proj.weight[c, c * 4 + 0, 0, 0] = 0.5

    def forward(self, x):
        d = F.conv2d(x, self.haar, stride=2, groups=self.in_ch)  # (B, 4*in_ch, H/2, W/2)
        return self.proj(d)


class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_ch=32, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, use_se=False, use_coord_attn=False,
                 use_bottleneck_attn=False, use_mixstyle=False,
                 use_attn_gates=False, use_aspp=False, bottleneck_attn_depth=1,
                 use_modern=False, detail_bypass=False, sharp_upsample=False,
                 scene_film=False, encoder_arch="unet"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch
        self.norm_kind = norm_kind
        self.use_se = bool(use_se)
        self.use_coord_attn = bool(use_coord_attn)
        self.use_bottleneck_attn = bool(use_bottleneck_attn)
        self.use_mixstyle = bool(use_mixstyle)
        self.use_attn_gates = bool(use_attn_gates)
        self.use_aspp = bool(use_aspp)
        self.bottleneck_attn_depth = int(bottleneck_attn_depth)
        self.use_modern = bool(use_modern)
        self.detail_bypass = bool(detail_bypass)
        self.sharp_upsample = bool(sharp_upsample)
        self.scene_film = bool(scene_film)
        # encoder_arch: "unet" (default, 3x downsample — unchanged baseline);
        # "shallow" (2 downsample stages, 1/4 bottleneck — keeps more small-object
        # detail); "dilated" (NO downsampling, atrous convs grow receptive field at
        # full res — directly preserves 1-4px buildings the pooling erases);
        # "hrnet" (parallel high-res stream maintained throughout). All produce a
        # full-resolution base_ch feature map, so the gated wrapper/head/Tessera
        # fusion (which operate on the full-res output) are unchanged.
        # encoder_arch may carry a "_wave" suffix (e.g. "unet_wave", "unetpp_wave")
        # → replace MaxPool2d with HaarDownsample (boundary-preserving). The base
        # name is stored in self.encoder_arch so all existing branches/checks are
        # unchanged; only the pooling op differs.
        _ea = str(encoder_arch)
        self.use_wavelet_pool = _ea.endswith("_wave")
        self.encoder_arch = _ea[:-5] if self.use_wavelet_pool else _ea
        self.supports_aux_outputs = True

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        dc_kw = dict(norm_kind=norm_kind, use_se=self.use_se,
                     use_coord_attn=self.use_coord_attn,
                     use_modern=self.use_modern)
        self._dc_kw = dc_kw
        self.inc = DoubleConv(n_channels, c1, **dc_kw)
        self.mixstyle = MixStyle() if self.use_mixstyle else None

        if self.encoder_arch == "unet":
            self.down1 = nn.Sequential(self._make_pool(c1), DoubleConv(c1, c2, **dc_kw))
            self.down2 = nn.Sequential(self._make_pool(c2), DoubleConv(c2, c3, **dc_kw))
            self.down3 = nn.Sequential(self._make_pool(c3), DoubleConv(c3, c4, **dc_kw))
            self.bottleneck_attn = (
                BottleneckSelfAttention(c4, depth=self.bottleneck_attn_depth)
                if self.use_bottleneck_attn else None
            )
            if self.use_aspp:
                from .blocks import ASPP
                self.aspp = ASPP(c4, c4, rates=(1, 6, 12, 18), dropout=0.1)
            else:
                self.aspp = None

            self.up1 = UpsampleBlock(c4, c3, norm_kind=norm_kind, use_modern=self.use_modern,
                                     sharp=self.sharp_upsample)
            self.conv1 = DoubleConv(c4, c3, **dc_kw)

            self.up2 = UpsampleBlock(c3, c2, norm_kind=norm_kind, use_modern=self.use_modern,
                                     sharp=self.sharp_upsample)
            self.conv2 = DoubleConv(c3, c2, **dc_kw)

            self.up3 = UpsampleBlock(c2, c1, norm_kind=norm_kind, use_modern=self.use_modern,
                                     sharp=self.sharp_upsample)
            self.conv3 = DoubleConv(c2, c1, **dc_kw)
        elif self.encoder_arch == "shallow":
            # 2 downsample stages → bottleneck at 1/4 res (c3). Half the pooling
            # of unet, so 1-4px buildings survive one extra stage.
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, **dc_kw))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, **dc_kw))
            self.bottleneck_attn = (
                BottleneckSelfAttention(c3, depth=self.bottleneck_attn_depth)
                if self.use_bottleneck_attn else None
            )
            self.aspp = None
            self.up1 = UpsampleBlock(c3, c2, norm_kind=norm_kind, use_modern=self.use_modern,
                                     sharp=self.sharp_upsample)
            self.conv1 = DoubleConv(c3, c2, **dc_kw)   # cat(up=c2, skip x2=c2)=c3
            self.up2 = UpsampleBlock(c2, c1, norm_kind=norm_kind, use_modern=self.use_modern,
                                     sharp=self.sharp_upsample)
            self.conv2 = DoubleConv(c2, c1, **dc_kw)   # cat(up=c1, skip x1=c1)=c2
        elif self.encoder_arch == "dilated":
            # Output stride 2: a SINGLE 2x downsample, then an atrous stack
            # (dilation 2/4/8) at 1/2 res grows the receptive field WITHOUT
            # further pooling — 4x the bottleneck resolution of the unet's 1/8,
            # so 1-4px buildings survive. Full-res fine detail is recovered by
            # fusing the un-pooled stem features at the end. (Full output-stride-1
            # would be ~5-8x compute and not finish in the walltime.) bottleneck
            # self-attn omitted; the dilated stack supplies the context.
            self.dil_down = nn.MaxPool2d(2)
            self.d1 = DoubleConv(c1, c2, dilation=2, **dc_kw)
            self.d2 = DoubleConv(c2, c3, dilation=4, **dc_kw)
            self.d3 = DoubleConv(c3, c4, dilation=8, **dc_kw)
            self.bottleneck_attn = None
            self.aspp = None
            # decoder = channel-reduction convs at 1/2 res (no upsampling between
            # stages), skip concats at matching resolution.
            self.rconv1 = nn.Conv2d(c4, c3, kernel_size=1, bias=False)
            self.dconv1 = DoubleConv(c3 + c3, c3, **dc_kw)
            self.rconv2 = nn.Conv2d(c3, c2, kernel_size=1, bias=False)
            self.dconv2 = DoubleConv(c2 + c2, c2, **dc_kw)
            self.rconv3 = nn.Conv2d(c2, c1, kernel_size=1, bias=False)
            self.dconv3 = DoubleConv(c1 + c1, c1, **dc_kw)
            # final full-res fuse with the un-pooled stem (fine detail recovery)
            self.dil_fuse = DoubleConv(c1 + c1, c1, **dc_kw)
        elif self.encoder_arch == "hrnet":
            self.bottleneck_attn = None
            self.aspp = None
            self.hrnet = _HRNetLite(c1, base_ch, dc_kw)
        elif self.encoder_arch == "unetpp":
            # UNet++ (nested dense skip connections). Encoder = same 3x downsample
            # path as unet; decoder fills the nested grid X[i][j] (j>0) where each
            # node fuses all shallower same-level nodes + the upsampled deeper
            # node. Output = X[0][3] (full-res, c1). Dense skips refine fine
            # detail → targets small-object/boundary building IoU.
            self.down1 = nn.Sequential(self._make_pool(c1), DoubleConv(c1, c2, **dc_kw))
            self.down2 = nn.Sequential(self._make_pool(c2), DoubleConv(c2, c3, **dc_kw))
            self.down3 = nn.Sequential(self._make_pool(c3), DoubleConv(c3, c4, **dc_kw))
            self.bottleneck_attn = (
                BottleneckSelfAttention(c4, depth=self.bottleneck_attn_depth)
                if self.use_bottleneck_attn else None
            )
            self.aspp = None
            _UB = lambda i, o: UpsampleBlock(i, o, norm_kind=norm_kind,
                                             use_modern=self.use_modern,
                                             sharp=self.sharp_upsample)
            # upsamplers: bring level i+1 node up to level i resolution/channels
            self.uxx_0_1 = _UB(c2, c1); self.uxx_1_1 = _UB(c3, c2); self.uxx_2_1 = _UB(c4, c3)
            self.uxx_0_2 = _UB(c2, c1); self.uxx_1_2 = _UB(c3, c2)
            self.uxx_0_3 = _UB(c2, c1)
            # dense fusion convs: (j+1) same-level inputs (each c_i) -> c_i
            self.cxx_0_1 = DoubleConv(2 * c1, c1, **dc_kw)
            self.cxx_1_1 = DoubleConv(2 * c2, c2, **dc_kw)
            self.cxx_2_1 = DoubleConv(2 * c3, c3, **dc_kw)
            self.cxx_0_2 = DoubleConv(3 * c1, c1, **dc_kw)
            self.cxx_1_2 = DoubleConv(3 * c2, c2, **dc_kw)
            self.cxx_0_3 = DoubleConv(4 * c1, c1, **dc_kw)
        else:
            raise ValueError(f"Unknown encoder_arch={self.encoder_arch!r}")

        # Detail bypass: a full-resolution branch straight from the input that
        # never gets downsampled, added back to the decoder output through a
        # zero-init scalar gate. At init it contributes nothing (model == the
        # proven baseline) and learns to inject fine spatial detail the
        # encoder/bottleneck path loses. Targets boundary-recall leakage.
        if self.detail_bypass:
            _Act = nn.GELU if self.use_modern else (lambda: nn.ReLU(inplace=True))
            self.detail_branch = nn.Sequential(
                nn.Conv2d(n_channels, c1, kernel_size=3, padding=1, bias=False),
                _light_norm(c1, norm_kind), _Act(),
                nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            )
            self.detail_gate = nn.Parameter(torch.zeros(1))
        else:
            self.detail_branch = None

        # Scene-conditioning FiLM: derive a GLOBAL scene/region descriptor from
        # the bottleneck (what kind of place is this?) and use it to FiLM-
        # modulate the full-res decoder output (gamma/beta over channels). This
        # is the "identify the region coarsely, then let the fine branch adapt"
        # idea, self-contained (no region labels needed). The FiLM head is
        # zero-init so gamma=beta=0 at start => exactly the baseline; the model
        # learns to lean on global context only if it helps.
        if self.scene_film:
            _SAct = nn.GELU if self.use_modern else (lambda: nn.ReLU(inplace=True))
            self.scene_descriptor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(c4, c4), _SAct(),
            )
            self.scene_film_head = nn.Linear(c4, 2 * c1)
            nn.init.zeros_(self.scene_film_head.weight)
            nn.init.zeros_(self.scene_film_head.bias)
        else:
            self.scene_descriptor = None

        if self.use_attn_gates:
            self.ag1 = AttentionGate(gate_channels=c3, skip_channels=c3)
            self.ag2 = AttentionGate(gate_channels=c2, skip_channels=c2)
            self.ag3 = AttentionGate(gate_channels=c1, skip_channels=c1)
        else:
            self.ag1 = self.ag2 = self.ag3 = None

        self.head = MultiTaskPredictionHead(
            in_ch=c1,
            out_channels=n_classes,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
        )

    def _make_pool(self, ch):
        """2x downsample op: Haar wavelet (boundary-preserving) if the encoder_arch
        carried a '_wave' suffix, else the baseline MaxPool2d. MaxPool case is
        byte-identical to the previous code."""
        if getattr(self, "use_wavelet_pool", False):
            return HaarDownsample(ch, ch)
        return nn.MaxPool2d(2)

    def _forward_unetpp(self, x):
        x00 = self.inc(x)
        if self.mixstyle is not None:
            x00 = self.mixstyle(x00)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        if self.bottleneck_attn is not None:
            x30 = self.bottleneck_attn(x30)
        x01 = self.cxx_0_1(torch.cat([x00, self.uxx_0_1(x10)], dim=1))
        x11 = self.cxx_1_1(torch.cat([x10, self.uxx_1_1(x20)], dim=1))
        x21 = self.cxx_2_1(torch.cat([x20, self.uxx_2_1(x30)], dim=1))
        x02 = self.cxx_0_2(torch.cat([x00, x01, self.uxx_0_2(x11)], dim=1))
        x12 = self.cxx_1_2(torch.cat([x10, x11, self.uxx_1_2(x21)], dim=1))
        x03 = self.cxx_0_3(torch.cat([x00, x01, x02, self.uxx_0_3(x12)], dim=1))
        return x03

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
        if self.aspp is not None:
            x4 = self.aspp(x4)
        return x4, (x1, x2, x3)

    def forward_decoder(self, x4, skips):
        """Runs decoder from bottleneck + skip connections."""
        x1, x2, x3 = skips
        u = self.up1(x4)
        s = self.ag1(u, x3) if self.ag1 is not None else x3
        x = torch.cat([s, u], dim=1)
        x = self.conv1(x)
        u = self.up2(x)
        s = self.ag2(u, x2) if self.ag2 is not None else x2
        x = torch.cat([s, u], dim=1)
        x = self.conv2(x)
        u = self.up3(x)
        s = self.ag3(u, x1) if self.ag3 is not None else x1
        x = torch.cat([s, u], dim=1)
        x = self.conv3(x)
        return x

    def _forward_shallow(self, x):
        x1 = self.inc(x)
        if self.mixstyle is not None:
            x1 = self.mixstyle(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if self.bottleneck_attn is not None:
            x3 = self.bottleneck_attn(x3)
        u = self.up1(x3)
        x = self.conv1(torch.cat([u, x2], dim=1))
        u = self.up2(x)
        x = self.conv2(torch.cat([u, x1], dim=1))
        return x

    def _forward_dilated(self, x):
        x1 = self.inc(x)
        if self.mixstyle is not None:
            x1 = self.mixstyle(x1)
        xp = self.dil_down(x1)                       # 1/2 res, c1 (single downsample)
        a = self.d1(xp)                              # 1/2, c2
        b = self.d2(a)                               # 1/2, c3
        c = self.d3(b)                               # 1/2, c4
        u = self.rconv1(c)
        u = self.dconv1(torch.cat([u, b], dim=1))    # 1/2, c3
        u = self.rconv2(u)
        u = self.dconv2(torch.cat([u, a], dim=1))    # 1/2, c2
        u = self.rconv3(u)
        u = self.dconv3(torch.cat([u, xp], dim=1))   # 1/2, c1
        u = F.interpolate(u, size=x1.shape[-2:], mode="bilinear", align_corners=True)
        return self.dil_fuse(torch.cat([u, x1], dim=1))  # full res, c1

    def _forward_hrnet(self, x):
        x1 = self.inc(x)
        if self.mixstyle is not None:
            x1 = self.mixstyle(x1)
        return self.hrnet(x1)

    def forward_features(self, x):
        if self.encoder_arch == "shallow":
            return self._forward_shallow(x)
        if self.encoder_arch == "dilated":
            return self._forward_dilated(x)
        if self.encoder_arch == "hrnet":
            return self._forward_hrnet(x)
        if self.encoder_arch == "unetpp":
            return self._forward_unetpp(x)
        x4, skips = self.forward_encoder(x)
        feat = self.forward_decoder(x4, skips)
        if self.detail_branch is not None:
            feat = feat + self.detail_gate * self.detail_branch(x)
        if self.scene_descriptor is not None:
            d = self.scene_descriptor(x4)            # [B, c4]
            g, b = self.scene_film_head(d).chunk(2, dim=1)  # each [B, c1]
            feat = feat * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
        return feat

    def forward(self, x, return_aux=False):
        x = self.forward_features(x)
        return self.head(x, return_aux=return_aux)
