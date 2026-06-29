import torch
import torch.nn as nn

import torch.nn.functional as F

from .blocks import ASPP, ChannelCalibration, ConvGNAct, ConvNeXtBlock, _group_count
from .backbones import AttentionGate, DoubleConv, LightUNet, UpsampleBlock
from .heads import MultiTaskPredictionHead


def _build_fusion_gate(channels, mode="simple", untied=False, init_bias=4.0):
    """Build the fusion module. Supports several blend formulations:

    - simple/rich: tied (or untied) sigmoid gate; output = g·A + (1-g)·B.
    - concat_mlp: full nonlinear mixing via 2-layer Conv1x1 MLP. No gate.
    - gated_mlp_residual: g·MLP(concat) + (1-g)·A. Strictly more expressive
      than simple because the transform branch can produce features outside
      the convex hull of A,B. Init: g→0 so output starts as A (identity).
    - addmul: g_add·A + (1-g_add)·B + g_mul·(A⊙B). Hadamard term captures
      consensus/disagreement multiplicatively. Mul gate zero-init.
    - film: γ·A + β where γ,β are functions of B; γ = 1 + δγ. Asymmetric:
      B modulates A. δγ, β zero-init so output starts as A.
    """
    if mode == "simple":
        n_out = 2 * channels if untied else channels
        gate = nn.Conv2d(2 * channels, n_out, kernel_size=1)
        nn.init.zeros_(gate.weight)
        if untied:
            bias = torch.empty(n_out)
            bias[:channels].fill_(init_bias)
            bias[channels:].fill_(-init_bias)
            gate.bias.data.copy_(bias)
        else:
            nn.init.constant_(gate.bias, init_bias)
        return gate

    if mode == "rich":
        n_out = 2 * channels if untied else channels
        hidden = max(channels, 32)
        gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, n_out, 1),
        )
        nn.init.zeros_(gate[-1].weight)
        if untied:
            bias = torch.empty(n_out)
            bias[:channels].fill_(init_bias)
            bias[channels:].fill_(-init_bias)
            gate[-1].bias.data.copy_(bias)
        else:
            nn.init.constant_(gate[-1].bias, init_bias)
        return gate

    if untied:
        raise ValueError(f"gate_untied=True is only valid for simple/rich modes, got {mode!r}")

    if mode == "concat_mlp":
        hidden = max(channels, 32)
        return nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    if mode == "gated_mlp_residual":
        hidden = max(channels, 32)
        m = nn.ModuleDict({
            "g": nn.Conv2d(2 * channels, channels, 1),
            "f": nn.Sequential(
                nn.Conv2d(2 * channels, hidden, 1, bias=False),
                nn.GroupNorm(_group_count(hidden), hidden),
                nn.GELU(),
                nn.Conv2d(hidden, channels, 1),
            ),
        })
        nn.init.zeros_(m["g"].weight)
        nn.init.constant_(m["g"].bias, -init_bias)  # gate≈0 → output≈A initially
        return m

    if mode == "addmul":
        m = nn.ModuleDict({
            "add": nn.Conv2d(2 * channels, channels, 1),
            "mul": nn.Conv2d(2 * channels, channels, 1),
        })
        nn.init.zeros_(m["add"].weight)
        nn.init.constant_(m["add"].bias, init_bias)   # add gate≈1 → output≈A
        nn.init.zeros_(m["mul"].weight)
        nn.init.zeros_(m["mul"].bias)                  # mul gate=0.5 sigmoid → small but nonzero start
        # Bias mul gate to start at zero contribution
        nn.init.constant_(m["mul"].bias, -init_bias)   # sigmoid(-4)≈0.018 → near-zero mul start
        return m

    if mode == "film":
        m = nn.ModuleDict({
            "gamma": nn.Conv2d(channels, channels, 1),
            "beta": nn.Conv2d(channels, channels, 1),
        })
        nn.init.zeros_(m["gamma"].weight)
        nn.init.zeros_(m["gamma"].bias)
        nn.init.zeros_(m["beta"].weight)
        nn.init.zeros_(m["beta"].bias)
        return m

    raise ValueError(f"Unknown gate mode: {mode!r}")


def _apply_fusion_gate(gate_module, ae_feat, tes_feat, untied, mode="simple"):
    if mode in ("simple", "rich"):
        raw = gate_module(torch.cat([ae_feat, tes_feat], dim=1))
        if untied:
            channels = ae_feat.size(1)
            g_ae = torch.sigmoid(raw[:, :channels])
            g_tes = torch.sigmoid(raw[:, channels:])
            return g_ae * ae_feat + g_tes * tes_feat
        gate = torch.sigmoid(raw)
        return gate * ae_feat + (1.0 - gate) * tes_feat

    if mode == "concat_mlp":
        return gate_module(torch.cat([ae_feat, tes_feat], dim=1))

    if mode == "gated_mlp_residual":
        concat = torch.cat([ae_feat, tes_feat], dim=1)
        g = torch.sigmoid(gate_module["g"](concat))
        transformed = gate_module["f"](concat)
        return g * transformed + (1.0 - g) * ae_feat

    if mode == "addmul":
        concat = torch.cat([ae_feat, tes_feat], dim=1)
        g_add = torch.sigmoid(gate_module["add"](concat))
        g_mul = torch.sigmoid(gate_module["mul"](concat))
        return g_add * ae_feat + (1.0 - g_add) * tes_feat + g_mul * (ae_feat * tes_feat)

    if mode == "film":
        delta_gamma = gate_module["gamma"](tes_feat)
        beta = gate_module["beta"](tes_feat)
        return (1.0 + delta_gamma) * ae_feat + beta

    raise ValueError(f"Unknown gate mode: {mode!r}")


def _maybe_drop_modality(tes_feat, p, training):
    if not training or p <= 0.0:
        return tes_feat
    batch = tes_feat.size(0)
    keep = (torch.rand(batch, 1, 1, 1, device=tes_feat.device) >= p).float()
    return tes_feat * keep / max(1e-6, 1.0 - p)


class TesseraCompressionStem(nn.Module):
    def __init__(self, in_ch, out_ch=16, hidden_ch=None, hidden_depth=0):
        super().__init__()
        hidden_ch = hidden_ch or max(out_ch * 2, 32)
        hidden_depth = max(0, int(hidden_depth))
        layers = [ConvGNAct(in_ch, hidden_ch, kernel_size=1, padding=0)]
        layers.extend(ConvGNAct(hidden_ch, hidden_ch, kernel_size=3)
                      for _ in range(hidden_depth))
        layers.extend([
            ConvGNAct(hidden_ch, out_ch, kernel_size=3),
            ConvGNAct(out_ch, out_ch, kernel_size=3),
        ])
        self.calib = ChannelCalibration(in_ch)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.calib(x))


class TesseraIoUFusionLightUNet(nn.Module):
    """AlphaEarth LightUNet with a Tessera residual IoU correction branch.

    Tessera feeds only the presence head (as a 16-ch residual correction),
    while the AlphaEarth UNet trunk drives fractions, FiLM, and height.
    This is the architecture that produced all good-performing gated_F and
    ctaskattn runs. Restored for checkpoint compatibility and new experiments.
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 tessera_presence_ch=16, tessera_hidden_ch=None,
                 tessera_hidden_depth=0, height_specialist_depth=0,
                 base_ch=32, height_gate_source="alpha",
                 height_hidden_ch=None, height_trunk_depth=2,
                 height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraIoUFusionLightUNet assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "TesseraIoUFusionLightUNet expects concatenated AlphaEarth+Tessera "
                f"input with >{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        self.alpha_unet = LightUNet(alpha_channels, n_classes, base_ch=base_ch,
                                    norm_kind=norm_kind)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=tessera_presence_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            presence_extra_ch=tessera_presence_ch,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
        )

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_stem(tessera)
        return self.head(alpha_feat, return_aux=return_aux, presence_extra=tessera_feat)


def _make_token_proj(in_ch, out_ch, depth=1):
    """1x1 projection MLP for per-token-source context."""
    layers = []
    cur = in_ch
    for _ in range(max(int(depth) - 1, 0)):
        layers += [nn.Conv2d(cur, out_ch, 1), nn.GELU()]
        cur = out_ch
    layers.append(nn.Conv2d(cur, out_ch, 1))
    return nn.Sequential(*layers)


class CrossSourceHybridFiLMFusion(nn.Module):
    """Cross-source self-attention + per-source FiLM + additive + spatial gate.

    Ported from origin/film (Dinghye, xf085). Takes N token sources concatenated
    along channel dim, refines them via one self-attention layer with learned
    modality embeddings + 2D positional encoding, then injects each refined
    source into pixel features via zero-init (FiLM γ/β + additive A + spatial
    gate σ(g)) residual.
    """

    _TOKEN_INPUT_CLAMP = 50.0
    _CTX_CLAMP = 50.0
    _FILM_PARAM_CLAMP = 4.0
    _ADD_CLAMP = 4.0

    def __init__(self, pixel_ch, token_channels, token_source_ch=768,
                 ctx_ch=96, token_calibration=False, token_proj_depth=1,
                 attn_heads=4, attn_dropout=0.05):
        super().__init__()
        if token_channels % token_source_ch != 0:
            raise ValueError(
                f"CrossSourceHybridFiLMFusion: token_channels={token_channels} must be "
                f"divisible by token_source_ch={token_source_ch}"
            )
        if attn_heads < 1:
            raise ValueError(
                f"CrossSourceHybridFiLMFusion: attn_heads must be >= 1, got {attn_heads}"
            )
        self.token_source_ch = int(token_source_ch)
        self.n_sources = token_channels // token_source_ch
        self.ctx_ch = int(ctx_ch)
        self.attn_heads = int(attn_heads)

        self.token_calibs = (
            nn.ModuleList(
                ChannelCalibration(token_source_ch) for _ in range(self.n_sources)
            )
            if token_calibration else None
        )
        self.token_projs = nn.ModuleList([
            _make_token_proj(token_source_ch, ctx_ch, depth=token_proj_depth)
            for _ in range(self.n_sources)
        ])

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, ctx_ch),
            nn.GELU(),
            nn.Linear(ctx_ch, ctx_ch),
        )
        self.modality_embed = nn.Parameter(torch.zeros(self.n_sources, ctx_ch))
        nn.init.normal_(self.modality_embed, std=0.02)
        self.attn_norm = nn.LayerNorm(ctx_ch)
        self.cross_source_attn = nn.MultiheadAttention(
            ctx_ch, self.attn_heads,
            dropout=attn_dropout, batch_first=True,
        )
        nn.init.zeros_(self.cross_source_attn.out_proj.weight)
        nn.init.zeros_(self.cross_source_attn.out_proj.bias)

        self.film_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, pixel_ch * 2, 1) for _ in range(self.n_sources)
        ])
        self.add_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, pixel_ch, 1) for _ in range(self.n_sources)
        ])
        self.gate_convs = nn.ModuleList([
            nn.Conv2d(ctx_ch, 1, 1) for _ in range(self.n_sources)
        ])
        for module_list in (self.film_convs, self.add_convs, self.gate_convs):
            for conv in module_list:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

    def _pos_tokens(self, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2)
        return self.pos_mlp(coords)

    def _refine_sources(self, ctx_list):
        b, _, h, w = ctx_list[0].shape
        pos = self._pos_tokens(h, w, ctx_list[0].device, ctx_list[0].dtype)
        seqs = []
        for i, ctx in enumerate(ctx_list):
            tokens = ctx.flatten(2).transpose(1, 2)
            tokens = tokens + pos
            tokens = tokens + self.modality_embed[i].view(1, 1, -1)
            seqs.append(tokens)
        x = torch.cat(seqs, dim=1)
        x_norm = self.attn_norm(x)
        with torch.amp.autocast("cuda", enabled=False):
            attn_out, _ = self.cross_source_attn(
                x_norm.float(), x_norm.float(), x_norm.float(),
                need_weights=False,
            )
        chunks = attn_out.to(x.dtype).split(h * w, dim=1)
        refined = []
        for i, ck in enumerate(chunks):
            delta = ck.transpose(1, 2).reshape(b, self.ctx_ch, h, w)
            refined.append(ctx_list[i] + delta)
        return refined

    def forward(self, F_pixel, token):
        H, W = F_pixel.shape[-2:]
        parts = token.float().split(self.token_source_ch, dim=1)
        with torch.amp.autocast("cuda", enabled=False):
            ctx_list = []
            for i, src in enumerate(parts):
                if self.token_calibs is not None:
                    src = self.token_calibs[i](src)
                src = src.clamp(-self._TOKEN_INPUT_CLAMP, self._TOKEN_INPUT_CLAMP)
                ctx = self.token_projs[i](src).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
                ctx_list.append(ctx)

            refined = self._refine_sources(ctx_list)
            refined = [r.clamp(-self._CTX_CLAMP, self._CTX_CLAMP) for r in refined]

            F_p_f = F_pixel.float()
            delta = torch.zeros_like(F_p_f)
            for i, ctx in enumerate(refined):
                ctx_up = F.interpolate(
                    ctx, size=(H, W), mode="bilinear", align_corners=False
                )
                gamma, beta = self.film_convs[i](ctx_up).chunk(2, dim=1)
                gamma = gamma.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                beta = beta.clamp(-self._FILM_PARAM_CLAMP, self._FILM_PARAM_CLAMP)
                add = self.add_convs[i](ctx_up).clamp(-self._ADD_CLAMP, self._ADD_CLAMP)
                g = torch.sigmoid(self.gate_convs[i](ctx_up))
                delta = delta + g * (gamma * F_p_f + beta + add)
            out = (F_p_f + delta).clamp(-self._CTX_CLAMP, self._CTX_CLAMP)
        return out.to(F_pixel.dtype)


class TesseraIoUFusionGatedLightUNet(nn.Module):
    """Active AlphaEarth + Tessera model: gated trunk fusion plus optional
    presence residual.
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 tessera_presence_ch=0, tessera_hidden_ch=None,
                 tessera_hidden_depth=0, height_specialist_depth=0,
                 base_ch=32, gate_init_bias=4.0,
                 gate_mode="simple", gate_untied=False,
                 modality_dropout=0.0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False,
                 ae_only_supervision=False,
                 token_channels=0, use_se=False, use_coord_attn=False,
                 use_bottleneck_attn=False, use_mixstyle=False, disable_head_film=False,
                 use_attn_gates=False, use_aspp=False, bottleneck_attn_depth=1,
                 use_modern=False, detail_bypass=False, sharp_upsample=False,
                 scene_film=False, encoder_arch="unet",
                 use_xsource_fusion=False, token_source_ch=768, token_ctx_ch=96,
                 xsource_attn_heads=4, xsource_token_calibration=False,
                 use_spatial_token_film=False, height_dropout=0.0,
                 use_shape_queries=False, shape_n_queries=32, shape_depth=2):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraIoUFusionGatedLightUNet assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "TesseraIoUFusionGatedLightUNet expects concatenated AlphaEarth+Tessera "
                f"input with >{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.gate_untied = bool(gate_untied)
        self.gate_mode = gate_mode
        self.modality_dropout = float(modality_dropout)
        tessera_channels = n_channels - alpha_channels

        self.alpha_unet = LightUNet(
            alpha_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind,
            use_se=bool(use_se), use_coord_attn=bool(use_coord_attn),
            use_bottleneck_attn=bool(use_bottleneck_attn),
            use_mixstyle=bool(use_mixstyle),
            use_attn_gates=bool(use_attn_gates),
            use_aspp=bool(use_aspp),
            bottleneck_attn_depth=int(bottleneck_attn_depth),
            use_modern=bool(use_modern),
            detail_bypass=bool(detail_bypass),
            sharp_upsample=bool(sharp_upsample),
            scene_film=bool(scene_film),
            encoder_arch=str(encoder_arch),
        )
        self.alpha_unet.head = nn.Identity()
        self.tessera_feature_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=base_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        self.gate_conv = _build_fusion_gate(
            base_ch,
            mode=gate_mode,
            untied=self.gate_untied,
            init_bias=gate_init_bias,
        )

        self.tessera_presence_ch = int(tessera_presence_ch)
        self.presence_extra_proj = (
            nn.Conv2d(base_ch, self.tessera_presence_ch, 1)
            if self.tessera_presence_ch > 0 else None
        )

        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            presence_extra_ch=self.tessera_presence_ch,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
            disable_head_film=disable_head_film,
            height_dropout=height_dropout,
            use_shape_queries=use_shape_queries,
            shape_n_queries=shape_n_queries,
            shape_depth=shape_depth,
        )

        # CMGFNet-style deep supervision: parallel light prediction head from
        # pre-fusion AE features. 3 presence channels + 1 height channel.
        # Supervised with same BCE+Tversky losses; not used for inference.
        if ae_only_supervision:
            self.ae_only_head = nn.Sequential(
                ConvGNAct(base_ch, base_ch, kernel_size=3),
                nn.Conv2d(base_ch, 4, kernel_size=1),
            )
        else:
            self.ae_only_head = None

        # Optional token-residual path: linear addition of token-derived
        # features into the AE+Tessera gated-fused features. Zero-init scalar
        # per-channel gate (sigmoid(-4) ≈ 0.018) so the path is near-identity
        # at training start and can only learn to help.
        self.token_channels = int(token_channels)
        self.use_xsource_fusion = bool(use_xsource_fusion)
        self.use_spatial_token_film = bool(use_spatial_token_film)
        if self.token_channels > 0 and self.use_xsource_fusion:
            self.xsource_fusion = CrossSourceHybridFiLMFusion(
                pixel_ch=base_ch,
                token_channels=self.token_channels,
                token_source_ch=int(token_source_ch),
                ctx_ch=int(token_ctx_ch),
                token_calibration=bool(xsource_token_calibration),
                attn_heads=int(xsource_attn_heads),
            )
            self.token_neck = None
            self.token_residual_proj = None
            self.token_residual_gate = None
        elif self.token_channels > 0:
            from .token_fusion import TokenPyramidNeck  # lazy import: avoid circular
            self.token_neck = TokenPyramidNeck(
                self.token_channels,
                level_channels=(base_ch * 2, base_ch, base_ch, base_ch),
            )
            self.token_residual_proj = nn.Conv2d(base_ch, base_ch, 1)
            nn.init.zeros_(self.token_residual_proj.weight)
            nn.init.zeros_(self.token_residual_proj.bias)
            if self.use_spatial_token_film:
                # xf095-style spatial-gated FiLM: F_out = F + sigmoid(g(t)) · (gamma·F + beta + A)
                # All zero-init: γ stays at 1 (identity), β=0, A=0, g=σ(-4)≈0.018 → near-identity at start.
                self.token_film_gamma = nn.Conv2d(base_ch, base_ch, 1)
                self.token_film_beta = nn.Conv2d(base_ch, base_ch, 1)
                nn.init.zeros_(self.token_film_gamma.weight)
                nn.init.zeros_(self.token_film_gamma.bias)
                nn.init.zeros_(self.token_film_beta.weight)
                nn.init.zeros_(self.token_film_beta.bias)
                # Spatial gate: (B, 1, H, W) — one mask per pixel, broadcast to all channels.
                self.token_spatial_gate = nn.Conv2d(base_ch, 1, 1)
                nn.init.zeros_(self.token_spatial_gate.weight)
                nn.init.constant_(self.token_spatial_gate.bias, -4.0)
                self.token_residual_gate = None
            else:
                self.token_film_gamma = None
                self.token_film_beta = None
                self.token_spatial_gate = None
                self.token_residual_gate = nn.Parameter(torch.full((1, base_ch, 1, 1), -4.0))
            self.xsource_fusion = None
        else:
            self.token_neck = None
            self.token_residual_proj = None
            self.token_residual_gate = None
            self.token_film_gamma = None
            self.token_film_beta = None
            self.token_spatial_gate = None
            self.xsource_fusion = None

    def forward(self, x, return_aux=False):
        if isinstance(x, (tuple, list)):
            pixel_x, token = x
        else:
            pixel_x, token = x, None
        alpha = pixel_x[:, :self.alpha_channels, :, :]
        tessera = pixel_x[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat,
            untied=self.gate_untied, mode=self.gate_mode
        )

        if self.xsource_fusion is not None and token is not None:
            fused = self.xsource_fusion(fused, token)
        elif self.token_neck is not None and token is not None:
            tpyr = self.token_neck(token)
            t_feat = tpyr[128]
            if t_feat.shape[-2:] != fused.shape[-2:]:
                t_feat = F.interpolate(
                    t_feat, size=fused.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            if self.use_spatial_token_film:
                gamma = self.token_film_gamma(t_feat).clamp(-4.0, 4.0)
                beta = self.token_film_beta(t_feat).clamp(-4.0, 4.0)
                a_res = self.token_residual_proj(t_feat).clamp(-4.0, 4.0)
                g = torch.sigmoid(self.token_spatial_gate(t_feat))
                fused = fused + g * (gamma * fused + beta + a_res)
            else:
                t_res = self.token_residual_proj(t_feat)
                fused = fused + torch.sigmoid(self.token_residual_gate) * t_res

        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        out = self.head(fused, return_aux=return_aux,
                        presence_extra=presence_extra)
        if return_aux and self.ae_only_head is not None and isinstance(out, dict):
            out["ae_only_logits"] = self.ae_only_head(alpha_feat)
        return out


class TesseraCrossAttnLightUNet(nn.Module):
    """AE encoder–Tessera cross-attention at the UNet bottleneck.

    Architecture:
      1. AE runs through LightUNet encoder → bottleneck (x4) + skip connections.
      2. Tessera is compressed to base_ch at full resolution, then downsampled
         (3× MaxPool + DoubleConv) to match the bottleneck spatial resolution.
      3. Multi-head cross-attention fuses the two bottlenecks: AE is Q, Tessera
         is K and V.  Residual connection preserves the AE bottleneck signal.
      4. Fused bottleneck passes through the LightUNet decoder → prediction head.

    Scales to N modalities: each additional source contributes its own K/V
    and the AE Q attends to all of them jointly (just concatenate K, V).
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 tessera_hidden_ch=None, tessera_hidden_depth=0,
                 height_specialist_depth=0, base_ch=32, n_heads=4,
                 modality_dropout=0.0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None):
        super().__init__()
        if n_classes != 4:
            raise ValueError("TesseraCrossAttnLightUNet assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "TesseraCrossAttnLightUNet expects concatenated AlphaEarth+Tessera "
                f"input with >{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.modality_dropout = float(modality_dropout)
        tessera_channels = n_channels - alpha_channels

        self.alpha_unet = LightUNet(
            alpha_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind
        )
        self.alpha_unet.head = nn.Identity()

        bottleneck_ch = base_ch * 8   # c4 from LightUNet encoder

        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=base_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        # Mirror the LightUNet encoder's 3 downsampling stages to reach the
        # same spatial resolution as the AE bottleneck (input / 8).
        self.tessera_downsample = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch, base_ch * 2, norm_kind=norm_kind),
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 2, base_ch * 4, norm_kind=norm_kind),
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 4, bottleneck_ch, norm_kind=norm_kind),
        )

        # Cross-attention: AE bottleneck (Q) attends to Tessera bottleneck (K,V).
        # batch_first=True → (B, seq_len, embed_dim) layout.
        self.xattn = nn.MultiheadAttention(
            embed_dim=bottleneck_ch,
            num_heads=n_heads,
            batch_first=True,
        )
        self.xattn_norm = nn.LayerNorm(bottleneck_ch)

        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
        )

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        # AE encoder
        x4, skips = self.alpha_unet.forward_encoder(alpha)
        B, C, H, W = x4.shape

        # Tessera to bottleneck resolution
        tes_feat = self.tessera_stem(tessera)
        tes_feat = _maybe_drop_modality(tes_feat, self.modality_dropout, self.training)
        tes_bn = self.tessera_downsample(tes_feat)   # (B, C, H, W)

        # Cross-attention at bottleneck: flatten spatial → (B, H*W, C)
        q = x4.flatten(2).permute(0, 2, 1)
        kv = tes_bn.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.xattn(q, kv, kv)
        attn_out = self.xattn_norm(attn_out)
        fused_x4 = x4 + attn_out.permute(0, 2, 1).reshape(B, C, H, W)

        features = self.alpha_unet.forward_decoder(fused_x4, skips)
        return self.head(features, return_aux=return_aux)


class SimpleConcatFusion(nn.Module):
    """Lightweight AE+Tessera fusion: no UNet, all-256x256, per-pixel.

    AE → GroupNorm + 1x1 → 48ch
    Tessera → ChannelCalibration + 1x1 → 80ch
    concat → 128ch → ConvGNAct(96) → ConvGNAct(64) → MultiTaskPredictionHead
    Hypothesis: the pretrained embeddings already encode spatial-spectral
    features, so a heavy UNet backbone is redundant. ~16% of the gated model's
    params. Risk is that lack of multi-scale context hurts large-building IoU.
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 ae_proj_ch=48, tessera_proj_ch=80,
                 trunk_hidden=96, trunk_out=64,
                 height_specialist_depth=0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False):
        super().__init__()
        if n_classes != 4:
            raise ValueError("SimpleConcatFusion assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "SimpleConcatFusion expects AlphaEarth+Tessera concat input "
                f">{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        # AE branch: GroupNorm + 1x1 projection
        self.ae_norm = nn.GroupNorm(1, alpha_channels)
        self.ae_proj = nn.Conv2d(alpha_channels, ae_proj_ch, kernel_size=1)

        # Tessera branch: per-channel calibration + 1x1 projection
        self.tessera_calib = ChannelCalibration(tessera_channels)
        self.tessera_proj = nn.Conv2d(tessera_channels, tessera_proj_ch, kernel_size=1)

        fused_ch = ae_proj_ch + tessera_proj_ch
        self.trunk = nn.Sequential(
            ConvGNAct(fused_ch, trunk_hidden, kernel_size=3),
            ConvGNAct(trunk_hidden, trunk_out, kernel_size=3),
        )

        self.head = MultiTaskPredictionHead(
            in_ch=trunk_out,
            out_channels=n_classes,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
        )

    def forward(self, x, return_aux=False):
        ae = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        ae_feat = self.ae_proj(self.ae_norm(ae))
        tessera_feat = self.tessera_proj(self.tessera_calib(tessera))

        fused = torch.cat([ae_feat, tessera_feat], dim=1)
        features = self.trunk(fused)
        return self.head(features, return_aux=return_aux)


class SimpleGatedFusion(nn.Module):
    """Hybrid: teammate's no-UNet trunk + our gated fusion mechanism.

    AE → GroupNorm + 1x1 → 64ch
    Tessera → ChannelCalibration + 1x1 → 64ch
    Gated mix → 64ch (sigmoid gate, init biases toward AE: sigmoid(4) ≈ 0.98)
    ConvGNAct(64 → 96) → ConvGNAct(96 → 64) → MultiTaskPredictionHead
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 branch_ch=64, trunk_hidden=96, trunk_out=64,
                 gate_init_bias=4.0,
                 height_specialist_depth=0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False):
        super().__init__()
        if n_classes != 4:
            raise ValueError("SimpleGatedFusion assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "SimpleGatedFusion expects AlphaEarth+Tessera concat input "
                f">{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        # Branches → equal channel dim for gated mixing
        self.ae_norm = nn.GroupNorm(1, alpha_channels)
        self.ae_proj = nn.Conv2d(alpha_channels, branch_ch, kernel_size=1)
        self.tessera_calib = ChannelCalibration(tessera_channels)
        self.tessera_proj = nn.Conv2d(tessera_channels, branch_ch, kernel_size=1)

        # Sigmoid gate, AE-dominant at init: g = sigmoid(b_init) ≈ 0.98.
        # fused = g·ae + (1-g)·tessera. Gate is per-pixel per-channel,
        # computed from concat([ae, tessera]).
        self.gate = nn.Conv2d(2 * branch_ch, branch_ch, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init_bias)

        self.trunk = nn.Sequential(
            ConvGNAct(branch_ch, trunk_hidden, kernel_size=3),
            ConvGNAct(trunk_hidden, trunk_out, kernel_size=3),
        )

        self.head = MultiTaskPredictionHead(
            in_ch=trunk_out,
            out_channels=n_classes,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
        )

    def forward(self, x, return_aux=False):
        ae = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        ae_feat = self.ae_proj(self.ae_norm(ae))
        tessera_feat = self.tessera_proj(self.tessera_calib(tessera))

        g = torch.sigmoid(self.gate(torch.cat([ae_feat, tessera_feat], dim=1)))
        fused = g * ae_feat + (1.0 - g) * tessera_feat

        features = self.trunk(fused)
        return self.head(features, return_aux=return_aux)


class SimpleConcatConvNeXt(nn.Module):
    """Lightweight per-pixel fusion with ConvNeXt trunk for larger receptive
    field. Targets building IoU specifically: 7x7 depthwise conv per block
    gives more spatial context than 3x3 ConvGNAct.

    AE → GroupNorm + 1x1 → 48ch
    Tessera → ChannelCalibration + 1x1 → 80ch
    concat → 128ch → 1x1 reduce → 64ch → ConvNeXtBlock(64) × n_blocks → head
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 ae_proj_ch=48, tessera_proj_ch=80,
                 trunk_ch=64, n_blocks=3,
                 height_specialist_depth=0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False):
        super().__init__()
        if n_classes != 4:
            raise ValueError("SimpleConcatConvNeXt assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "SimpleConcatConvNeXt expects AlphaEarth+Tessera concat input "
                f">{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        self.ae_norm = nn.GroupNorm(1, alpha_channels)
        self.ae_proj = nn.Conv2d(alpha_channels, ae_proj_ch, kernel_size=1)
        self.tessera_calib = ChannelCalibration(tessera_channels)
        self.tessera_proj = nn.Conv2d(tessera_channels, tessera_proj_ch, kernel_size=1)

        fused_ch = ae_proj_ch + tessera_proj_ch
        self.reduce = nn.Sequential(
            nn.Conv2d(fused_ch, trunk_ch, kernel_size=1),
            nn.GroupNorm(_group_count(trunk_ch), trunk_ch),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(trunk_ch) for _ in range(n_blocks)
        ])

        self.head = MultiTaskPredictionHead(
            in_ch=trunk_ch,
            out_channels=n_classes,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
        )

    def forward(self, x, return_aux=False):
        ae = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        ae_feat = self.ae_proj(self.ae_norm(ae))
        tessera_feat = self.tessera_proj(self.tessera_calib(tessera))

        fused = torch.cat([ae_feat, tessera_feat], dim=1)
        features = self.blocks(self.reduce(fused))
        return self.head(features, return_aux=return_aux)


class SimpleConcatASPP(nn.Module):
    """Lightweight per-pixel fusion with ASPP for multi-scale context.

    AE → GroupNorm + 1x1 → 48ch
    Tessera → ChannelCalibration + 1x1 → 80ch
    concat → 128ch → ASPP(128 → 96, rates=(1,3,6,12)) → ConvGNAct(64) → head
    Smaller dilation rates than the default since building variations are
    bounded; rates=(1,3,6,12) captures up to ~25px context at 256x256.
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 ae_proj_ch=48, tessera_proj_ch=80,
                 aspp_ch=96, trunk_out=64,
                 aspp_rates=(1, 3, 6, 12),
                 height_specialist_depth=0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False):
        super().__init__()
        if n_classes != 4:
            raise ValueError("SimpleConcatASPP assumes 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "SimpleConcatASPP expects AlphaEarth+Tessera concat input "
                f">{alpha_channels} channels, got {n_channels}"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        self.ae_norm = nn.GroupNorm(1, alpha_channels)
        self.ae_proj = nn.Conv2d(alpha_channels, ae_proj_ch, kernel_size=1)
        self.tessera_calib = ChannelCalibration(tessera_channels)
        self.tessera_proj = nn.Conv2d(tessera_channels, tessera_proj_ch, kernel_size=1)

        fused_ch = ae_proj_ch + tessera_proj_ch
        self.aspp = ASPP(fused_ch, aspp_ch, rates=aspp_rates)
        self.reduce = ConvGNAct(aspp_ch, trunk_out, kernel_size=3)

        self.head = MultiTaskPredictionHead(
            in_ch=trunk_out,
            out_channels=n_classes,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
        )

    def forward(self, x, return_aux=False):
        ae = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        ae_feat = self.ae_proj(self.ae_norm(ae))
        tessera_feat = self.tessera_proj(self.tessera_calib(tessera))

        fused = torch.cat([ae_feat, tessera_feat], dim=1)
        features = self.reduce(self.aspp(fused))
        return self.head(features, return_aux=return_aux)


class PixelMoEFusion(nn.Module):
    """Pixel-level Top-K=2 Mixture-of-Experts fusion of AE + Tessera.

    Per deep research recommendation #1 — dynamic routing replaces static
    gating. Router projects concat(AE, Tessera) → expert logits per pixel,
    Top-K=2 experts process the concat features in parallel, outputs are
    weighted-summed by the (renormalized) top-K softmax probs. Each pixel
    routes to a different expert combination based on its content.

    Strong fit for small-data regimes (research): sparse activation acts
    as structural regularization — only K/N of the model's capacity is
    active per pixel per forward pass, so the effective parameter budget
    seen by any single training sample stays modest.

    Wired upstream of the alpha_unet branch in TesseraIoUFusionGatedLightUNet,
    so the LightUNet trunk and multi-task head are unchanged. Only the
    fusion layer differs.
    """
    def __init__(self, in_channels=192, out_channels=64, num_experts=4, k=2,
                 expert_hidden=128):
        super().__init__()
        self.k = int(k)
        self.num_experts = int(num_experts)
        # Router
        self.router = nn.Conv2d(in_channels, num_experts, kernel_size=1)
        # Lightweight experts (1x1 → hidden → 1x1 with GELU)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, expert_hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(expert_hidden, out_channels, kernel_size=1),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (B, in_channels, H, W); both AE and Tessera concatenated
        b, c, h, w = x.shape
        logits = self.router(x)                          # (B, E, H, W)
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_idx = torch.topk(probs, self.k, dim=1)  # (B, k, H, W)
        topk_probs = topk_probs / (topk_probs.sum(dim=1, keepdim=True) + 1e-8)

        # Forward through ALL experts then gather (simple but not most efficient;
        # acceptable for E=4, k=2 at our scale)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # expert_outs: (B, E, out_channels, H, W)
        out = torch.zeros(b, expert_outs.shape[2], h, w, device=x.device, dtype=x.dtype)
        for i in range(self.k):
            idx_i = topk_idx[:, i:i+1, :, :]              # (B, 1, H, W) values in [0, E)
            prob_i = topk_probs[:, i:i+1, :, :]
            # Gather selected expert output: index along E-dim with idx_i
            expanded_idx = idx_i.unsqueeze(2).expand(-1, -1, expert_outs.shape[2], -1, -1)
            selected = torch.gather(expert_outs, 1, expanded_idx).squeeze(1)  # (B, out, H, W)
            out = out + prob_i * selected
        return out


class AeTesseraMoeFusion(nn.Module):
    """AE+Tessera with per-pixel MoE fusion → LightUNet → MultiTaskHead.

    Mirrors TesseraIoUFusionGatedLightUNet's structure but the gated fusion
    layer is replaced by a PixelMoEFusion module. Everything downstream
    (UNet trunk, multi-task head, bidirectional cross-task, softbin height)
    is identical to the canon recipe — so any score delta is directly
    attributable to the fusion mechanism.
    """
    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 num_experts=4, k=2, expert_hidden=128, base_ch=48,
                 norm_kind="bn", height_specialist_depth=0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False, disable_head_film=False,
                 use_bottleneck_attn=False, **_ignored):
        super().__init__()
        if n_classes != 4:
            raise ValueError("AeTesseraMoeFusion assumes 4 output channels")
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        # MoE fusion outputs base_ch (matches LightUNet first-layer width)
        self.moe = PixelMoEFusion(
            in_channels=n_channels, out_channels=base_ch,
            num_experts=num_experts, k=k, expert_hidden=expert_hidden,
        )
        # Then a LightUNet trunk + multi-task head, but starting from base_ch
        # input rather than full concat. Mirrors the canon recipe with the
        # fusion stage replaced.
        self.unet = LightUNet(
            base_ch, n_classes, base_ch=base_ch, norm_kind=norm_kind,
            use_bottleneck_attn=bool(use_bottleneck_attn),
        )
        # Replace UNet's default head with our richer multi-task head
        self.unet.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
            disable_head_film=disable_head_film,
        )

    def forward(self, x, return_aux=False):
        # x: (B, n_channels, H, W) — full concat of AE + Tessera
        fused = self.moe(x)
        # Use UNet's forward via direct call
        x4, skips = self.unet.forward_encoder(fused)
        feat = self.unet.forward_decoder(x4, skips)
        return self.unet.head(feat, return_aux=return_aux)


class AeTesseraMlpFusion(nn.Module):
    """Lightweight MLP-only decoder over AE+Tessera embeddings.

    Tests the hypothesis (THOR 2026, TESSERA 2026) that strong foundation-
    model embeddings (AE 64-D ViT, Tessera 128-D Transformer+GRU) already
    encode all the spatial+temporal context required for dense prediction,
    and a heavy UNet decoder is just a 2.5M-param invitation to overfit in
    our 2024-tile, leave-region-out CV regime.

    Architecture:
      1. Per-modality InstanceNorm to neutralize the magnitude mismatch
         (AE range [-90, 80] vs Tessera [-13, 16]) — addresses regional
         style/scale shifts before feature mixing.
      2. Concat → single 1x1 conv into shared hidden width (default 256)
         with BN + GELU. Zero spatial receptive-field expansion in the
         decoder; all spatial context comes from the encoders.
      3. Pass through MultiTaskPredictionHead (FiLM/softbin/multi-task
         outputs) for loss compatibility with the canon recipe.

    Total: ~250-400k params depending on head config — ~5-10% of the
    canonical 5.2M model. The structural bottleneck forces the network
    to rely on the pre-trained embeddings rather than memorizing training-
    tile spatial idiosyncrasies.
    """
    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 hidden_ch=256, height_specialist_depth=0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0, norm_kind="bn",
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False, disable_head_film=False,
                 use_bottleneck_attn=False, **_ignored):
        super().__init__()
        if n_classes != 4:
            raise ValueError("AeTesseraMlpFusion assumes 4 output channels")
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels
        if tessera_channels <= 0:
            raise ValueError(
                f"Expected n_channels > alpha_channels ({alpha_channels}); got {n_channels}"
            )

        # Per-modality InstanceNorm: strip regional style/illumination/scale.
        # Operates on each (B, C, H, W) independently per sample, per channel.
        self.ae_norm = nn.InstanceNorm2d(alpha_channels, affine=True)
        self.tessera_norm = nn.InstanceNorm2d(tessera_channels, affine=True)

        # Single 1x1 mixing layer — no spatial conv, no receptive field growth.
        self.mix = nn.Sequential(
            nn.Conv2d(n_channels, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch) if norm_kind == "bn" else nn.GroupNorm(
                _group_count(hidden_ch), hidden_ch),
            nn.GELU(),
        )

        # Optional bottleneck self-attention block (treats the entire feature
        # map as tokens; here, no downsampling so it acts at full 256x256).
        # Use sparingly — quadratic memory.
        from .backbones import BottleneckSelfAttention
        self.bottleneck_attn = (
            BottleneckSelfAttention(hidden_ch) if use_bottleneck_attn else None
        )

        # Existing multi-task head provides FiLM, softbin, aux heights, etc.
        self.head = MultiTaskPredictionHead(
            in_ch=hidden_ch,
            out_channels=n_classes,
            hidden_ch=hidden_ch,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
            disable_head_film=disable_head_film,
        )

    def forward(self, x, return_aux=False):
        ae = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]
        ae = self.ae_norm(ae)
        tessera = self.tessera_norm(tessera)
        x = torch.cat([ae, tessera], dim=1)
        x = self.mix(x)
        if self.bottleneck_attn is not None:
            x = self.bottleneck_attn(x)
        return self.head(x, return_aux=return_aux)


class TesseraPyramidEncoder(nn.Module):
    """Tessera-side hierarchical encoder producing 4 levels of features that
    match the AlphaEarth LightUNet encoder shape: (t1, t2, t3, t4) at
    (base_ch, 2*base_ch, 4*base_ch, 8*base_ch) channels and (1x, 1/2x, 1/4x,
    1/8x) spatial. Used by TesseraIoUFusionMultiLevelGatedLightUNet to gate at
    every decoder level (CMGFNet-style multi-level cross-modal fusion).

    Hardcodes GroupNorm (not BN) — matches the canonical TesseraCompressionStem
    pattern. BN here was empirically unstable at eval time because Tessera
    embedding statistics differ enough from AE that BN running stats diverge
    between train (per-batch normalization) and eval (running-stats normalization).
    """
    def __init__(self, in_ch, base_ch=32, norm_kind="gn"):  # kept for signature
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.calib = ChannelCalibration(in_ch)
        dc_kw = dict(norm_kind="gn")
        self.stem = DoubleConv(in_ch, c1, **dc_kw)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, **dc_kw))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, **dc_kw))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4, **dc_kw))

    def forward(self, x):
        x = self.calib(x)
        t1 = self.stem(x)
        t2 = self.down1(t1)
        t3 = self.down2(t2)
        t4 = self.down3(t3)
        return t1, t2, t3, t4


class TesseraIoUFusionMultiLevelGatedLightUNet(nn.Module):
    """Multi-level cross-modal gated fusion (CMGFNet-style).

    Both AE and Tessera get parallel hierarchical encoders. Fusion happens at
    EVERY decoder level via independent sigmoid gates, not just at the final
    output as in TesseraIoUFusionGatedLightUNet. This lets the model decide
    per-scale how much each modality should contribute.

    Forward:
        ae → alpha_unet.forward_encoder → (a4_bottleneck, (a1, a2, a3))
        tessera → tessera_pyramid → (t1, t2, t3, t4)
        b = gate4(a4, t4)                            # bottleneck fusion
        d3 = up1(b); d3 = conv1(cat[a3, d3])         # AE decoder
        d3 = gate3(d3, t3)                           # per-level Tessera fusion
        d2 = up2(d3); d2 = conv2(cat[a2, d2])
        d2 = gate2(d2, t2)
        d1 = up3(d2); d1 = conv3(cat[a1, d1])
        d1 = gate1(d1, t1)
        return head(d1)
    """
    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 base_ch=32, gate_init_bias=4.0, norm_kind="bn",
                 height_specialist_depth=0, height_gate_source="alpha",
                 height_hidden_ch=None, height_trunk_depth=2,
                 height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False, use_se=False, use_coord_attn=False,
                 use_bottleneck_attn=False, use_mixstyle=False,
                 use_attn_gates=False, disable_head_film=False):
        super().__init__()
        if n_classes != 4:
            raise ValueError("MultiLevel CMGF expects 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError(
                "MultiLevel CMGF expects concatenated AE+Tessera input"
            )
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        # AE side: only the ENCODER is reused from LightUNet. The decoder is
        # built fresh below because re-using alpha_unet's decoder caused
        # train/eval BN running-stat divergence (its BN was being fed
        # gated-fused features instead of the canonical decoder inputs).
        self.alpha_unet = LightUNet(
            alpha_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind,
            use_se=bool(use_se), use_coord_attn=bool(use_coord_attn),
            use_bottleneck_attn=bool(use_bottleneck_attn),
            use_mixstyle=bool(use_mixstyle),
            use_attn_gates=False,  # use OUR attn gates below, not alpha_unet's
        )
        # We don't use alpha_unet's head OR decoder; null them out so the
        # state dict stays tidy and unused params don't waste memory.
        self.alpha_unet.head = nn.Identity()

        self.tessera_pyramid = TesseraPyramidEncoder(
            tessera_channels, base_ch=base_ch, norm_kind=norm_kind,
        )

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Fresh decoder owned by THIS module.
        dc_kw = dict(norm_kind=norm_kind, use_se=bool(use_se), use_coord_attn=bool(use_coord_attn))
        self.up1 = UpsampleBlock(c4, c3, norm_kind=norm_kind)
        self.conv1 = DoubleConv(c4, c3, **dc_kw)
        self.up2 = UpsampleBlock(c3, c2, norm_kind=norm_kind)
        self.conv2 = DoubleConv(c3, c2, **dc_kw)
        self.up3 = UpsampleBlock(c2, c1, norm_kind=norm_kind)
        self.conv3 = DoubleConv(c2, c1, **dc_kw)

        # Optional skip-connection attention gates (Attention U-Net pattern)
        if use_attn_gates:
            self.ag1 = AttentionGate(gate_channels=c3, skip_channels=c3)
            self.ag2 = AttentionGate(gate_channels=c2, skip_channels=c2)
            self.ag3 = AttentionGate(gate_channels=c1, skip_channels=c1)
        else:
            self.ag1 = self.ag2 = self.ag3 = None

        # Per-level cross-modal gates. Use "simple" mode = single tied sigmoid;
        # init bias so AE dominates at start (g≈sigmoid(4)≈0.98) — model starts
        # equivalent to AE-only and can only learn to incorporate Tessera.
        self.gate_b = _build_fusion_gate(c4, mode="simple", untied=False, init_bias=gate_init_bias)
        self.gate_3 = _build_fusion_gate(c3, mode="simple", untied=False, init_bias=gate_init_bias)
        self.gate_2 = _build_fusion_gate(c2, mode="simple", untied=False, init_bias=gate_init_bias)
        self.gate_1 = _build_fusion_gate(c1, mode="simple", untied=False, init_bias=gate_init_bias)

        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            presence_extra_ch=0,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
            disable_head_film=disable_head_film,
        )

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        a4, (a1, a2, a3) = self.alpha_unet.forward_encoder(alpha)
        t1, t2, t3, t4 = self.tessera_pyramid(tessera)

        # Bottleneck fusion
        b = _apply_fusion_gate(self.gate_b, a4, t4, untied=False, mode="simple")

        # Decoder level 3 (at c3, 1/4 resolution): own decoder block.
        u = self.up1(b)
        s = self.ag1(u, a3) if self.ag1 is not None else a3
        d3 = self.conv1(torch.cat([s, u], dim=1))
        d3 = _apply_fusion_gate(self.gate_3, d3, t3, untied=False, mode="simple")

        # Decoder level 2 (at c2, 1/2 resolution)
        u = self.up2(d3)
        s = self.ag2(u, a2) if self.ag2 is not None else a2
        d2 = self.conv2(torch.cat([s, u], dim=1))
        d2 = _apply_fusion_gate(self.gate_2, d2, t2, untied=False, mode="simple")

        # Decoder level 1 (at c1, full resolution)
        u = self.up3(d2)
        s = self.ag3(u, a1) if self.ag3 is not None else a1
        d1 = self.conv3(torch.cat([s, u], dim=1))
        d1 = _apply_fusion_gate(self.gate_1, d1, t1, untied=False, mode="simple")

        return self.head(d1, return_aux=return_aux)


class _DropPath(nn.Module):
    """Stochastic depth: randomly drop the (residual) branch per sample during
    training. Parameter-free and a no-op at eval, so it does not change the
    checkpoint or inference outputs (predict-time compatible)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x.div(keep) * mask


class _EfficientSelfAttention(nn.Module):
    """SegFormer-style efficient self-attention with spatial reduction in K, V.

    Reduces the K, V sequence length by `sr_ratio` via a strided conv, then
    does standard MHA between full-length Q and reduced K, V. This makes
    attention tractable at high-resolution encoder stages where vanilla MSA
    would OOM (e.g. 16k tokens at 1/2 spatial).
    """
    def __init__(self, dim, n_heads, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = int(sr_ratio)
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x_seq, h, w):
        # x_seq: [B, N=H*W, C]
        b, n, c = x_seq.shape
        q = self.q(x_seq).reshape(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        if self.sr is not None:
            x_spatial = x_seq.transpose(1, 2).reshape(b, c, h, w)
            x_red = self.sr(x_spatial).flatten(2).transpose(1, 2)
            x_red = self.norm(x_red)
        else:
            x_red = x_seq
        kv = self.kv(x_red).reshape(b, -1, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class _SegFormerBlock(nn.Module):
    """Transformer block with efficient self-attention + Mix-FFN (depthwise
    conv inside the MLP, per SegFormer paper). Pre-norm."""
    def __init__(self, dim, n_heads, sr_ratio=1, mlp_ratio=2.0,
                 drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _EfficientSelfAttention(dim, n_heads, sr_ratio=sr_ratio,
                                            attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
        self.drop_path = _DropPath(drop_path)

    def _mlp(self, x_seq, h, w):
        b, n, _ = x_seq.shape
        x = self.fc1(x_seq)
        # Mix-FFN: spatial 3x3 depthwise conv inside the MLP
        x_spatial = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dwconv(x_spatial).flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        return self.drop(self.fc2(x))

    def forward(self, x_seq, h, w):
        x_seq = x_seq + self.drop_path(self.attn(self.norm1(x_seq), h, w))
        x_seq = x_seq + self.drop_path(self._mlp(self.norm2(x_seq), h, w))
        return x_seq


class _OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding: strided 3x3 (or 7x7) conv → LayerNorm."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, padding=kernel // 2)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return self.norm(x_seq), h, w


class SegFormerLiteEncoder(nn.Module):
    """SegFormer-style hierarchical transformer encoder, drop-in for
    LightUNet's conv encoder.

    Stage layout (matches LightUNet's encoder output shapes so the existing
    decoder can be reused):
      stem (DoubleConv, 1x at c1) — pure conv, attention at 256x256 is OOM
      stage 2 (1/2 at c2, depth=2, SR=4)
      stage 3 (1/4 at c3, depth=2, SR=2)
      stage 4 (1/8 at c4, depth=2, SR=1)

    Returns (x4, (x1, x2, x3)) like LightUNet.forward_encoder.
    """
    def __init__(self, in_ch, base_ch=32, norm_kind="bn",
                 depths=(2, 2, 2), n_heads=(2, 4, 8), sr_ratios=(4, 2, 1),
                 mlp_ratio=2.0, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Linear stochastic-depth schedule: drop prob ramps 0 -> drop_path_rate
        # across all transformer blocks (SegFormer / DeiT convention).
        total_blocks = sum(depths)
        dpr = [drop_path_rate * i / max(1, total_blocks - 1)
               for i in range(total_blocks)]

        # Stem: pure conv at full resolution.
        self.stem = DoubleConv(in_ch, c1, norm_kind=norm_kind)

        # Stage 2: 1x -> 1/2 spatial, c1 -> c2 channels.
        self.patch2 = _OverlapPatchEmbed(c1, c2, kernel=3, stride=2)
        self.stage2 = nn.ModuleList([
            _SegFormerBlock(c2, n_heads=n_heads[0], sr_ratio=sr_ratios[0],
                            mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i])
            for i in range(depths[0])
        ])
        self.norm2 = nn.LayerNorm(c2)

        # Stage 3: 1/2 -> 1/4 spatial, c2 -> c3 channels.
        self.patch3 = _OverlapPatchEmbed(c2, c3, kernel=3, stride=2)
        self.stage3 = nn.ModuleList([
            _SegFormerBlock(c3, n_heads=n_heads[1], sr_ratio=sr_ratios[1],
                            mlp_ratio=mlp_ratio, drop=drop_rate,
                            drop_path=dpr[depths[0] + i])
            for i in range(depths[1])
        ])
        self.norm3 = nn.LayerNorm(c3)

        # Stage 4: 1/4 -> 1/8 spatial, c3 -> c4 channels (the bottleneck).
        self.patch4 = _OverlapPatchEmbed(c3, c4, kernel=3, stride=2)
        self.stage4 = nn.ModuleList([
            _SegFormerBlock(c4, n_heads=n_heads[2], sr_ratio=sr_ratios[2],
                            mlp_ratio=mlp_ratio, drop=drop_rate,
                            drop_path=dpr[depths[0] + depths[1] + i])
            for i in range(depths[2])
        ])
        self.norm4 = nn.LayerNorm(c4)

    def forward_encoder(self, x):
        # Stem at 1x
        x1 = self.stem(x)

        # Stage 2 at 1/2
        x2_seq, h2, w2 = self.patch2(x1)
        for blk in self.stage2:
            x2_seq = blk(x2_seq, h2, w2)
        x2_seq = self.norm2(x2_seq)
        b = x2_seq.size(0)
        x2 = x2_seq.transpose(1, 2).reshape(b, -1, h2, w2)

        # Stage 3 at 1/4
        x3_seq, h3, w3 = self.patch3(x2)
        for blk in self.stage3:
            x3_seq = blk(x3_seq, h3, w3)
        x3_seq = self.norm3(x3_seq)
        x3 = x3_seq.transpose(1, 2).reshape(b, -1, h3, w3)

        # Stage 4 at 1/8 (bottleneck)
        x4_seq, h4, w4 = self.patch4(x3)
        for blk in self.stage4:
            x4_seq = blk(x4_seq, h4, w4)
        x4_seq = self.norm4(x4_seq)
        x4 = x4_seq.transpose(1, 2).reshape(b, -1, h4, w4)

        return x4, (x1, x2, x3)


class TesseraIoUFusionSegFormerLite(nn.Module):
    """AlphaEarth+Tessera fusion with a SegFormer-Lite transformer encoder
    on the AE side. Decoder mirrors LightUNet's (UpsampleBlock + DoubleConv
    skip-cat) since the encoder outputs the same shapes. Fusion happens at
    the final decoder output (matching canon ae_tessera_gated).
    """
    def __init__(self, n_channels, n_classes=4, alpha_channels=64,
                 tessera_presence_ch=0, tessera_hidden_ch=None,
                 tessera_hidden_depth=0, height_specialist_depth=0,
                 base_ch=32, gate_init_bias=4.0, norm_kind="bn",
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False, disable_head_film=False,
                 drop_rate=0.0, drop_path_rate=0.0,
                 token_channels=0, token_n_heads=8):
        super().__init__()
        if n_classes != 4:
            raise ValueError("SegFormerLite expects 4 output channels")
        if drop_rate > 0.0 or drop_path_rate > 0.0:
            print(f"SegFormerLite AugReg: dropout={drop_rate}, "
                  f"stochastic_depth={drop_path_rate}")
        if n_channels <= alpha_channels:
            raise ValueError("SegFormerLite expects concatenated AE+Tessera input")
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        # SegFormer-Lite encoder for AE
        self.ae_encoder = SegFormerLiteEncoder(alpha_channels, base_ch=base_ch,
                                               norm_kind=norm_kind,
                                               drop_rate=drop_rate,
                                               drop_path_rate=drop_path_rate)

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        # Decoder mirrors LightUNet (BN-based, GeLU is fine too but stick to canon)
        dc_kw = dict(norm_kind=norm_kind)
        self.up1 = UpsampleBlock(c4, c3, norm_kind=norm_kind)
        self.conv1 = DoubleConv(c4, c3, **dc_kw)
        self.up2 = UpsampleBlock(c3, c2, norm_kind=norm_kind)
        self.conv2 = DoubleConv(c3, c2, **dc_kw)
        self.up3 = UpsampleBlock(c2, c1, norm_kind=norm_kind)
        self.conv3 = DoubleConv(c2, c1, **dc_kw)

        # Tessera side: same compression stem as canon, fuse at final layer.
        self.tessera_feature_stem = TesseraCompressionStem(
            tessera_channels, out_ch=base_ch,
            hidden_ch=tessera_hidden_ch, hidden_depth=tessera_hidden_depth,
        )
        self.gate_conv = _build_fusion_gate(
            base_ch, mode="simple", untied=False, init_bias=gate_init_bias,
        )
        self.tessera_presence_ch = int(tessera_presence_ch)
        self.presence_extra_proj = (
            nn.Conv2d(base_ch, self.tessera_presence_ch, 1)
            if self.tessera_presence_ch > 0 else None
        )

        self.head = MultiTaskPredictionHead(
            in_ch=base_ch, out_channels=n_classes,
            presence_extra_ch=self.tessera_presence_ch,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
            disable_head_film=disable_head_film,
        )

        # Multi-source token fusion (e.g. TerraMind S1 — best building feature):
        # cross-attention at the bottleneck. Queries = SegFormer bottleneck x4
        # (c4 ch); Keys/Values = projected 16x16 token grid. Zero-init output proj
        # + sigmoid(-4) gate => starts EXACTLY as the no-token model and only
        # borrows token signal when it helps (the pattern that made TM_S2 work
        # on the conv backbone).
        self.token_channels = int(token_channels)
        if self.token_channels > 0:
            self.token_proj = nn.Conv2d(self.token_channels, c4, 1)
            self.token_q_norm = nn.LayerNorm(c4)
            self.token_kv_norm = nn.LayerNorm(c4)
            self.token_xattn = nn.MultiheadAttention(c4, token_n_heads, batch_first=True)
            self.token_out_proj = nn.Conv2d(c4, c4, 1)
            nn.init.zeros_(self.token_out_proj.weight)
            nn.init.zeros_(self.token_out_proj.bias)
            self.token_gate = nn.Parameter(torch.full((1, c4, 1, 1), -4.0))
        else:
            self.token_proj = None

    def forward(self, x, return_aux=False):
        if isinstance(x, (tuple, list)):
            pixel_x, token = x
        else:
            pixel_x, token = x, None
        alpha = pixel_x[:, :self.alpha_channels, :, :]
        tessera = pixel_x[:, self.alpha_channels:, :, :]

        # Encode AE via SegFormer-Lite
        x4, (x1, x2, x3) = self.ae_encoder.forward_encoder(alpha)

        # Multi-source token cross-attention at the bottleneck (zero-init gated).
        if self.token_proj is not None and token is not None:
            b, c, h4, w4 = x4.shape
            t = self.token_proj(token)                                  # [B, c4, 16, 16]
            t_seq = self.token_kv_norm(t.flatten(2).transpose(1, 2))    # [B, 256, c4]
            q_seq = self.token_q_norm(x4.flatten(2).transpose(1, 2))    # [B, H4*W4, c4]
            attn_out, _ = self.token_xattn(q_seq, t_seq, t_seq)
            attn_map = attn_out.transpose(1, 2).reshape(b, c, h4, w4)
            x4 = x4 + torch.sigmoid(self.token_gate) * self.token_out_proj(attn_map)

        # Decode (LightUNet-style)
        x = self.up1(x4)
        x = self.conv1(torch.cat([x3, x], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x2, x], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x1, x], dim=1))
        ae_feat = x  # [B, base_ch, H, W]

        # Tessera + final-layer fusion (canon pattern)
        tessera_feat = self.tessera_feature_stem(tessera)
        fused = _apply_fusion_gate(self.gate_conv, ae_feat, tessera_feat,
                                   untied=False, mode="simple")
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        return self.head(fused, return_aux=return_aux,
                         presence_extra=presence_extra)


class MultiBackboneFusion(nn.Module):
    """Fuse a from-scratch LightUNet (primary) with a PRETRAINED remote-sensing
    backbone (timm ResNet50 body, Sentinel-2 DINO weights) via a zero-init gate.

    The pretrained branch contributes ~0 at init (sigmoid(-4) ≈ 0.018 gate on a
    zero-init 1x1 projection), so the model starts EXACTLY as the primary
    LightUNet baseline and can only learn to borrow the pretrained features if
    they help. The primary branch consumes the full AE+Tessera concat (n_channels
    = 192); the pretrained branch also consumes all 192 channels (input stem is
    re-initialized to accept 192 channels — the checkpoint stem is intentionally
    dropped via strict=False).
    """

    def __init__(self, n_channels, n_classes=4, alpha_channels=64, base_ch=48,
                 tessera_presence_ch=0, height_specialist_depth=0,
                 norm_kind="bn", gate_init_bias=4.0,
                 height_gate_source="alpha", height_hidden_ch=None,
                 height_trunk_depth=2, height_independent_branches=False,
                 height_head_kind="linear", height_n_bins=64,
                 height_bin_max_m=80.0,
                 presence_head_kind="shared", presence_head_depth=1,
                 presence_branch_ch=None, bidirectional_ctask=False,
                 height_blend_mode="presence_gated",
                 dual_presence=False, disable_head_film=False,
                 pretrained_backbone_path=None,
                 backbone_input_proj_ch=None,
                 backbone_input_norm=None,
                 backbone_pretrained_source=None,
                 freeze_backbone_stages=0,
                 **head_kwargs):
        super().__init__()
        if n_classes != 4:
            raise ValueError("MultiBackboneFusion assumes 4 output channels")
        import timm  # local import: only needed for this model
        self.supports_aux_outputs = True
        self.n_channels = int(n_channels)

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        dc_kw = dict(norm_kind=norm_kind)

        # --- Best-shot pretrained transfer config. ---
        # When backbone_input_proj_ch is set, an input adapter maps the full
        # 192-ch input down to proj_ch channels, and the timm ResNet50 is built
        # with in_chans=proj_ch so the PRETRAINED stem (conv1) is KEPT/loaded.
        # backbone_pretrained_source takes precedence over the legacy
        # pretrained_backbone_path arg when set.
        self.backbone_input_proj_ch = (
            int(backbone_input_proj_ch) if backbone_input_proj_ch is not None else None
        )
        self.backbone_input_norm_kind = backbone_input_norm
        if backbone_pretrained_source is not None:
            _bb_source = backbone_pretrained_source
        elif pretrained_backbone_path is not None:
            _bb_source = pretrained_backbone_path
        else:
            _bb_source = None
        # in_chans the backbone is built with: proj_ch when adapter is used,
        # else the full input width (current/default behavior, stem re-init).
        bb_in_chans = (
            self.backbone_input_proj_ch
            if self.backbone_input_proj_ch is not None
            else self.n_channels
        )

        # --- PRIMARY branch: compact LightUNet on the full 192-ch input. ---
        # Encoder: DoubleConv + MaxPool downsamples; decoder: UpsampleBlock +
        # DoubleConv with skip-cats (mirrors TesseraIoUFusionSegFormerLite).
        self.p_inc = DoubleConv(self.n_channels, c1, **dc_kw)
        self.p_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, **dc_kw))
        self.p_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, **dc_kw))
        self.p_down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4, **dc_kw))
        self.p_up1 = UpsampleBlock(c4, c3, norm_kind=norm_kind)
        self.p_conv1 = DoubleConv(c4, c3, **dc_kw)
        self.p_up2 = UpsampleBlock(c3, c2, norm_kind=norm_kind)
        self.p_conv2 = DoubleConv(c3, c2, **dc_kw)
        self.p_up3 = UpsampleBlock(c2, c1, norm_kind=norm_kind)
        self.p_conv3 = DoubleConv(c2, c1, **dc_kw)

        # --- INPUT ADAPTER (optional): 192 -> proj_ch so the pretrained stem
        # is KEPT. Conv(192,64,3) -> GroupNorm -> GELU -> Conv(64, proj_ch, 1). ---
        if self.backbone_input_proj_ch is not None:
            self.bb_input_adapter = nn.Sequential(
                nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(64), 64),
                nn.GELU(),
                nn.Conv2d(64, self.backbone_input_proj_ch, kernel_size=1),
            )
        else:
            self.bb_input_adapter = None

        # --- INPUT NORMALIZATION (optional): match pretrained input stats. ---
        # "imagenet": subtract/divide fixed RGB mean/std (only valid proj_ch==3).
        # "instance": per-sample InstanceNorm over proj_ch (valid any proj_ch).
        self.bb_input_instancenorm = None
        if backbone_input_norm == "imagenet":
            if bb_in_chans != 3:
                raise ValueError(
                    "backbone_input_norm='imagenet' requires backbone_input_proj_ch==3, "
                    f"got effective in_chans={bb_in_chans}"
                )
            self.register_buffer(
                "bb_input_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "bb_input_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )
        elif backbone_input_norm == "instance":
            self.bb_input_instancenorm = nn.InstanceNorm2d(
                bb_in_chans, affine=False, track_running_stats=False
            )
        elif backbone_input_norm is not None:
            raise ValueError(
                f"Unknown backbone_input_norm={backbone_input_norm!r}; "
                "expected None, 'imagenet', or 'instance'."
            )

        # --- PRETRAINED branch: timm ResNet50 body (features_only). ---
        # When the adapter is used, in_chans=proj_ch so the pretrained stem
        # (conv1) is KEPT. With None, in_chans=n_channels re-inits the stem.
        # source: "imagenet" -> timm pretrained=True; a path -> strict=False
        # full state_dict load; None -> random init.
        timm_pretrained = (_bb_source == "imagenet")
        self.bb = timm.create_model(
            "resnet50", in_chans=bb_in_chans, num_classes=0,
            features_only=True, out_indices=(1, 2, 3, 4),
            pretrained=timm_pretrained,
        )
        bb_chs = self.bb.feature_info.channels()  # [256, 512, 1024, 2048]
        if timm_pretrained:
            print(
                f"MultiBackboneFusion: loaded ImageNet pretrained resnet50 stem "
                f"(in_chans={bb_in_chans}, conv1 kept)."
            )
        elif _bb_source is not None:
            sd = torch.load(_bb_source, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            result = self.bb.load_state_dict(sd, strict=False)
            n_total = len(sd)
            n_missing = len(result.missing_keys)
            n_loaded = n_total - len(result.unexpected_keys)
            bb_keys = set(self.bb.state_dict().keys())
            missing_set = set(result.missing_keys)
            conv1_keys = [k for k in bb_keys if "conv1.weight" in k and "layer" not in k]
            conv1_loaded = bool(conv1_keys) and all(
                k not in missing_set for k in conv1_keys
            )
            print(
                f"MultiBackboneFusion: loaded pretrained backbone from "
                f"{_bb_source}: {n_total} tensors in checkpoint, "
                f"{n_loaded} matched into backbone, {n_missing} backbone keys "
                f"missing. Stem conv1 loaded (kept): {conv1_loaded} "
                f"(in_chans={bb_in_chans})."
            )

        # UNet-style decoder for the pretrained branch: upsample 2048-ch deepest
        # map back toward full resolution, fusing the shallower stages, then
        # project to base_ch. Feature strides are 4/8/16/32, so we upsample
        # 32->16->8->4 with skip-fusion, then a final 4x upsample to full res.
        self.bb_up1 = UpsampleBlock(bb_chs[3], bb_chs[2], norm_kind=norm_kind)  # /32 -> /16
        self.bb_conv1 = DoubleConv(bb_chs[2] * 2, bb_chs[2], **dc_kw)
        self.bb_up2 = UpsampleBlock(bb_chs[2], bb_chs[1], norm_kind=norm_kind)  # /16 -> /8
        self.bb_conv2 = DoubleConv(bb_chs[1] * 2, bb_chs[1], **dc_kw)
        self.bb_up3 = UpsampleBlock(bb_chs[1], bb_chs[0], norm_kind=norm_kind)  # /8 -> /4
        self.bb_conv3 = DoubleConv(bb_chs[0] * 2, bb_chs[0], **dc_kw)
        # /4 -> /2 -> /1, projecting down to base_ch.
        self.bb_up4 = UpsampleBlock(bb_chs[0], base_ch * 2, norm_kind=norm_kind)
        self.bb_up5 = UpsampleBlock(base_ch * 2, base_ch, norm_kind=norm_kind)
        self.bb_out = DoubleConv(base_ch, base_ch, **dc_kw)

        # --- FUSION: zero-init projection + sigmoid(-4) per-channel gate. ---
        self.bb_proj = nn.Conv2d(base_ch, base_ch, 1)
        nn.init.zeros_(self.bb_proj.weight)
        nn.init.zeros_(self.bb_proj.bias)
        self.bb_gate = nn.Parameter(torch.full((1, base_ch, 1, 1), -4.0))

        # --- FREEZE early pretrained stages to PRESERVE spatial knowledge. ---
        # Freeze in order: stem (conv1+bn1) + layer1 + layer2 + ... up to
        # `freeze_backbone_stages` groups. Adapter / late layers / decoder /
        # gate / head remain trainable.
        self.freeze_backbone_stages = int(freeze_backbone_stages)
        self._frozen_bb_modules = []
        if self.freeze_backbone_stages > 0:
            # timm features_only resnet exposes conv1/bn1/act1/maxpool + layer1..4
            freeze_groups = ["__stem__", "layer1", "layer2", "layer3", "layer4"]
            to_freeze = freeze_groups[: self.freeze_backbone_stages]
            stem_attrs = ("conv1", "bn1", "act1", "maxpool")
            for group in to_freeze:
                if group == "__stem__":
                    for attr in stem_attrs:
                        mod = getattr(self.bb, attr, None)
                        if mod is not None:
                            for prm in mod.parameters():
                                prm.requires_grad = False
                            self._frozen_bb_modules.append(mod)
                else:
                    mod = getattr(self.bb, group, None)
                    if mod is not None:
                        for prm in mod.parameters():
                            prm.requires_grad = False
                        self._frozen_bb_modules.append(mod)
            # Freezing requires_grad does NOT stop BN running-stat updates
            # (they are buffers). Put frozen groups in eval() so their BN uses
            # the pretrained running stats and never accumulates batch stats.
            # train() is overridden below to keep them in eval permanently.
            for mod in self._frozen_bb_modules:
                mod.eval()
            bb_frozen = sum(
                p.numel() for p in self.bb.parameters() if not p.requires_grad
            )
            bb_trainable = sum(
                p.numel() for p in self.bb.parameters() if p.requires_grad
            )
            print(
                f"MultiBackboneFusion: froze {self.freeze_backbone_stages} backbone "
                f"group(s) {to_freeze}: {bb_frozen} params frozen, "
                f"{bb_trainable} backbone params trainable "
                f"(adapter/decoder/gate/head stay trainable)."
            )

        # --- HEAD (identical contract to the SegFormer model). ---
        self.tessera_presence_ch = int(tessera_presence_ch)
        self.head = MultiTaskPredictionHead(
            in_ch=base_ch, out_channels=n_classes,
            presence_extra_ch=self.tessera_presence_ch,
            height_specialist_depth=height_specialist_depth,
            height_gate_source=height_gate_source,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_independent_branches=height_independent_branches,
            height_head_kind=height_head_kind,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            presence_head_kind=presence_head_kind,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            bidirectional_ctask=bidirectional_ctask,
            height_blend_mode=height_blend_mode,
            dual_presence=dual_presence,
            disable_head_film=disable_head_film,
        )

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen pretrained backbone groups in eval mode so their BN uses
        # (and never updates/corrupts) the pretrained running statistics.
        for mod in getattr(self, "_frozen_bb_modules", []):
            mod.eval()
        return self

    def _primary_features(self, x):
        x1 = self.p_inc(x)
        x2 = self.p_down1(x1)
        x3 = self.p_down2(x2)
        x4 = self.p_down3(x3)
        u = self.p_up1(x4)
        u = self.p_conv1(torch.cat([x3, u], dim=1))
        u = self.p_up2(u)
        u = self.p_conv2(torch.cat([x2, u], dim=1))
        u = self.p_up3(u)
        u = self.p_conv3(torch.cat([x1, u], dim=1))
        return u  # (B, base_ch, H, W)

    def _adapt_input(self, x):
        """Map the 192-ch input to the backbone's expected in_chans, applying
        the optional input adapter then optional per-channel normalization."""
        if self.bb_input_adapter is not None:
            x = self.bb_input_adapter(x)
        if getattr(self, "bb_input_instancenorm", None) is not None:
            x = self.bb_input_instancenorm(x)
        elif hasattr(self, "bb_input_mean"):
            x = (x - self.bb_input_mean) / self.bb_input_std
        return x

    def _backbone_features(self, x):
        # The pretrained backbone branch is numerically fragile under fp16
        # autocast: extreme embedding activations overflow its BatchNorm,
        # corrupting running_mean/running_var (buffers, updated every forward
        # regardless of requires_grad) -> nan features at eval time -> the
        # additive gated fusion propagates nan everywhere (Val=0). Run the
        # whole branch in fp32 so BN stats stay finite. The primary LightUNet
        # path + head keep their fp16 speedup.
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            return self._backbone_features_fp32(x)

    def _backbone_features_fp32(self, x):
        H, W = x.shape[-2:]
        bb_in = self._adapt_input(x)
        f1, f2, f3, f4 = self.bb(bb_in)  # strides 4, 8, 16, 32
        u = self.bb_up1(f4)
        if u.shape[-2:] != f3.shape[-2:]:
            u = F.interpolate(u, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        u = self.bb_conv1(torch.cat([f3, u], dim=1))
        u = self.bb_up2(u)
        if u.shape[-2:] != f2.shape[-2:]:
            u = F.interpolate(u, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u = self.bb_conv2(torch.cat([f2, u], dim=1))
        u = self.bb_up3(u)
        if u.shape[-2:] != f1.shape[-2:]:
            u = F.interpolate(u, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u = self.bb_conv3(torch.cat([f1, u], dim=1))
        u = self.bb_up4(u)
        u = self.bb_up5(u)
        if u.shape[-2:] != (H, W):
            u = F.interpolate(u, size=(H, W), mode="bilinear", align_corners=False)
        return self.bb_out(u)  # (B, base_ch, H, W)

    def forward(self, x, return_aux=False):
        if isinstance(x, (tuple, list)):
            x = x[0]
        primary_feat = self._primary_features(x)
        bb_feat = self._backbone_features(x)
        fused = primary_feat + torch.sigmoid(self.bb_gate) * self.bb_proj(bb_feat)
        return self.head(fused, return_aux=return_aux, presence_extra=None)
