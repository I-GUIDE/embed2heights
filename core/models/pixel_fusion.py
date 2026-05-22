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
                 use_modern=False):
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
        if self.token_channels > 0:
            from .token_fusion import TokenPyramidNeck  # lazy import: avoid circular
            self.token_neck = TokenPyramidNeck(
                self.token_channels,
                level_channels=(base_ch * 2, base_ch, base_ch, base_ch),
            )
            self.token_residual_proj = nn.Conv2d(base_ch, base_ch, 1)
            nn.init.zeros_(self.token_residual_proj.weight)
            nn.init.zeros_(self.token_residual_proj.bias)
            self.token_residual_gate = nn.Parameter(torch.full((1, base_ch, 1, 1), -4.0))
        else:
            self.token_neck = None
            self.token_residual_proj = None
            self.token_residual_gate = None

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

        # Linear token-residual: project token-pyramid features (upsampled to
        # AE resolution) through a 1x1 conv with a learned per-channel sigmoid
        # gate that's zero-init biased to ~0.02. Adds zero contribution at
        # step 0; gradient lets the model learn to incorporate token info.
        if self.token_neck is not None and token is not None:
            tpyr = self.token_neck(token)
            t_feat = tpyr[128]
            if t_feat.shape[-2:] != fused.shape[-2:]:
                t_feat = F.interpolate(
                    t_feat, size=fused.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
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


class _EfficientSelfAttention(nn.Module):
    """SegFormer-style efficient self-attention with spatial reduction in K, V.

    Reduces the K, V sequence length by `sr_ratio` via a strided conv, then
    does standard MHA between full-length Q and reduced K, V. This makes
    attention tractable at high-resolution encoder stages where vanilla MSA
    would OOM (e.g. 16k tokens at 1/2 spatial).
    """
    def __init__(self, dim, n_heads, sr_ratio=1):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
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
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class _SegFormerBlock(nn.Module):
    """Transformer block with efficient self-attention + Mix-FFN (depthwise
    conv inside the MLP, per SegFormer paper). Pre-norm."""
    def __init__(self, dim, n_heads, sr_ratio=1, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _EfficientSelfAttention(dim, n_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def _mlp(self, x_seq, h, w):
        b, n, _ = x_seq.shape
        x = self.fc1(x_seq)
        # Mix-FFN: spatial 3x3 depthwise conv inside the MLP
        x_spatial = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dwconv(x_spatial).flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)

    def forward(self, x_seq, h, w):
        x_seq = x_seq + self.attn(self.norm1(x_seq), h, w)
        x_seq = x_seq + self._mlp(self.norm2(x_seq), h, w)
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
                 mlp_ratio=2.0):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Stem: pure conv at full resolution.
        self.stem = DoubleConv(in_ch, c1, norm_kind=norm_kind)

        # Stage 2: 1x -> 1/2 spatial, c1 -> c2 channels.
        self.patch2 = _OverlapPatchEmbed(c1, c2, kernel=3, stride=2)
        self.stage2 = nn.ModuleList([
            _SegFormerBlock(c2, n_heads=n_heads[0], sr_ratio=sr_ratios[0],
                            mlp_ratio=mlp_ratio)
            for _ in range(depths[0])
        ])
        self.norm2 = nn.LayerNorm(c2)

        # Stage 3: 1/2 -> 1/4 spatial, c2 -> c3 channels.
        self.patch3 = _OverlapPatchEmbed(c2, c3, kernel=3, stride=2)
        self.stage3 = nn.ModuleList([
            _SegFormerBlock(c3, n_heads=n_heads[1], sr_ratio=sr_ratios[1],
                            mlp_ratio=mlp_ratio)
            for _ in range(depths[1])
        ])
        self.norm3 = nn.LayerNorm(c3)

        # Stage 4: 1/4 -> 1/8 spatial, c3 -> c4 channels (the bottleneck).
        self.patch4 = _OverlapPatchEmbed(c3, c4, kernel=3, stride=2)
        self.stage4 = nn.ModuleList([
            _SegFormerBlock(c4, n_heads=n_heads[2], sr_ratio=sr_ratios[2],
                            mlp_ratio=mlp_ratio)
            for _ in range(depths[2])
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
                 dual_presence=False, disable_head_film=False):
        super().__init__()
        if n_classes != 4:
            raise ValueError("SegFormerLite expects 4 output channels")
        if n_channels <= alpha_channels:
            raise ValueError("SegFormerLite expects concatenated AE+Tessera input")
        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        tessera_channels = n_channels - alpha_channels

        # SegFormer-Lite encoder for AE
        self.ae_encoder = SegFormerLiteEncoder(alpha_channels, base_ch=base_ch,
                                               norm_kind=norm_kind)

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

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        # Encode AE via SegFormer-Lite
        x4, (x1, x2, x3) = self.ae_encoder.forward_encoder(alpha)

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
