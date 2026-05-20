import torch
import torch.nn as nn

import torch.nn.functional as F

from .blocks import ASPP, ChannelCalibration, ConvGNAct, ConvNeXtBlock, _group_count
from .backbones import DoubleConv, LightUNet
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
                 use_bottleneck_attn=False, disable_head_film=False):
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
