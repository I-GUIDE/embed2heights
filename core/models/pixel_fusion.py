import torch
import torch.nn as nn

from .blocks import ChannelCalibration, ConvGNAct, _group_count
from .backbones import LightUNet
from .heads import MultiTaskPredictionHead


def _build_fusion_gate(channels, mode="simple", untied=False, init_bias=4.0):
    n_out = 2 * channels if untied else channels
    if mode == "simple":
        gate = nn.Conv2d(2 * channels, n_out, kernel_size=1)
        final = gate
    elif mode == "rich":
        hidden = max(channels, 32)
        gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, n_out, 1),
        )
        final = gate[-1]
    else:
        raise ValueError(f"Unknown gate mode: {mode!r}")

    nn.init.zeros_(final.weight)
    if untied:
        bias = torch.empty(n_out)
        bias[:channels].fill_(init_bias)
        bias[channels:].fill_(-init_bias)
        final.bias.data.copy_(bias)
    else:
        nn.init.constant_(final.bias, init_bias)
    return gate


def _apply_fusion_gate(gate_module, ae_feat, tes_feat, untied):
    raw = gate_module(torch.cat([ae_feat, tes_feat], dim=1))
    if untied:
        channels = ae_feat.size(1)
        g_ae = torch.sigmoid(raw[:, :channels])
        g_tes = torch.sigmoid(raw[:, channels:])
        return g_ae * ae_feat + g_tes * tes_feat
    gate = torch.sigmoid(raw)
    return gate * ae_feat + (1.0 - gate) * tes_feat


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
                 presence_branch_ch=None):
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
        self.modality_dropout = float(modality_dropout)
        tessera_channels = n_channels - alpha_channels

        self.alpha_unet = LightUNet(
            alpha_channels, n_classes, base_ch=base_ch, norm_kind=norm_kind
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
        )

    def forward(self, x, return_aux=False):
        alpha = x[:, :self.alpha_channels, :, :]
        tessera = x[:, self.alpha_channels:, :, :]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        presence_extra = (
            self.presence_extra_proj(tessera_feat)
            if self.presence_extra_proj is not None else None
        )
        return self.head(fused, return_aux=return_aux,
                         presence_extra=presence_extra)
