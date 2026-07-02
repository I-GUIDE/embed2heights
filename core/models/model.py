"""The one assembled competition model: GatedPixelFusionHybridLightUNet.

This is the top-level architecture that ties the component modules together:
a dual pixel backbone (``backbones.build_pixel_backbone``) merged by a learned
spatial gate (``pixel_fusion``), conditioned on token sources via the
cross-source hybrid fusion (``token_fusion.CrossSourceHybridFiLMFusion``), and
read out by the multi-task head (``heads.MultiTaskPredictionHead``).

It is constructed by name through ``factory.build_model``.
"""

import torch.nn as nn

from .backbones import ChannelCalibration, ConvGNAct, build_pixel_backbone
from .heads import MultiTaskPredictionHead
from .pixel_fusion import _apply_fusion_gate, _build_fusion_gate, _maybe_drop_modality
from .token_fusion import CrossSourceHybridFiLMFusion


class GatedPixelFusionHybridLightUNet(nn.Module):
    """xfusion_085 SoTA: dual-LightUNet pixel backbone + cross-source hybrid token fusion.

    Pixel backbone: symmetric AlphaEarth + Tessera pixel branches merged by a
    learned spatial gate. Token conditioning is CrossSourceHybridFiLMFusion: the
    N token sources are refined via cross-source self-attention, then each refined
    source contributes a zero-init (FiLM gamma/beta + additive A + spatial-gate
    sigma(g)) residual.
    """

    def __init__(self, pixel_channels, token_channels, n_classes=4,
                 alpha_channels=64,
                 height_specialist_depth=0, base_ch=32,
                 gate_init_bias=4.0, gate_mode="simple", gate_untied=False,
                 modality_dropout=0.0,
                 height_hidden_ch=None,
                 height_trunk_depth=2, height_n_bins=64,
                 height_bin_max_m=80.0,
                 use_fraction_aux=True, norm_kind="bn",
                 presence_head_depth=1,
                 presence_branch_ch=None, token_calibration=False,
                 token_ctx_ch=96, attn_heads=4,
                 token_calibration_source_indices=None,
                 pixel_backbone_kind="unet",
                 use_boundary_head=False,
                 presence_tower_depth=0,
                 split_trunk=False,
                 presence_trunk_grad_scale=1.0,
                 height_trunk_grad_scale=1.0,
                 unetpp_bottleneck_attn=False,
                 **unused):
        super().__init__()
        if n_classes != 4:
            raise ValueError(
                "GatedPixelFusionHybridLightUNet assumes 4 output channels"
            )
        self.unetpp_bottleneck_attn = bool(unetpp_bottleneck_attn)
        self.pixel_backbone_kind = (pixel_backbone_kind or "unet").lower()
        if pixel_channels <= alpha_channels:
            raise ValueError(
                "GatedPixelFusionHybridLightUNet expects AlphaEarth+Tessera "
                f"pixel input with >{alpha_channels} channels, got {pixel_channels}"
            )

        self.supports_aux_outputs = True
        self.alpha_channels = alpha_channels
        self.gate_untied = bool(gate_untied)
        self.modality_dropout = float(modality_dropout)
        tessera_channels = pixel_channels - alpha_channels

        self.alpha_unet = build_pixel_backbone(
            self.pixel_backbone_kind, alpha_channels,
            base_ch=base_ch, norm_kind=norm_kind,
            bottleneck_attn=self.unetpp_bottleneck_attn,
        )

        self.tessera_entry = nn.Sequential(
            ChannelCalibration(tessera_channels),
            ConvGNAct(tessera_channels, tessera_channels, kernel_size=1, padding=0),
        )
        self.tessera_unet = build_pixel_backbone(
            self.pixel_backbone_kind, tessera_channels,
            base_ch=base_ch, norm_kind=norm_kind,
            bottleneck_attn=self.unetpp_bottleneck_attn,
        )

        self.gate_conv = _build_fusion_gate(
            base_ch,
            mode=gate_mode,
            untied=self.gate_untied,
            init_bias=gate_init_bias,
        )

        self.hybrid_fusion = CrossSourceHybridFiLMFusion(
            pixel_ch=base_ch,
            token_channels=token_channels,
            ctx_ch=token_ctx_ch,
            token_calibration=token_calibration,
            token_calibration_source_indices=token_calibration_source_indices,
            attn_heads=attn_heads,
        )

        self.head = MultiTaskPredictionHead(
            in_ch=base_ch,
            out_channels=n_classes,
            height_specialist_depth=height_specialist_depth,
            height_hidden_ch=height_hidden_ch,
            height_trunk_depth=height_trunk_depth,
            height_n_bins=height_n_bins,
            height_bin_max_m=height_bin_max_m,
            use_fraction_aux=use_fraction_aux,
            presence_head_depth=presence_head_depth,
            presence_branch_ch=presence_branch_ch,
            use_boundary_head=use_boundary_head,
            presence_tower_depth=presence_tower_depth,
            split_trunk=split_trunk,
            presence_trunk_grad_scale=presence_trunk_grad_scale,
            height_trunk_grad_scale=height_trunk_grad_scale,
        )

    def forward(self, x, return_aux=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "GatedPixelFusionHybridLightUNet expects (pixel, token) input"
            )
        pixel, token = x
        alpha = pixel[:, :self.alpha_channels]
        tessera = pixel[:, self.alpha_channels:]

        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_unet.forward_features(self.tessera_entry(tessera))
        tessera_feat = _maybe_drop_modality(
            tessera_feat, self.modality_dropout, self.training
        )
        F_pixel = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat, untied=self.gate_untied
        )
        fused = self.hybrid_fusion(F_pixel, token)
        return self.head(fused, return_aux=return_aux)
