"""Active competition model registry.

New experiments should use the short strategy names below. Very old model
variants are recorded in logs rather than kept as live code.
"""

from .backbones import LightUNet
from .pixel_fusion import (
    TesseraIoUFusionGatedLightUNet,
    TesseraUNetGatedLightUNet,
    TesseraUNetHierarchicalGatedLightUNet,
)
from .token_fusion import (
    GatedPixelFusionBottleneckAdaptiveLightUNet,
    GatedPixelFusionTerraMindNonWaterLightUNet,
    GatedPixelFusionTwoGateAttentionLightUNet,
    GatedPixelFusionTwoGateBnAttentionLightUNet,
    GatedPixelFusionTwoGateGroupedLightUNet,
    TesseraTokenCrossLevelFusionLightUNet,
)


ACTIVE_MODEL_ALIASES = {
    "ae_only": "lightunet",
    "ae_tessera_gated": "tessera_iou_fusion_gated",
    "ae_tessera_unet_gated": "tessera_unet_gated",
    "ae_tessera_unet_hierarchical": "tessera_unet_hierarchical_gated",
    "xfusion_crosslevel": "tessera_token_crosslevel_s2_decoder64_presence_3way_deep",
    "xfusion_gated_nonwater": "gated_pixel_fusion_terramind_nonwater",
    "xfusion_bottleneck_adaptive": "gated_pixel_fusion_bottleneck_adaptive",
    "xfusion_twogate_attention":    "gated_pixel_fusion_twogate_attention",
    "xfusion_twogate_grouped":      "gated_pixel_fusion_twogate_grouped",
    "xfusion_twogate_bn_attention": "gated_pixel_fusion_twogate_bn_attention",
}

ACTIVE_MODEL_TYPES = set(ACTIVE_MODEL_ALIASES) | set(ACTIVE_MODEL_ALIASES.values())


def canonical_model_type(model_type):
    selected = model_type.lower()
    return ACTIVE_MODEL_ALIASES.get(selected, selected)


def build_active_model(model_type, n_channels, n_classes, *,
                       tessera_presence_ch=16,
                       tessera_hidden_ch=None,
                       tessera_hidden_depth=0,
                       height_specialist_depth=0,
                       lightunet_base_ch=32,
                       height_gate_source="alpha",
                       height_hidden_ch=None,
                       height_trunk_depth=2,
                       height_independent_branches=False,
                       height_head_kind="linear",
                       height_n_bins=64,
                       height_bin_max_m=80.0,
                       lightunet_norm_kind="bn",
                       gate_mode="simple",
                       gate_untied=False,
                       gate_init_bias=4.0,
                       modality_dropout=0.0,
                       presence_head_kind="shared",
                       presence_head_depth=1,
                       presence_branch_ch=None,
                       use_token_extractor=False,
                       token_extractor_mid_ch=None):
    selected = canonical_model_type(model_type)
    if selected not in ACTIVE_MODEL_TYPES:
        return None

    if selected == "lightunet":
        return (
            LightUNet(
                n_channels,
                n_classes,
                base_ch=lightunet_base_ch,
                norm_kind=lightunet_norm_kind,
            ),
            selected,
        )

    if selected == "tessera_iou_fusion_gated":
        return (
            TesseraIoUFusionGatedLightUNet(
                n_channels=n_channels,
                n_classes=n_classes,
                tessera_presence_ch=tessera_presence_ch,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                gate_mode=gate_mode,
                gate_untied=gate_untied,
                gate_init_bias=gate_init_bias,
                modality_dropout=modality_dropout,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
            ),
            selected,
        )

    if selected == "tessera_unet_gated":
        return (
            TesseraUNetGatedLightUNet(
                n_channels=n_channels,
                n_classes=n_classes,
                tessera_hidden_ch=tessera_hidden_ch,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                gate_mode=gate_mode,
                gate_untied=gate_untied,
                gate_init_bias=gate_init_bias,
                modality_dropout=modality_dropout,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
            ),
            selected,
        )

    if selected == "tessera_unet_hierarchical_gated":
        return (
            TesseraUNetHierarchicalGatedLightUNet(
                n_channels=n_channels,
                n_classes=n_classes,
                tessera_hidden_ch=tessera_hidden_ch,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                gate_mode=gate_mode,
                gate_untied=gate_untied,
                gate_init_bias=gate_init_bias,
                modality_dropout=modality_dropout,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
            ),
            selected,
        )

    if selected == "tessera_token_crosslevel_s2_decoder64_presence_3way_deep":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_crosslevel expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            TesseraTokenCrossLevelFusionLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                tessera_presence_ch=tessera_presence_ch,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                token_fusion_kind="single",
                fusion_points=("decoder64",),
                presence_head_kind="split_all",
                presence_head_depth=2,
                presence_branch_ch=48,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_terramind_nonwater":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_gated_nonwater expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionTerraMindNonWaterLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                tessera_presence_ch=tessera_presence_ch,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                gate_mode=gate_mode,
                gate_untied=gate_untied,
                gate_init_bias=gate_init_bias,
                modality_dropout=modality_dropout,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                norm_kind=lightunet_norm_kind,
                presence_head_kind="split_all",
                presence_head_depth=2,
                presence_branch_ch=48,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_bottleneck_adaptive":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_bottleneck_adaptive expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionBottleneckAdaptiveLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                tessera_presence_ch=tessera_presence_ch,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                gate_mode=gate_mode,
                gate_untied=gate_untied,
                gate_init_bias=gate_init_bias,
                modality_dropout=modality_dropout,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_twogate_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_twogate_attention expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionTwoGateAttentionLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                tessera_presence_ch=tessera_presence_ch,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                gate_mode=gate_mode,
                gate_untied=gate_untied,
                gate_init_bias=gate_init_bias,
                modality_dropout=modality_dropout,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
                use_token_extractor=use_token_extractor,
                token_extractor_mid_ch=token_extractor_mid_ch,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_twogate_grouped":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_twogate_grouped expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionTwoGateGroupedLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                tessera_presence_ch=tessera_presence_ch,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                gate_mode=gate_mode,
                gate_untied=gate_untied,
                gate_init_bias=gate_init_bias,
                modality_dropout=modality_dropout,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
                token_extractor_mid_ch=token_extractor_mid_ch,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_twogate_bn_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_twogate_bn_attention expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionTwoGateBnAttentionLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                tessera_presence_ch=tessera_presence_ch,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                gate_mode=gate_mode,
                gate_untied=gate_untied,
                gate_init_bias=gate_init_bias,
                modality_dropout=modality_dropout,
                height_gate_source=height_gate_source,
                height_hidden_ch=height_hidden_ch,
                height_trunk_depth=height_trunk_depth,
                height_independent_branches=height_independent_branches,
                height_head_kind=height_head_kind,
                height_n_bins=height_n_bins,
                height_bin_max_m=height_bin_max_m,
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
            ),
            selected,
        )

    return None
