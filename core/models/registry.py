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
    FiLMFusionLightUNet,
    GatedPixelFusionAlignedConcatAttentionLightUNet,
    GatedPixelFusionAlignedConcatAuxLightUNet,
    GatedPixelFusionAlignedGroupedConcatAttentionLightUNet,
    GatedPixelFusionAlignedTwoGateAttentionLightUNet,
    GatedPixelFusionBnAttentionUNetLightUNet,
    GatedPixelFusionBottleneckAdaptiveLightUNet,
    GatedPixelFusionCQATwoGateAttentionLightUNet,
    GatedPixelFusionConcatAttentionLightUNet,
    GatedPixelFusionConcatAttentionUNetLightUNet,
    GatedPixelFusionResidualConcatAttentionLightUNet,
    GatedPixelFusionResidualConcatAttentionUNetLightUNet,
    GatedPixelFusionPerModalityLightUNet,
    GatedPixelFusionPerModalityUNetLightUNet,
    GatedPixelFusionTaskRouterLightUNet,
    GatedPixelFusionTerraMindNonWaterLightUNet,
    GatedPixelFusionTwoGateAttentionLightUNet,
    GatedPixelFusionTwoGateAttentionUNetLightUNet,
    GatedPixelFusionTwoGateBnAttentionLightUNet,
    GatedPixelFusionTwoGateGroupedLightUNet,
    SixModalBottleneckLightUNet,
    TesseraTokenCrossLevelFusionLightUNet,
    TokenHeightErrorRouterLightUNet,
    TokenHeightCorrectionLightUNet,
)


ACTIVE_MODEL_ALIASES = {
    "xfusion_film_fusion": "film_fusion_lightunet",
    "ae_only": "lightunet",
    "ae_tessera_gated": "tessera_iou_fusion_gated",
    "ae_tessera_unet_gated": "tessera_unet_gated",
    "ae_tessera_unet_hierarchical": "tessera_unet_hierarchical_gated",
    "xfusion_crosslevel": "tessera_token_crosslevel_s2_decoder64_presence_3way_deep",
    "xfusion_gated_nonwater": "gated_pixel_fusion_terramind_nonwater",
    "xfusion_bottleneck_adaptive": "gated_pixel_fusion_bottleneck_adaptive",
    "xfusion_twogate_attention":      "gated_pixel_fusion_twogate_attention",
    "xfusion_cqa_twogate_attention":  "gated_pixel_fusion_cqa_twogate_attention",
    "xfusion_twogate_aligned_attention": "gated_pixel_fusion_twogate_aligned_attention",
    "xfusion_aligned_concat_attention": "gated_pixel_fusion_aligned_concat_attention",
    "xfusion_aligned_grouped_concat_attention": "gated_pixel_fusion_aligned_grouped_concat_attention",
    "xfusion_aligned_concat_aux": "gated_pixel_fusion_aligned_concat_aux",
    "xfusion_twogate_grouped":        "gated_pixel_fusion_twogate_grouped",
    "xfusion_twogate_bn_attention":   "gated_pixel_fusion_twogate_bn_attention",
    "xfusion_per_modality":              "gated_pixel_fusion_per_modality",
    "xfusion_task_router":               "gated_pixel_fusion_task_router",
    "xfusion_twogate_unet_attention":    "gated_pixel_fusion_twogate_unet_attention",
    "xfusion_per_modality_unet":         "gated_pixel_fusion_per_modality_unet",
    "xfusion_bn_attention_unet":         "gated_pixel_fusion_bn_attention_unet",
    "xfusion_concat_attention":                  "gated_pixel_fusion_concat_attention",
    "xfusion_concat_unet_attention":             "gated_pixel_fusion_concat_unet_attention",
    "xfusion_residual_concat_attention":         "gated_pixel_fusion_residual_concat_attention",
    "xfusion_residual_concat_unet_attention":    "gated_pixel_fusion_residual_concat_unet_attention",
    "xfusion_sixmodal_bottleneck":               "gated_pixel_fusion_sixmodal_bottleneck",
    "xfusion_error_router_height_only":          "gated_pixel_fusion_error_router_height_only",
    "xfusion_error_router_height_aux":           "gated_pixel_fusion_error_router_height_aux",
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
                       token_extractor_mid_ch=None,
                       token_query_grid=32,
                       token_calibration=False,
                       token_gate_init_bias=(2.0, -2.0),
                       token_aux_weight=0.2):
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
                token_query_grid=token_query_grid,
                token_calibration=token_calibration,
                token_gate_init_bias=token_gate_init_bias,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_cqa_twogate_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_cqa_twogate_attention expects "
                "n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionCQATwoGateAttentionLightUNet(
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
                token_query_grid=token_query_grid,
                token_calibration=token_calibration,
                token_gate_init_bias=token_gate_init_bias,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_twogate_aligned_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_twogate_aligned_attention expects "
                "n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionAlignedTwoGateAttentionLightUNet(
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
                token_query_grid=token_query_grid,
                token_calibration=token_calibration,
                token_gate_init_bias=token_gate_init_bias,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_aligned_concat_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_aligned_concat_attention expects "
                "n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionAlignedConcatAttentionLightUNet(
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
                token_query_grid=token_query_grid,
                token_calibration=token_calibration,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_aligned_grouped_concat_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_aligned_grouped_concat_attention expects "
                "n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionAlignedGroupedConcatAttentionLightUNet(
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
                token_query_grid=token_query_grid,
                token_calibration=token_calibration,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_aligned_concat_aux":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_aligned_concat_aux expects "
                "n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionAlignedConcatAuxLightUNet(
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
                token_query_grid=token_query_grid,
                token_calibration=token_calibration,
                token_aux_weight=token_aux_weight,
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

    if selected == "gated_pixel_fusion_per_modality":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_per_modality expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionPerModalityLightUNet(
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
                token_query_grid=token_query_grid,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_task_router":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_task_router expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionTaskRouterLightUNet(
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
                token_query_grid=token_query_grid,
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

    if selected == "gated_pixel_fusion_twogate_unet_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_twogate_unet_attention expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionTwoGateAttentionUNetLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
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
                token_query_grid=token_query_grid,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_concat_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_concat_attention expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionConcatAttentionLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
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
                token_query_grid=token_query_grid,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_concat_unet_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_concat_unet_attention expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionConcatAttentionUNetLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
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
                token_query_grid=token_query_grid,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_residual_concat_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_residual_concat_attention expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionResidualConcatAttentionLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
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
                token_query_grid=token_query_grid,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_residual_concat_unet_attention":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_residual_concat_unet_attention expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionResidualConcatAttentionUNetLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
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
                token_query_grid=token_query_grid,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_per_modality_unet":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_per_modality_unet expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionPerModalityUNetLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
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
                token_query_grid=token_query_grid,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_sixmodal_bottleneck":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_sixmodal_bottleneck expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            SixModalBottleneckLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                alpha_channels=64,
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
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
                modality_dropout=modality_dropout,
                token_calibration=token_calibration,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_error_router_height_only":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_error_router_height_only expects "
                "n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            TokenHeightCorrectionLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                alpha_channels=64,
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

    if selected == "film_fusion_lightunet":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_film_fusion expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            FiLMFusionLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
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
                modality_dropout=modality_dropout,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_error_router_height_aux":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_error_router_height_aux expects "
                "n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            TokenHeightErrorRouterLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                alpha_channels=64,
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

    if selected == "gated_pixel_fusion_bn_attention_unet":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_bn_attention_unet expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionBnAttentionUNetLightUNet(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
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
