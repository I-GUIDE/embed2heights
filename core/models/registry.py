"""Active competition model registry."""

from .backbones import LightUNet
from .pixel_fusion import (
    TesseraIoUFusionGatedLightUNet,
)
from .token_fusion import (
    GatedPixelFusionHybridLightUNet,
    GatedPixelFusionPerSourceEnsembleLightUNet,
)


ACTIVE_MODEL_ALIASES = {
    "ae_only":                       "lightunet",
    "ae_tessera_gated":              "tessera_iou_fusion_gated",
    "xfusion_unet_hybrid_cross_source": "gated_pixel_fusion_hybrid_cross_source",
    "xfusion_unet_per_source_ensemble": "gated_pixel_fusion_per_source_ensemble",
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
                       token_calibration=False,
                       token_calibration_source_indices=None,
                       use_fraction_film=True,
                       use_fraction_aux=None,
                       attn_heads=4,
                       use_additive=True,
                       token_ctx_ch=96,
                       token_proj_depth=1,
                       token_in_source_attn=False,
                       token_cross_source_attn=True,
                       pixel_noise_std=0.0,
                       n_head_replicas=1,
                       symmetric_modality_dropout=0.0,
                       symmetric_modality_dropout_alpha_share=0.5,
                       height_from_pixel=False,
                       feat_aggregation="mean",
                       token_input_clamp=None,
                       pixel_backbone_kind="unet",
                       use_boundary_head=False,
                       presence_tower_depth=0,
                       split_trunk=False,
                presence_trunk_grad_scale=1.0,
                height_trunk_grad_scale=1.0):
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

    if selected == "gated_pixel_fusion_hybrid_cross_source":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_unet_hybrid_cross_source expects "
                "n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionHybridLightUNet(
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
                use_fraction_film=use_fraction_film,
                use_fraction_aux=use_fraction_aux,
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
                token_calibration=token_calibration,
                token_calibration_source_indices=token_calibration_source_indices,
                attn_heads=attn_heads,
                use_additive=use_additive,
                token_ctx_ch=token_ctx_ch,
                token_in_source_attn=token_in_source_attn,
                token_cross_source_attn=token_cross_source_attn,
                pixel_noise_std=pixel_noise_std,
                n_head_replicas=n_head_replicas,
                token_input_clamp=token_input_clamp,
                symmetric_modality_dropout=symmetric_modality_dropout,
                symmetric_modality_dropout_alpha_share=symmetric_modality_dropout_alpha_share,
                pixel_backbone_kind=pixel_backbone_kind,
                use_boundary_head=use_boundary_head,
                presence_tower_depth=presence_tower_depth,
                split_trunk=split_trunk,
                presence_trunk_grad_scale=presence_trunk_grad_scale,
                height_trunk_grad_scale=height_trunk_grad_scale,
            ),
            selected,
        )

    if selected == "gated_pixel_fusion_per_source_ensemble":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_unet_per_source_ensemble expects "
                "n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            GatedPixelFusionPerSourceEnsembleLightUNet(
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
                use_fraction_film=use_fraction_film,
                use_fraction_aux=use_fraction_aux,
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
                token_calibration=token_calibration,
                token_ctx_ch=token_ctx_ch,
                attn_heads=attn_heads,
                use_additive=use_additive,
                token_proj_depth=token_proj_depth,
                token_in_source_attn=token_in_source_attn,
                height_from_pixel=height_from_pixel,
                feat_aggregation=feat_aggregation,
                pixel_noise_std=pixel_noise_std,
                use_boundary_head=use_boundary_head,
                presence_tower_depth=presence_tower_depth,
                split_trunk=split_trunk,
                presence_trunk_grad_scale=presence_trunk_grad_scale,
                height_trunk_grad_scale=height_trunk_grad_scale,
            ),
            selected,
        )

    return None
