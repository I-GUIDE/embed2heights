"""Active competition model registry.

New experiments should use the short strategy names below. Very old model
variants are recorded in logs rather than kept as live code.
"""

from .backbones import LightUNet
from .pixel_fusion import (
    SimpleConcatASPP,
    SimpleConcatConvNeXt,
    SimpleConcatFusion,
    SimpleGatedFusion,
    TesseraCrossAttnLightUNet,
    TesseraIoUFusionGatedLightUNet,
    TesseraIoUFusionLightUNet,
)
from .token_fusion import TesseraTokenCrossLevelFusionLightUNet, HierarchicalGatedFusion


ACTIVE_MODEL_ALIASES = {
    "ae_only": "lightunet",
    # ae_tessera: the proven un-gated architecture (Tessera → presence head only)
    "ae_tessera": "tessera_iou_fusion",
    "ae_tessera_gated": "tessera_iou_fusion_gated",
    "ae_tessera_crossattn": "tessera_crossattn_bottleneck",
    "xfusion_crosslevel": "tessera_token_crosslevel_s2_decoder64_presence_3way_deep",
    # Per-pixel sigmoid-gated token fusion: model learns WHERE to borrow tokens
    "xfusion_pp": "tessera_token_crosslevel_s2_decoder64_perpixel",
    # Hierarchical output-level gated fusion: Stage 1 (AE+Tessera gated UNet)
    # → 4-channel logits. Stage 2 (token branch) → 4-channel logits. Per-pixel
    # sigmoid gate between them, initialized so Stage 1 dominates at start.
    "hier_gated": "hierarchical_gated_token_fusion",
    # Lightweight per-pixel fusion: no UNet, ~16% params of gated.
    "ae_tessera_simple": "simple_concat_fusion",
    # Hybrid: lightweight trunk + our gated mixing.
    "ae_tessera_simple_gated": "simple_gated_fusion",
    # Lightweight with ConvNeXt trunk (7x7 receptive field for buildings).
    "ae_tessera_simple_convnext": "simple_concat_convnext",
    # Lightweight with ASPP for multi-scale building context.
    "ae_tessera_simple_aspp": "simple_concat_aspp",
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
                       bidirectional_ctask=False,
                       crossattn_n_heads=4,
                       height_blend_mode="presence_gated",
                       dual_presence=False,
                       ae_only_supervision=False,
                       use_se=False,
                       disable_head_film=False):
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

    if selected == "tessera_iou_fusion":
        return (
            TesseraIoUFusionLightUNet(
                n_channels=n_channels,
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
                norm_kind=lightunet_norm_kind,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
                bidirectional_ctask=bidirectional_ctask,
                height_blend_mode=height_blend_mode,
                dual_presence=dual_presence,
            ),
            selected,
        )

    if selected == "tessera_iou_fusion_gated":
        # n_channels may be int (pixel-only) or (pixel_channels, token_channels)
        # when --token-train-embeddings-dir is supplied. Route accordingly.
        if isinstance(n_channels, (tuple, list)):
            _pixel_channels, _token_channels = n_channels
        else:
            _pixel_channels, _token_channels = n_channels, 0
        return (
            TesseraIoUFusionGatedLightUNet(
                n_channels=_pixel_channels,
                n_classes=n_classes,
                token_channels=_token_channels,
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
                bidirectional_ctask=bidirectional_ctask,
                height_blend_mode=height_blend_mode,
                dual_presence=dual_presence,
                ae_only_supervision=ae_only_supervision,
                use_se=use_se,
            ),
            selected,
        )

    if selected == "simple_concat_fusion":
        return (
            SimpleConcatFusion(
                n_channels=n_channels,
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
                bidirectional_ctask=bidirectional_ctask,
                height_blend_mode=height_blend_mode,
                dual_presence=dual_presence,
            ),
            selected,
        )

    if selected == "simple_concat_convnext":
        return (
            SimpleConcatConvNeXt(
                n_channels=n_channels,
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
                bidirectional_ctask=bidirectional_ctask,
                height_blend_mode=height_blend_mode,
                dual_presence=dual_presence,
            ),
            selected,
        )

    if selected == "simple_concat_aspp":
        return (
            SimpleConcatASPP(
                n_channels=n_channels,
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
                bidirectional_ctask=bidirectional_ctask,
                height_blend_mode=height_blend_mode,
                dual_presence=dual_presence,
            ),
            selected,
        )

    if selected == "simple_gated_fusion":
        return (
            SimpleGatedFusion(
                n_channels=n_channels,
                n_classes=n_classes,
                gate_init_bias=gate_init_bias,
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
            ),
            selected,
        )

    if selected == "tessera_crossattn_bottleneck":
        return (
            TesseraCrossAttnLightUNet(
                n_channels=n_channels,
                n_classes=n_classes,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                n_heads=crossattn_n_heads,
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

    if selected == "tessera_token_crosslevel_s2_decoder64_perpixel":
        # Same as xfusion_crosslevel but with per-pixel gate: model learns
        # WHERE to borrow tokens. Safe init (gate≈0) so it starts as the
        # AE+Tessera baseline and only borrows when helpful.
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "xfusion_pp expects n_channels=(pixel_channels, token_channels)"
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
                per_pixel_gate=True,
            ),
            selected,
        )

    if selected == "hierarchical_gated_token_fusion":
        if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
            raise ValueError(
                "hier_gated expects n_channels=(pixel_channels, token_channels)"
            )
        pixel_channels, token_channels = n_channels
        return (
            HierarchicalGatedFusion(
                pixel_channels=pixel_channels,
                token_channels=token_channels,
                n_classes=n_classes,
                # Stage1 kwargs (forwarded to TesseraIoUFusionGatedLightUNet)
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
                norm_kind=lightunet_norm_kind,
                gate_mode=gate_mode,
                gate_untied=gate_untied,
                gate_init_bias=gate_init_bias,
                modality_dropout=modality_dropout,
                presence_head_kind=presence_head_kind,
                presence_head_depth=presence_head_depth,
                presence_branch_ch=presence_branch_ch,
                bidirectional_ctask=bidirectional_ctask,
                height_blend_mode=height_blend_mode,
                dual_presence=dual_presence,
                use_se=use_se,
                disable_head_film=disable_head_film,
            ),
            selected,
        )

    return None
