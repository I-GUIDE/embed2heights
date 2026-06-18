"""Active competition model registry.

New experiments should use the short strategy names below. Very old model
variants are recorded in logs rather than kept as live code.
"""

from .backbones import LightUNet
from .pixel_fusion import (
    AeTesseraMlpFusion,
    AeTesseraMoeFusion,
    MultiBackboneFusion,
    SimpleConcatASPP,
    SimpleConcatConvNeXt,
    SimpleConcatFusion,
    SimpleGatedFusion,
    TesseraCrossAttnLightUNet,
    TesseraIoUFusionGatedLightUNet,
    TesseraIoUFusionLightUNet,
    TesseraIoUFusionMultiLevelGatedLightUNet,
    TesseraIoUFusionSegFormerLite,
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
    # MLP-only decoder (no spatial conv) + per-modality InstanceNorm. Tests
    # THOR/TESSERA-paper hypothesis: heavy decoders overfit when embeddings
    # are strong. Targets the OOF→LB generalization gap.
    "ae_tessera_mlp": "ae_tessera_mlp_fusion",
    # Per-pixel Mixture-of-Experts fusion (deep research #1). Top-K=2 routing
    # over 4 lightweight experts; sparse activation regularizes small-data
    # training while expanding effective parameter capacity.
    "ae_tessera_moe": "ae_tessera_moe_fusion",
    # Multi-level cross-modal gated fusion (CMGFNet-style): parallel
    # AE and Tessera encoders, sigmoid gate at EVERY decoder level (not just
    # the final output). Targets the fusion bottleneck — current best fold0
    # arch fuses only at the head; multilevel lets the model decide per-scale.
    "ae_tessera_multilevel": "tessera_iou_fusion_multilevel_gated",
    # SegFormer-Lite encoder swap: replaces LightUNet's conv encoder with a
    # 4-stage hierarchical transformer (efficient self-attn + Mix-FFN), then
    # uses the canon LightUNet decoder + gated Tessera fusion at the head.
    "ae_tessera_segformer": "tessera_iou_fusion_segformer_lite",
    # Multi-backbone fusion: from-scratch LightUNet (primary) + pretrained
    # remote-sensing ResNet50 body, combined via zero-init gate so it starts
    # as the proven primary baseline and only adds the pretrained branch if it
    # helps. Pass --pretrained-backbone-path for the pretrained variant; omit
    # it for the random-init control.
    "multibackbone_fusion": "multi_backbone_fusion",
    "multi_backbone_fusion": "multi_backbone_fusion",
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
                       use_coord_attn=False,
                       use_bottleneck_attn=False,
                       use_mixstyle=False,
                       use_attn_gates=False,
                       use_aspp=False,
                       bottleneck_attn_depth=1,
                       use_modern=False,
                       disable_head_film=False,
                       use_xsource_fusion=False,
                       token_source_ch=768,
                       token_ctx_ch=96,
                       xsource_attn_heads=4,
                       xsource_token_calibration=False,
                       use_spatial_token_film=False,
                       vit_drop_rate=0.0,
                       vit_drop_path_rate=0.0,
                       pretrained_backbone_path=None,
                       backbone_input_proj_ch=None,
                       backbone_input_norm=None,
                       backbone_pretrained_source=None,
                       freeze_backbone_stages=0):
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
                use_coord_attn=use_coord_attn,
                use_bottleneck_attn=use_bottleneck_attn,
                use_mixstyle=use_mixstyle,
                use_attn_gates=use_attn_gates,
                use_aspp=use_aspp,
                bottleneck_attn_depth=bottleneck_attn_depth,
                use_modern=use_modern,
                use_xsource_fusion=use_xsource_fusion,
                token_source_ch=token_source_ch,
                token_ctx_ch=token_ctx_ch,
                xsource_attn_heads=xsource_attn_heads,
                xsource_token_calibration=xsource_token_calibration,
                use_spatial_token_film=use_spatial_token_film,
            ),
            selected,
        )

    if selected == "tessera_iou_fusion_multilevel_gated":
        return (
            TesseraIoUFusionMultiLevelGatedLightUNet(
                n_channels=n_channels,
                n_classes=n_classes,
                base_ch=lightunet_base_ch,
                gate_init_bias=gate_init_bias,
                norm_kind=lightunet_norm_kind,
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
                use_se=use_se,
                use_coord_attn=use_coord_attn,
                use_bottleneck_attn=use_bottleneck_attn,
                use_mixstyle=use_mixstyle,
                use_attn_gates=use_attn_gates,
                disable_head_film=disable_head_film,
            ),
            selected,
        )

    if selected == "tessera_iou_fusion_segformer_lite":
        if isinstance(n_channels, (tuple, list)):
            _seg_pixel_ch, _seg_token_ch = n_channels
        else:
            _seg_pixel_ch, _seg_token_ch = n_channels, 0
        return (
            TesseraIoUFusionSegFormerLite(
                n_channels=_seg_pixel_ch,
                token_channels=_seg_token_ch,
                n_classes=n_classes,
                tessera_presence_ch=tessera_presence_ch,
                tessera_hidden_ch=tessera_hidden_ch,
                tessera_hidden_depth=tessera_hidden_depth,
                height_specialist_depth=height_specialist_depth,
                base_ch=lightunet_base_ch,
                gate_init_bias=gate_init_bias,
                norm_kind=lightunet_norm_kind,
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
                drop_rate=vit_drop_rate,
                drop_path_rate=vit_drop_path_rate,
            ),
            selected,
        )

    if selected == "multi_backbone_fusion":
        # n_channels may be int (pixel-only) or (pixel_channels, token_channels);
        # this model uses only the pixel part.
        if isinstance(n_channels, (tuple, list)):
            _mb_pixel_ch = n_channels[0]
        else:
            _mb_pixel_ch = n_channels
        return (
            MultiBackboneFusion(
                n_channels=_mb_pixel_ch,
                n_classes=n_classes,
                base_ch=lightunet_base_ch,
                tessera_presence_ch=0,
                height_specialist_depth=height_specialist_depth,
                norm_kind=lightunet_norm_kind,
                gate_init_bias=gate_init_bias,
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
                pretrained_backbone_path=pretrained_backbone_path,
                backbone_input_proj_ch=backbone_input_proj_ch,
                backbone_input_norm=backbone_input_norm,
                backbone_pretrained_source=backbone_pretrained_source,
                freeze_backbone_stages=freeze_backbone_stages,
            ),
            selected,
        )

    if selected == "ae_tessera_moe_fusion":
        return (
            AeTesseraMoeFusion(
                n_channels=n_channels,
                n_classes=n_classes,
                num_experts=4, k=2, expert_hidden=128,
                base_ch=lightunet_base_ch,
                norm_kind=lightunet_norm_kind,
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
                use_bottleneck_attn=use_bottleneck_attn,
                use_mixstyle=use_mixstyle,
            ),
            selected,
        )

    if selected == "ae_tessera_mlp_fusion":
        # Use a small fixed hidden width to keep the MLP truly lightweight.
        # Research recommends ~256 hidden; head adds its own params on top.
        return (
            AeTesseraMlpFusion(
                n_channels=n_channels,
                n_classes=n_classes,
                hidden_ch=128,
                height_specialist_depth=height_specialist_depth,
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
                disable_head_film=disable_head_film,
                use_bottleneck_attn=use_bottleneck_attn,
                use_mixstyle=use_mixstyle,
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
                use_coord_attn=use_coord_attn,
                use_bottleneck_attn=use_bottleneck_attn,
                use_mixstyle=use_mixstyle,
                disable_head_film=disable_head_film,
            ),
            selected,
        )

    return None
