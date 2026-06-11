from .registry import ACTIVE_MODEL_ALIASES, ACTIVE_MODEL_TYPES, build_active_model


def infer_model_type(n_channels):
    """Best-effort default for active experiments."""
    if isinstance(n_channels, (tuple, list)):
        return "xfusion_unet_per_source_ensemble"
    if n_channels > 64:
        return "ae_tessera_gated"
    return "ae_only"


def build_model(model_type, n_channels, n_classes, tessera_presence_ch=16,
                tessera_hidden_ch=None, tessera_hidden_depth=0,
                height_specialist_depth=0, lightunet_base_ch=32,
                height_gate_source="alpha", height_hidden_ch=None,
                height_trunk_depth=2, height_independent_branches=False,
                height_head_kind="linear", height_n_bins=64,
                height_bin_max_m=80.0, lightunet_norm_kind="bn",
                gate_mode="simple", gate_untied=False, gate_init_bias=4.0,
                modality_dropout=0.0, presence_head_kind="shared",
                presence_head_depth=1, presence_branch_ch=None,
                use_fraction_film=True, use_fraction_aux=None,
                attn_heads=4, token_calibration=False, use_additive=True,
                token_calibration_source_indices=None,
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
                presence_detach_trunk=False,
                presence_trunk_grad_scale=1.0):
    selected = model_type.lower()
    if selected == "auto":
        selected = infer_model_type(n_channels)

    active = build_active_model(
        selected,
        n_channels,
        n_classes,
        tessera_presence_ch=tessera_presence_ch,
        tessera_hidden_ch=tessera_hidden_ch,
        tessera_hidden_depth=tessera_hidden_depth,
        height_specialist_depth=height_specialist_depth,
        lightunet_base_ch=lightunet_base_ch,
        height_gate_source=height_gate_source,
        height_hidden_ch=height_hidden_ch,
        height_trunk_depth=height_trunk_depth,
        height_independent_branches=height_independent_branches,
        height_head_kind=height_head_kind,
        height_n_bins=height_n_bins,
        height_bin_max_m=height_bin_max_m,
        lightunet_norm_kind=lightunet_norm_kind,
        gate_mode=gate_mode,
        gate_untied=gate_untied,
        gate_init_bias=gate_init_bias,
        modality_dropout=modality_dropout,
        presence_head_kind=presence_head_kind,
        presence_head_depth=presence_head_depth,
        presence_branch_ch=presence_branch_ch,
        use_fraction_film=use_fraction_film,
        use_fraction_aux=use_fraction_aux,
        attn_heads=attn_heads,
        token_calibration=token_calibration,
        token_calibration_source_indices=token_calibration_source_indices,
        use_additive=use_additive,
        token_ctx_ch=token_ctx_ch,
        token_proj_depth=token_proj_depth,
        token_in_source_attn=token_in_source_attn,
        token_cross_source_attn=token_cross_source_attn,
        pixel_noise_std=pixel_noise_std,
        n_head_replicas=n_head_replicas,
        symmetric_modality_dropout=symmetric_modality_dropout,
        symmetric_modality_dropout_alpha_share=symmetric_modality_dropout_alpha_share,
        height_from_pixel=height_from_pixel,
        feat_aggregation=feat_aggregation,
        token_input_clamp=token_input_clamp,
        pixel_backbone_kind=pixel_backbone_kind,
        use_boundary_head=use_boundary_head,
        presence_tower_depth=presence_tower_depth,
        split_trunk=split_trunk,
                presence_detach_trunk=presence_detach_trunk,
                presence_trunk_grad_scale=presence_trunk_grad_scale,
    )
    if active is not None:
        return active

    aliases = ", ".join(sorted(ACTIVE_MODEL_ALIASES))
    canonical = ", ".join(sorted(ACTIVE_MODEL_TYPES - set(ACTIVE_MODEL_ALIASES)))
    raise ValueError(
        f"Unsupported model_type={model_type!r}. Active aliases: {aliases}. "
        f"Canonical active names accepted for checkpoint compatibility: {canonical}."
    )
