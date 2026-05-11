from .registry import ACTIVE_MODEL_ALIASES, ACTIVE_MODEL_TYPES, build_active_model


def infer_model_type(n_channels):
    """Best-effort default for active experiments."""
    if isinstance(n_channels, (tuple, list)):
        return "xfusion_crosslevel"
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
                bidirectional_ctask=False, crossattn_n_heads=4,
                height_blend_mode="presence_gated",
                dual_presence=False,
                ae_only_supervision=False):
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
        bidirectional_ctask=bidirectional_ctask,
        crossattn_n_heads=crossattn_n_heads,
        height_blend_mode=height_blend_mode,
        dual_presence=dual_presence,
        ae_only_supervision=ae_only_supervision,
    )
    if active is not None:
        return active

    aliases = ", ".join(sorted(ACTIVE_MODEL_ALIASES))
    canonical = ", ".join(sorted(ACTIVE_MODEL_TYPES - set(ACTIVE_MODEL_ALIASES)))
    raise ValueError(
        f"Unsupported model_type={model_type!r}. Active aliases: {aliases}. "
        f"Canonical active names accepted for checkpoint compatibility: {canonical}."
    )
