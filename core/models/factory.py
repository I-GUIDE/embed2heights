"""Build the active competition model by name.

Only one model survives from the competition: the hybrid cross-source U-Net
(:class:`~core.models.model.GatedPixelFusionHybridLightUNet`). ``build_model``
resolves the config-facing name (or ``"auto"``) and forwards every
hyper-parameter straight through to it via **kwargs -- the two ``lightunet_*``
names are renamed to the model's ``base_ch`` / ``norm_kind``, so there is no
per-parameter plumbing to keep in sync.
"""

from .model import GatedPixelFusionHybridLightUNet


# config-facing name -> internal canonical name
ACTIVE_MODEL_ALIASES = {
    "xfusion_unet_hybrid_cross_source": "gated_pixel_fusion_hybrid_cross_source",
}
ACTIVE_MODEL_TYPES = set(ACTIVE_MODEL_ALIASES) | set(ACTIVE_MODEL_ALIASES.values())


def infer_model_type(n_channels):
    """Best-effort default for active experiments (only the hybrid model remains)."""
    return "xfusion_unet_hybrid_cross_source"


def build_model(model_type, n_channels, n_classes, *,
                lightunet_base_ch=32, lightunet_norm_kind="bn", **kwargs):
    """Build the sole active model by name. ``model_type='auto'`` resolves to it.

    Returns (model, canonical_name). All model hyper-parameters flow through
    **kwargs; ``n_channels`` must be ``(pixel_channels, token_channels)``.
    """
    selected = model_type.lower()
    if selected == "auto":
        selected = infer_model_type(n_channels)
    selected = ACTIVE_MODEL_ALIASES.get(selected, selected)

    if selected not in ACTIVE_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. "
            f"Expected one of: {', '.join(sorted(ACTIVE_MODEL_TYPES))}."
        )
    if not isinstance(n_channels, (tuple, list)) or len(n_channels) != 2:
        raise ValueError(
            "xfusion_unet_hybrid_cross_source expects "
            "n_channels=(pixel_channels, token_channels)"
        )

    pixel_channels, token_channels = n_channels
    model = GatedPixelFusionHybridLightUNet(
        pixel_channels=pixel_channels,
        token_channels=token_channels,
        n_classes=n_classes,
        base_ch=lightunet_base_ch,
        norm_kind=lightunet_norm_kind,
        **kwargs,
    )
    return model, selected
