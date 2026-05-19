"""Active model API."""

from .registry import (
    ACTIVE_MODEL_ALIASES,
    ACTIVE_MODEL_TYPES,
    build_active_model,
    canonical_model_type,
)
from .factory import build_model, infer_model_type
from .blocks import (
    ASPP,
    HEIGHT_NORM_CONSTANT,
    ChannelCalibration,
    ConvGNAct,
    ConvNeXtBlock,
    _group_count,
)
from .backbones import DoubleConv, LightUNet, UpsampleBlock, _light_norm
from .heads import MultiTaskPredictionHead
from .pixel_fusion import (
    TesseraCompressionStem,
    TesseraIoUFusionGatedLightUNet,
    _apply_fusion_gate,
    _build_fusion_gate,
    _maybe_drop_modality,
)
from .token_fusion import (
    GatedPixelFusionFiLMPerModalityLightUNet,
    GatedPixelFusionTwoGateBnAttentionLightUNet,
)

__all__ = [
    "ACTIVE_MODEL_ALIASES",
    "ACTIVE_MODEL_TYPES",
    "ASPP",
    "ChannelCalibration",
    "ConvGNAct",
    "ConvNeXtBlock",
    "DoubleConv",
    "GatedPixelFusionFiLMPerModalityLightUNet",
    "GatedPixelFusionTwoGateBnAttentionLightUNet",
    "HEIGHT_NORM_CONSTANT",
    "LightUNet",
    "MultiTaskPredictionHead",
    "TesseraCompressionStem",
    "TesseraIoUFusionGatedLightUNet",
    "UpsampleBlock",
    "_apply_fusion_gate",
    "_build_fusion_gate",
    "_group_count",
    "_light_norm",
    "_maybe_drop_modality",
    "build_active_model",
    "build_model",
    "canonical_model_type",
    "infer_model_type",
]
