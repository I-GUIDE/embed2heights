"""Active model API. The only public entry point is ``build_model``."""

from .factory import build_model, infer_model_type

__all__ = ["build_model", "infer_model_type"]
