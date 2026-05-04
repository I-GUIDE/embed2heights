# Model Package Layout

This package is organized by model abstraction level, not by experiment name.

- `blocks.py`: reusable neural network blocks and low-level layers.
- `backbones.py`: feature extractors that produce dense spatial features.
- `heads.py`: prediction heads that convert features into task outputs.
- `pixel_fusion.py`: pixel-aligned multimodal fusion modules and architectures.
- `token_fusion.py`: token-grid fusion modules and three-modal architectures.
- `registry.py`: active strategy aliases and checkpoint-compatible names.
- `factory.py`: `build_model()` and lightweight auto-inference.

Use `from core.models import ...` from training, prediction, and tools. Avoid
importing from individual files unless a diagnostic needs an internal helper.
