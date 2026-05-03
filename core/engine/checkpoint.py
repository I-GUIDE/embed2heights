"""Checkpoint save/load helpers."""

import torch


def state_dict_for_save(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_pretrain_weights(model, checkpoint_path, *, strict=False):
    """Load compatible self-supervised weights without touching unmatched heads."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }

    if strict:
        result = model.load_state_dict(state, strict=True)
        print(f"Loaded strict pretrain checkpoint: {checkpoint_path}")
        return result

    target = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in target and target[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    if not compatible:
        raise ValueError(
            f"No compatible tensors found in pretrain checkpoint: {checkpoint_path}"
        )

    result = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded {len(compatible)} compatible tensors from pretrain checkpoint "
        f"({len(skipped)} skipped): {checkpoint_path}"
    )
    if skipped:
        preview = ", ".join(skipped[:8])
        suffix = "..." if len(skipped) > 8 else ""
        print(f"Skipped pretrain-only or incompatible tensors: {preview}{suffix}")
    return result
