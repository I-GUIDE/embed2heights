"""Prediction ensembling helpers."""

import json
from pathlib import Path

import numpy as np

from core.data.discovery import normalize_core_id, submission_id


def index_prediction_dir(pred_dir):
    files = sorted(Path(pred_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")
    return {normalize_core_id(path): path for path in files}


def _common_ids(indexed_dirs):
    id_sets = [set(paths) for paths in indexed_dirs.values()]
    common = set.intersection(*id_sets)
    for name, paths in indexed_dirs.items():
        missing = common.symmetric_difference(paths)
        if missing:
            extra = set(paths) - common
            gap = common - set(paths)
            raise ValueError(
                f"Input '{name}' has a different id set. "
                f"missing {len(gap)}, extra {len(extra)}; "
                f"examples missing={sorted(gap)[:3]}, extra={sorted(extra)[:3]}"
            )
    return sorted(common)


def _output_names(indexed_dirs, ids):
    ref = next(iter(indexed_dirs.values()))
    return {cid: submission_id(ref[cid]) for cid in ids}


def load_weighted_ensemble_spec(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    if "inputs" not in spec or "channels" not in spec:
        raise ValueError("Spec JSON must have top-level 'inputs' and 'channels' keys.")
    inputs = {name: Path(path) for name, path in spec["inputs"].items()}
    return inputs, spec["channels"]


def ensemble_mean(input_dirs, output_dir):
    if len(input_dirs) < 2:
        raise ValueError("Mean ensemble requires at least two input directories.")
    indexed = {f"input_{i}": index_prediction_dir(path) for i, path in enumerate(input_dirs)}
    ids = _common_ids(indexed)
    names = _output_names(indexed, ids)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cid in ids:
        arrs = [np.load(indexed[name][cid]).astype(np.float32) for name in indexed]
        out = np.mean(arrs, axis=0).astype(np.float32)
        out[:3] = np.clip(out[:3], 0.0, 1.0)
        out[3] = np.maximum(out[3], 0.0)
        np.save(output_dir / f"{names[cid]}.npy", out)
    return len(ids)


def ensemble_weighted(input_dirs, channel_weights, output_dir):
    indexed = {name: index_prediction_dir(path) for name, path in input_dirs.items()}
    ids = _common_ids(indexed)
    names = _output_names(indexed, ids)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cid in ids:
        loaded = {name: np.load(paths[cid]).astype(np.float32) for name, paths in indexed.items()}
        shape = next(iter(loaded.values())).shape
        out = np.zeros(shape, dtype=np.float32)
        for ch_str, weights in channel_weights.items():
            ch = int(ch_str)
            total = sum(weights.values())
            if total == 0:
                raise ValueError(f"Channel {ch_str} has zero total weight.")
            for name, weight in weights.items():
                if name not in loaded:
                    raise KeyError(f"Spec references '{name}' not in inputs.")
                out[ch] += loaded[name][ch] * (weight / total)
        out[:3] = np.clip(out[:3], 0.0, 1.0)
        out[3] = np.maximum(out[3], 0.0)
        np.save(output_dir / f"{names[cid]}.npy", out)
    return len(ids)
