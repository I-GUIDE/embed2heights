"""
Blend multiple predictions/ directories into one, writing [4, H, W] .npy files
to an output directory. Decoupled from evaluation — the output dir is a
self-contained predictions/ folder that any downstream tool
(sweep_thresholds.py, submission builder) can consume.

Two methods:

  1. mean — simple per-pixel average across N input dirs.
     python tools/ensemble.py mean \\
         --inputs runs/a/predictions runs/b/predictions runs/c/predictions \\
         --output-dir runs/ens_abc/predictions

  2. weighted — per-channel weighted blend via a JSON spec. Useful when
     different models are strongest on different channels.
     python tools/ensemble.py weighted \\
         --spec configs/weighted_v1.json \\
         --output-dir runs/ens_v1/predictions

     Spec JSON format:
       {
         "inputs": {
           "w18":       "runs/alphaearth_hrnet_w18.../predictions",
           "lightunet": "runs/lightunet_alphaearth/predictions",
           "refiner":   "runs/alphaearth_refiner.../predictions"
         },
         "channels": {
           "0": {"lightunet": 0.45, "refiner": 0.30, "w18": 0.25},
           "1": {"w18": 0.45, "refiner": 0.35, "lightunet": 0.20},
           "2": {"lightunet": 0.50, "w18": 0.30, "refiner": 0.20},
           "3": {"w18": 0.50, "refiner": 0.35, "lightunet": 0.15}
         }
       }

Channel-0..2 outputs are clipped to [0, 1]; channel-3 (height) is clipped to
[0, +inf). File pairing uses normalized core_ids — all input dirs must
cover the same set of ids (the tool fails loudly if any id is missing).
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.dataset import normalize_core_id  # noqa: E402


def index_dir(pred_dir):
    """Return {core_id: file_path} for all .npy files in a predictions dir."""
    files = glob.glob(str(Path(pred_dir) / "*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")
    return {normalize_core_id(p): p for p in files}


def common_ids(indexed_dirs):
    """Intersect core_id sets across all indexed dirs; error if any dir is partial."""
    id_sets = [set(d.keys()) for d in indexed_dirs.values()]
    common = set.intersection(*id_sets)
    for name, d in indexed_dirs.items():
        missing = common.symmetric_difference(d.keys())
        if missing:
            extra = d.keys() - common
            gap   = common - d.keys()
            if gap or extra:
                raise ValueError(
                    f"Input '{name}' has a different id set than others. "
                    f"missing {len(gap)}, extra {len(extra)} (e.g. "
                    f"missing: {sorted(gap)[:3]}, extra: {sorted(extra)[:3]})"
                )
    return sorted(common)


def blend_mean(indexed_dirs, ids, output_dir):
    for cid in tqdm(ids, desc="Blending (mean)"):
        arrs = [np.load(indexed_dirs[name][cid]).astype(np.float32) for name in indexed_dirs]
        out = np.mean(arrs, axis=0).astype(np.float32)
        out[:3] = np.clip(out[:3], 0.0, 1.0)
        out[3]  = np.maximum(out[3], 0.0)
        np.save(output_dir / f"{cid}.npy", out)


def blend_weighted(indexed_dirs, channel_weights, ids, output_dir):
    for cid in tqdm(ids, desc="Blending (weighted_channels)"):
        loaded = {name: np.load(p[cid]).astype(np.float32) for name, p in indexed_dirs.items()}
        shape  = next(iter(loaded.values())).shape
        out = np.zeros(shape, dtype=np.float32)
        for ch_str, weights in channel_weights.items():
            ch = int(ch_str)
            total = sum(weights.values())
            for name, w in weights.items():
                if name not in loaded:
                    raise KeyError(f"Spec references '{name}' not in inputs dict")
                out[ch] += loaded[name][ch] * (w / total)
        out[:3] = np.clip(out[:3], 0.0, 1.0)
        out[3]  = np.maximum(out[3], 0.0)
        np.save(output_dir / f"{cid}.npy", out)


def cmd_mean(args):
    if len(args.inputs) < 2:
        raise ValueError("--inputs needs at least 2 directories for a mean ensemble")
    indexed = {f"input_{i}": index_dir(d) for i, d in enumerate(args.inputs)}
    ids = common_ids(indexed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blend_mean(indexed, ids, args.output_dir)
    print(f"Wrote {len(ids)} blended files to {args.output_dir} (mean of {len(args.inputs)} inputs)")


def cmd_weighted(args):
    spec = json.loads(args.spec.read_text())
    if "inputs" not in spec or "channels" not in spec:
        raise ValueError("Spec JSON must have top-level 'inputs' and 'channels' keys.")
    indexed = {name: index_dir(path) for name, path in spec["inputs"].items()}
    ids = common_ids(indexed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blend_weighted(indexed, spec["channels"], ids, args.output_dir)
    print(f"Wrote {len(ids)} blended files to {args.output_dir} (weighted channels from {args.spec})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="method", required=True)

    pm = sub.add_parser("mean", help="Per-pixel mean across N input dirs")
    pm.add_argument("--inputs", type=Path, nargs="+", required=True,
                    help="Two or more predictions/ directories.")
    pm.add_argument("--output-dir", type=Path, required=True)
    pm.set_defaults(func=cmd_mean)

    pw = sub.add_parser("weighted", help="Per-channel weighted blend from a JSON spec")
    pw.add_argument("--spec", type=Path, required=True, help="JSON spec file (see module docstring).")
    pw.add_argument("--output-dir", type=Path, required=True)
    pw.set_defaults(func=cmd_weighted)

    return p.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
