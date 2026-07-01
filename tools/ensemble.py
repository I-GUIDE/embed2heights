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
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.inference.ensemble import (  # noqa: E402
    ensemble_mean,
    ensemble_weighted,
    load_weighted_ensemble_spec,
)


def cmd_mean(args):
    count = ensemble_mean(args.inputs, args.output_dir)
    print(f"Wrote {count} blended files to {args.output_dir} (mean of {len(args.inputs)} inputs)")


def cmd_weighted(args):
    inputs, channels = load_weighted_ensemble_spec(args.spec)
    count = ensemble_weighted(inputs, channels, args.output_dir)
    print(f"Wrote {count} blended files to {args.output_dir} (weighted channels from {args.spec})")


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
