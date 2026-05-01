"""
Combine predictions from a presence-only run and a height-only run into a
single submission-format prediction directory.

Use case: dual-model SPLIT runs (see logs/SPLIT_TASK_REPORT.md).
  - `<presence_run>` was trained with `--task presence`. Its channel 3 is
    untrained noise; we read only channels 0-2 (building/vegetation/water
    presence probability).
  - `<height_run>` was trained with `--task height`. Its channels 0-2 are
    weakly supervised via the in-head gate but not IoU-trained; we read only
    channel 3 (the gated height in metres). The internal gate uses the
    height-run's own presence_logits (good enough for routing-for-height).

Output: a new pseudo-experiment dir `runs/<output_name>/predictions/` that
behaves identically to a single-model output for downstream tools
(evaluate.py, sweep_thresholds.py, diagnostic_height_rmse.py). No model
checkpoint is written.

Usage:
    python tools/combine_split_predictions.py \\
        --presence-run split1_presence \\
        --height-run   split1_height \\
        --output-name  split1_combined

Both `--{presence,height}-run` are looked up under `runs/`. Predictions
must already exist in `runs/<run>/predictions/`.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))


def _expect_pred_dir(run_name, predictions_subdir):
    pred_dir = os.path.join(REPO_DIR, "runs", run_name, predictions_subdir)
    if not os.path.isdir(pred_dir):
        raise SystemExit(
            f"Missing predictions dir: {pred_dir}\n"
            f"Run predict.py for {run_name} first."
        )
    return pred_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presence-run", required=True,
                    help="Name of the run trained with --task presence "
                         "(under runs/).")
    ap.add_argument("--height-run", required=True,
                    help="Name of the run trained with --task height (under runs/).")
    ap.add_argument("--output-name", required=True,
                    help="Pseudo-experiment name. Output goes to "
                         "runs/<output_name>/predictions/.")
    ap.add_argument("--predictions-subdir", default="predictions",
                    help="Subdir under each run dir that holds .npy files. "
                         "Default 'predictions' (val mode); use "
                         "'predictions_test' for the test-set combine.")
    args = ap.parse_args()

    pres_dir = _expect_pred_dir(args.presence_run, args.predictions_subdir)
    hgt_dir  = _expect_pred_dir(args.height_run,   args.predictions_subdir)
    out_dir  = os.path.join(REPO_DIR, "runs", args.output_name, args.predictions_subdir)
    os.makedirs(out_dir, exist_ok=True)

    pres_files = {os.path.basename(p): p for p in glob.glob(os.path.join(pres_dir, "*.npy"))}
    hgt_files  = {os.path.basename(p): p for p in glob.glob(os.path.join(hgt_dir, "*.npy"))}

    common = sorted(set(pres_files) & set(hgt_files))
    pres_only = sorted(set(pres_files) - set(hgt_files))
    hgt_only  = sorted(set(hgt_files) - set(pres_files))
    if pres_only:
        print(f"WARN: {len(pres_only)} files only in presence run "
              f"(first 3: {pres_only[:3]})")
    if hgt_only:
        print(f"WARN: {len(hgt_only)} files only in height run "
              f"(first 3: {hgt_only[:3]})")
    if not common:
        raise SystemExit("No matching .npy basenames between the two runs. "
                         "Did you run predict.py with the same data?")

    print(f"presence run: {pres_dir} ({len(pres_files)} files)")
    print(f"height run  : {hgt_dir} ({len(hgt_files)} files)")
    print(f"common      : {len(common)} files")
    print(f"writing to  : {out_dir}")

    sample_shape = None
    for fname in common:
        pres = np.load(pres_files[fname])
        hgt  = np.load(hgt_files[fname])
        if pres.shape != hgt.shape:
            raise SystemExit(
                f"Shape mismatch on {fname}: presence={pres.shape} height={hgt.shape}"
            )
        if pres.shape[0] != 4:
            raise SystemExit(
                f"Expected 4-channel predictions, got {pres.shape} on {fname}"
            )
        combined = np.empty_like(pres)
        # Submission contract: 0=building, 1=veg, 2=water (probabilities),
        # 3=height (metres). Combine the two trained heads.
        combined[0:3] = pres[0:3]
        combined[3]   = hgt[3]
        np.save(os.path.join(out_dir, fname), combined)
        sample_shape = combined.shape

    print(f"\nWrote {len(common)} combined predictions, shape={sample_shape}")
    print(f"Next steps:")
    print(f"  python evaluate.py --only {args.output_name} --val-only")
    print(f"  python tools/diagnostic_height_rmse.py {args.output_name}")
    print(f"  python tools/sweep_thresholds.py "
          f"--pred-dir runs/{args.output_name}/{args.predictions_subdir}")


if __name__ == "__main__":
    main()
