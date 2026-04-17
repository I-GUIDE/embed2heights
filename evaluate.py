"""
Evaluate experiment predictions against ground-truth labels.

Computes the 5 leaderboard metrics (see logs/METRIC_PROBE_REPORT.md for the
probe that fixed the formulas on 2026-04-17):
  - iou_buildings         (25%)
  - iou_trees             (15%)
  - iou_water             (15%)
  - RMSE_building_height  (25%)
  - RMSE_vegetation_height (20%)

Usage:
    python evaluate.py                            # evaluate every runs/*/predictions
    python evaluate.py --only alphaearth_baseline
    python evaluate.py --pred-threshold 0.3       # custom pred binarization
    python evaluate.py --val-only                 # use split.json stored in the experiment dir
"""
import argparse
import glob
import os
import sys

import numpy as np
import rasterio

from core.metrics import (
    WEIGHTS, LABEL_THRESHOLD,
    CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT,
    binary_iou, compute_weighted_score,
    build_label_map, load_val_ids,
)
from core.dataset import normalize_core_id


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "train"))
DEFAULT_LABELS_DIR = os.path.join(DEFAULT_DATA_DIR, "labels")
DEFAULT_SPLIT_FILE = os.path.join(SCRIPT_DIR, "splits", "split.json")


def evaluate_experiment(pred_dir, labels_dir, *, pred_threshold=0.5,
                        label_threshold=LABEL_THRESHOLD, val_only_ids=None):
    """
    Compute the 5 leaderboard metrics for one experiment.

    Aggregation:
      - IoU: per-image positive-only Jaccard (empty/empty -> 1), sample-averaged.
      - RMSE: per-image RMSE on pixels where label_class > label_threshold,
              sample-averaged (NOT pixel-accumulated).
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    if not pred_files:
        return None

    label_map = build_label_map(labels_dir)

    iou_b, iou_v, iou_w = [], [], []
    rmse_b, rmse_v = [], []
    matched = 0

    for pf in pred_files:
        core_id = normalize_core_id(pf)
        if core_id not in label_map:
            continue
        if val_only_ids is not None and core_id not in val_only_ids:
            continue
        matched += 1

        pred = np.load(pf)  # (4, H, W)
        with rasterio.open(label_map[core_id]) as src:
            label = src.read().astype(np.float32)  # (4, H, W)

        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        pred, label = pred[:, :h, :w], label[:, :h, :w]

        iou_b.append(binary_iou(pred[CH_BUILDING]  > pred_threshold, label[CH_BUILDING]   > label_threshold))
        iou_v.append(binary_iou(pred[CH_VEGETATION] > pred_threshold, label[CH_VEGETATION] > label_threshold))
        iou_w.append(binary_iou(pred[CH_WATER]     > pred_threshold, label[CH_WATER]      > label_threshold))

        # RMSE conditioned on GT class presence — per-image averaging.
        bld_mask = label[CH_BUILDING]    > label_threshold
        veg_mask = label[CH_VEGETATION]  > label_threshold
        if bld_mask.any():
            diff = pred[CH_HEIGHT][bld_mask] - label[CH_HEIGHT][bld_mask]
            rmse_b.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))
        if veg_mask.any():
            diff = pred[CH_HEIGHT][veg_mask] - label[CH_HEIGHT][veg_mask]
            rmse_v.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))

    if matched == 0:
        return None

    def safe_nanmean(arr):
        vals = [v for v in arr if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "iou_buildings":          safe_nanmean(iou_b),
        "iou_trees":              safe_nanmean(iou_v),
        "iou_water":              safe_nanmean(iou_w),
        "RMSE_building_height":   safe_nanmean(rmse_b),
        "RMSE_vegetation_height": safe_nanmean(rmse_v),
        "n_samples":              matched,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir",   default=DEFAULT_RUNS_DIR)
    p.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    p.add_argument("--only",       nargs="+", default=None, help="Evaluate only these experiment names")
    p.add_argument("--pred-threshold", "--threshold", dest="pred_threshold", type=float, default=0.5,
                   help="Binarization threshold for PREDICTION channels (default 0.5).")
    p.add_argument("--label-threshold", type=float, default=LABEL_THRESHOLD,
                   help="Binarization threshold for LABEL channels (default 0, matches leaderboard).")
    p.add_argument("--val-only", action="store_true",
                   help="Evaluate only on the val split loaded from --split-file.")
    p.add_argument("--split-file", default=DEFAULT_SPLIT_FILE,
                   help="Split JSON to use when --val-only is set. Falls back to "
                        "<runs-dir>/<exp>/split.json if this one is missing.")
    return p.parse_args()


def main():
    args = parse_args()

    exp_dirs = sorted(glob.glob(os.path.join(args.runs_dir, "*", "predictions")))
    if args.only:
        exp_dirs = [d for d in exp_dirs if os.path.basename(os.path.dirname(d)) in args.only]
    if not exp_dirs:
        print("No experiments with predictions/ found.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for pred_dir in exp_dirs:
        exp_name = os.path.basename(os.path.dirname(pred_dir))
        val_ids = None
        if args.val_only:
            split_file = args.split_file
            if not os.path.exists(split_file):
                split_file = os.path.join(args.runs_dir, exp_name, "split.json")
            if not os.path.exists(split_file):
                print(f"Evaluating {exp_name} ... no split.json found, skipping.")
                continue
            val_ids = load_val_ids(split_file)

        split_label = f"val-only, {len(val_ids)} samples" if val_ids else "all"
        print(f"Evaluating {exp_name} ({split_label}) ...", end=" ", flush=True)
        metrics = evaluate_experiment(
            pred_dir, args.labels_dir,
            pred_threshold=args.pred_threshold,
            label_threshold=args.label_threshold,
            val_only_ids=val_ids,
        )
        if metrics is None:
            print("no matched samples, skipping.")
            continue
        score, parts = compute_weighted_score(metrics)
        metrics["weighted_score"] = score
        metrics["score_parts"] = parts
        all_results[exp_name] = metrics
        print(f"done ({metrics['n_samples']} samples)")

    split_mode = "val-only (split.json)" if args.val_only else "all samples"
    print("\n" + "=" * 90)
    print(f"  Evaluation Results  (pred>{args.pred_threshold}, label>{args.label_threshold}, split={split_mode})")
    print("=" * 90)
    header = f"{'Experiment':<36} {'iou_bld':>9} {'iou_tree':>9} {'iou_wat':>9} {'RMSE_bH':>9} {'RMSE_vH':>9} {'Score':>8}"
    print(header)
    print("-" * 90)
    for exp, m in sorted(all_results.items(), key=lambda x: -x[1].get("weighted_score", 0)):
        print(
            f"{exp:<36} "
            f"{m['iou_buildings']:>9.4f} "
            f"{m['iou_trees']:>9.4f} "
            f"{m['iou_water']:>9.4f} "
            f"{m['RMSE_building_height']:>9.4f} "
            f"{m['RMSE_vegetation_height']:>9.4f} "
            f"{m['weighted_score']:>8.4f}"
        )
    print("-" * 90)
    print(f"  Weights: iou_bld={WEIGHTS['iou_buildings']:.0%}  iou_tree={WEIGHTS['iou_trees']:.0%}  "
          f"iou_wat={WEIGHTS['iou_water']:.0%}  RMSE_bH={WEIGHTS['RMSE_building_height']:.0%}  "
          f"RMSE_vH={WEIGHTS['RMSE_vegetation_height']:.0%}")
    print("  Score = sum(iou_i * w_i) + sum(max(0, 1 - RMSE_i / 30) * w_i)   [higher is better]")
    print("  NOTE: RMSE normalization is a placeholder; true X_class is smaller. See METRIC_PROBE_REPORT.md.")
    print("=" * 90)


if __name__ == "__main__":
    main()
