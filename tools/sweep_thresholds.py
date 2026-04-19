"""
Sweep the prediction binarization threshold for a single predictions/ directory,
optimizing the leaderboard metric. Decoupled from ensemble — feed it any dir of
[4, H, W] .npy files and it will report the best global threshold and the best
per-class thresholds.

Typical use:
    # Sweep a single model
    python tools/sweep_thresholds.py --pred-dir runs/lightunet_alphaearth/predictions

    # Sweep an ensemble (first run tools/ensemble.py to produce the dir)
    python tools/sweep_thresholds.py --pred-dir runs/ens_mean/predictions

    # Restrict to a val split (JSON with "val" key)
    python tools/sweep_thresholds.py \\
        --pred-dir runs/lightunet_alphaearth/predictions \\
        --split-file splits/split.json

Metric definitions come from core/metrics.py (leaderboard-aligned, verified
via the 2026-04-17 dummy probe — see logs/METRIC_PROBE_REPORT.md).
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.dataset import normalize_core_id  # noqa: E402
from core.metrics import (  # noqa: E402
    WEIGHTS, LABEL_THRESHOLD,
    binary_iou, compute_weighted_score,
    build_label_map, load_val_ids,
)

DEFAULT_LABELS_DIR = SCRIPT_DIR.parent / "data" / "train" / "labels"


def evaluate_at_thresholds(labels, predictions, pred_thresholds):
    """
    labels, predictions: {core_id: np.ndarray(4, H, W)}
    pred_thresholds: (t_bld, t_veg, t_wat)
    """
    iou_lists = [[], [], []]
    rmse_bld, rmse_veg = [], []

    for core_id, pred in predictions.items():
        label = labels[core_id]
        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        pred = pred[:, :h, :w]
        label = label[:, :h, :w]

        for c in range(3):
            iou_lists[c].append(binary_iou(
                pred[c] > pred_thresholds[c],
                label[c] > LABEL_THRESHOLD,
            ))

        for ch, bucket in ((0, rmse_bld), (1, rmse_veg)):
            mask = label[ch] > LABEL_THRESHOLD
            if mask.any():
                diff = pred[3][mask] - label[3][mask]
                bucket.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))

    def sm(a): return float(np.mean([v for v in a if not np.isnan(v)])) if a else float("nan")
    metrics = {
        "iou_buildings":          sm(iou_lists[0]),
        "iou_trees":              sm(iou_lists[1]),
        "iou_water":              sm(iou_lists[2]),
        "RMSE_building_height":   sm(rmse_bld),
        "RMSE_vegetation_height": sm(rmse_veg),
        "n_samples":              len(predictions),
    }
    score, _ = compute_weighted_score(metrics)
    metrics["weighted_score"] = score
    return metrics


def fmt(m):
    return (f"{m['iou_buildings']:.4f} {m['iou_trees']:.4f} {m['iou_water']:.4f} "
            f"{m['RMSE_building_height']:.4f} {m['RMSE_vegetation_height']:.4f} "
            f"{m['weighted_score']:.4f}")


def load_inputs(pred_dir, labels_dir, val_ids):
    pred_files = sorted(glob.glob(str(pred_dir / "*.npy")))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")

    label_map = build_label_map(str(labels_dir))
    predictions, labels = {}, {}
    missing_labels = []
    for pf in pred_files:
        cid = normalize_core_id(pf)
        if val_ids is not None and cid not in val_ids:
            continue
        if cid not in label_map:
            missing_labels.append(cid)
            continue
        predictions[cid] = np.load(pf).astype(np.float32)
        with rasterio.open(label_map[cid]) as src:
            labels[cid] = src.read().astype(np.float32)

    if missing_labels:
        print(f"WARN: {len(missing_labels)} predictions had no matching label (skipped).", file=sys.stderr)
    if not predictions:
        raise RuntimeError("No matched pred/label pairs after filtering.")
    return predictions, labels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", type=Path, required=True,
                   help="Directory containing [4,H,W] .npy prediction files.")
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    p.add_argument("--split-file", type=Path, default=None,
                   help="Optional JSON with a 'val' key to restrict evaluation to val only.")
    p.add_argument("--grid-start", type=float, default=0.05)
    p.add_argument("--grid-stop",  type=float, default=0.90)
    p.add_argument("--grid-step",  type=float, default=0.025)
    return p.parse_args()


def main():
    args = parse_args()

    val_ids = load_val_ids(str(args.split_file)) if args.split_file else None
    print(f"Loading predictions from {args.pred_dir}"
          + (f" (val-only: {len(val_ids)} ids)" if val_ids else " (all samples)"))
    predictions, labels = load_inputs(args.pred_dir, args.labels_dir, val_ids)
    print(f"Matched {len(predictions)} pred/label pairs.")

    grid = np.arange(args.grid_start, args.grid_stop + 1e-9, args.grid_step)

    # 1. Base
    base = evaluate_at_thresholds(labels, predictions, (0.5, 0.5, 0.5))

    # 2. Single global threshold sweep
    best_global = None
    for t in grid:
        m = evaluate_at_thresholds(labels, predictions, (t, t, t))
        if best_global is None or m["weighted_score"] > best_global[0]["weighted_score"]:
            best_global = (m, float(t))

    # 3. Greedy per-class — each class optimized on its own IoU
    per_class = [0.5, 0.5, 0.5]
    for c, key in enumerate(("iou_buildings", "iou_trees", "iou_water")):
        best_t, best_v = 0.5, -1.0
        for t in grid:
            trial = per_class.copy()
            trial[c] = float(t)
            m = evaluate_at_thresholds(labels, predictions, tuple(trial))
            if m[key] > best_v:
                best_t, best_v = float(t), m[key]
        per_class[c] = best_t
    best_per_class = evaluate_at_thresholds(labels, predictions, tuple(per_class))

    # --- Report ---
    print("\n" + "=" * 95)
    print(f"  Threshold sweep for: {args.pred_dir}")
    print(f"  Label threshold fixed at {LABEL_THRESHOLD}  (leaderboard convention)")
    print("=" * 95)
    print(f"{'Row':<34} {'iou_bld':>8} {'iou_tree':>9} {'iou_wat':>8} {'RMSE_bH':>8} {'RMSE_vH':>8} {'Score':>8}")
    print("-" * 95)
    print(f"{'base (0.5, 0.5, 0.5)':<34} {fmt(base)}")
    print(f"{'global-best @' + f'{best_global[1]:.3f}':<34} {fmt(best_global[0])}")
    thr_label = f"per-class ({per_class[0]:.3f},{per_class[1]:.3f},{per_class[2]:.3f})"
    print(f"{thr_label:<34} {fmt(best_per_class)}")
    print("=" * 95)
    print(f"  Weights: iou_bld={WEIGHTS['iou_buildings']:.0%}  iou_tree={WEIGHTS['iou_trees']:.0%}  "
          f"iou_wat={WEIGHTS['iou_water']:.0%}  RMSE_bH={WEIGHTS['RMSE_building_height']:.0%}  "
          f"RMSE_vH={WEIGHTS['RMSE_vegetation_height']:.0%}")
    print(f"  Score uses per-class RMSE ceilings from core/metrics.RMSE_NORMALIZATION "
          f"(building=3.0m, vegetation=5.0m).")


if __name__ == "__main__":
    main()
