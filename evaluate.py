"""
Evaluate baseline predictions against ground-truth labels.

Computes the 5 leaderboard metrics:
  - mIoU_buildings  (weight 25%)
  - mIoU_trees      (weight 15%)
  - mIoU_water      (weight 15%)
  - RMSE_building_height  (weight 25%)
  - RMSE_vegetation_height (weight 20%)

Usage:
    python evaluate.py                       # evaluate all baselines with predictions
    python evaluate.py --only alphaearth     # evaluate one baseline
    python evaluate.py --threshold 0.3       # custom binarization threshold
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import rasterio
from collections import defaultdict
from sklearn.model_selection import train_test_split

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "train"))
DEFAULT_LABELS_DIR = os.path.join(DEFAULT_DATA_DIR, "labels")

# Mapping from experiment name to embedding subdir under data/train/
EXP_TO_EMB = {
    "alphaearth_baseline": "alphaearth_emb",
    "alphaearth_nodata_mask": "alphaearth_emb",
    "alphaearth_nodata_mask_2": "alphaearth_emb",
    
    "tessera_baseline": "tessera_emb",
    "terramind_s2_baseline": "terramind_s2_emb",
    "thor_s2_baseline": "thor_s2_emb",
    "terramind_s1_baseline": "terramind_s1_emb",
    "thor_s1_baseline": "thor_s1_emb",
}

# train.py split parameters (must match exactly to reproduce the same split)
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Channel indices (same for pred and label)
CH_BUILDING = 0
CH_VEGETATION = 1
CH_WATER = 2
CH_HEIGHT = 3

# Leaderboard weights
WEIGHTS = {
    "mIoU_buildings": 0.25,
    "mIoU_trees": 0.15,
    "mIoU_water": 0.15,
    "RMSE_building_height": 0.25,
    "RMSE_vegetation_height": 0.20,
}


def normalize_core_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.startswith("label_"):
        base = base[len("label_"):]
    if base.startswith("pred_"):
        base = base[len("pred_"):]
    for prefix in ("gee_emb_", "tessera_emb_", "emb_", "s2_", "s1_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    for suffix in ("_embedding", "_embeddings", "_quantized", "_merged"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    base = re.sub(r'_\d{4}$', '', base)
    return base


def binary_iou(pred_mask, true_mask):
    """IoU for a single binary class."""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return float('nan')  # class absent in both pred and gt
    return intersection / union


def mean_iou(pred_mask, true_mask):
    """
    Mean IoU = mean(IoU_positive, IoU_negative).
    Standard definition for binary segmentation.
    """
    iou_pos = binary_iou(pred_mask, true_mask)
    iou_neg = binary_iou(~pred_mask, ~true_mask)
    vals = [v for v in [iou_pos, iou_neg] if not np.isnan(v)]
    return np.mean(vals) if vals else float('nan')


def get_val_core_ids(emb_dir, labels_dir, split_file=None):
    """
    Return the set of normalized core IDs in the val split.

    If split_file exists (saved by train.py via --split-file), load it directly
    — this is the authoritative source and guarantees exact match with training.

    Otherwise, reproduce the split from train.py:
      all_pairs = find_file_pairs(emb_dir, labels_dir)
      _, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)
    Note: the fallback depends on glob order matching train.py's glob order,
    which is filesystem-dependent and NOT guaranteed across machines.
    """
    import json

    if split_file and os.path.exists(split_file):
        with open(split_file) as f:
            data = json.load(f)
        val_ids = set(data["val"])
        print(f"(loaded split from {os.path.basename(split_file)}) ", end="", flush=True)
        return val_ids

    # Fallback: reproduce the split (same logic as train.py)
    emb_files = sorted(glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True))
    label_files = glob.glob(os.path.join(labels_dir, "**", "label_*.tif"), recursive=True)
    label_map = {normalize_core_id(f): f for f in label_files}

    all_pairs = []
    for e_path in emb_files:
        cid = normalize_core_id(e_path)
        if cid in label_map:
            all_pairs.append((e_path, label_map[cid]))

    _, val_pairs = train_test_split(all_pairs, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    print("(reproduced split, no split.json found) ", end="", flush=True)
    return {normalize_core_id(e) for e, _ in val_pairs}


def evaluate_experiment(pred_dir, labels_dir, threshold=0.5, val_only_ids=None):
    """Compute the 5 metrics for one experiment."""
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "pred_*.npy")))
    if not pred_files:
        return None

    # Build label lookup
    label_files = glob.glob(os.path.join(labels_dir, "**", "label_*.tif"), recursive=True)
    label_map = {normalize_core_id(f): f for f in label_files}

    # Accumulators
    miou_building_list = []
    miou_trees_list = []
    miou_water_list = []
    se_building_height = []   # squared errors for building pixels
    se_vegetation_height = [] # squared errors for vegetation pixels
    n_building_px = 0
    n_vegetation_px = 0
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

        # Ensure spatial dims match (crop to min)
        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        pred = pred[:, :h, :w]
        label = label[:, :h, :w]

        # --- Binarize for mIoU ---
        pred_bld = pred[CH_BUILDING] > threshold
        true_bld = label[CH_BUILDING] > threshold
        pred_veg = pred[CH_VEGETATION] > threshold
        true_veg = label[CH_VEGETATION] > threshold
        pred_wat = pred[CH_WATER] > threshold
        true_wat = label[CH_WATER] > threshold

        miou_building_list.append(mean_iou(pred_bld, true_bld))
        miou_trees_list.append(mean_iou(pred_veg, true_veg))
        miou_water_list.append(mean_iou(pred_wat, true_wat))

        # --- RMSE on height channel, conditioned on class presence in GT ---
        bld_mask = label[CH_BUILDING] > threshold
        veg_mask = label[CH_VEGETATION] > threshold

        if bld_mask.any():
            diff = pred[CH_HEIGHT][bld_mask] - label[CH_HEIGHT][bld_mask]
            se_building_height.append(np.sum(diff ** 2))
            n_building_px += bld_mask.sum()

        if veg_mask.any():
            diff = pred[CH_HEIGHT][veg_mask] - label[CH_HEIGHT][veg_mask]
            se_vegetation_height.append(np.sum(diff ** 2))
            n_vegetation_px += veg_mask.sum()

    if matched == 0:
        return None

    def safe_nanmean(arr):
        vals = [v for v in arr if not np.isnan(v)]
        return np.mean(vals) if vals else float('nan')

    rmse_bld_h = np.sqrt(sum(se_building_height) / n_building_px) if n_building_px > 0 else float('nan')
    rmse_veg_h = np.sqrt(sum(se_vegetation_height) / n_vegetation_px) if n_vegetation_px > 0 else float('nan')

    metrics = {
        "mIoU_buildings": safe_nanmean(miou_building_list),
        "mIoU_trees": safe_nanmean(miou_trees_list),
        "mIoU_water": safe_nanmean(miou_water_list),
        "RMSE_building_height": rmse_bld_h,
        "RMSE_vegetation_height": rmse_veg_h,
        "n_samples": matched,
    }
    return metrics


def compute_weighted_score(metrics):
    """
    Combine into a single leaderboard-style score.
    mIoU metrics: higher is better (0-1)
    RMSE metrics: lower is better — we convert to (1 - RMSE/max_RMSE) so
    all components are "higher is better" before weighting.
    We use 30m as the max height (HEIGHT_NORM_CONSTANT from dataset.py).
    """
    MAX_HEIGHT = 30.0
    parts = {}
    for k, w in WEIGHTS.items():
        v = metrics.get(k, float('nan'))
        if np.isnan(v):
            parts[k] = float('nan')
        elif "RMSE" in k:
            # Clamp RMSE contribution to [0, 1] then invert
            parts[k] = max(0.0, 1.0 - v / MAX_HEIGHT) * w
        else:
            parts[k] = v * w
    score = sum(v for v in parts.values() if not np.isnan(v))
    return score, parts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Root data directory containing <emb>/ and labels/")
    p.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    p.add_argument("--only", nargs="+", default=None, help="Evaluate only these experiment names")
    p.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for mIoU (default 0.5)")
    p.add_argument("--val-only", action="store_true",
                   help="Evaluate only on the 20%% val split (reproduces train.py's split with seed=42)")
    return p.parse_args()


def main():
    args = parse_args()

    # Find experiments with predictions/
    exp_dirs = sorted(glob.glob(os.path.join(args.runs_dir, "*", "predictions")))
    if args.only:
        exp_dirs = [d for d in exp_dirs if os.path.basename(os.path.dirname(d)) in args.only]

    if not exp_dirs:
        print("No experiments with predictions/ found.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for pred_dir in exp_dirs:
        exp_name = os.path.basename(os.path.dirname(pred_dir))

        # Resolve val-only filter if requested
        val_ids = None
        if args.val_only:
            emb_subdir = EXP_TO_EMB.get(exp_name)
            if emb_subdir is None:
                print(f"Evaluating {exp_name} ... unknown experiment, cannot reconstruct val split, skipping.")
                continue
            emb_dir = os.path.join(args.data_dir, emb_subdir)
            split_file = os.path.join(args.runs_dir, exp_name, "split.json")
            val_ids = get_val_core_ids(emb_dir, args.labels_dir, split_file=split_file)

        split_label = f"val-only, {len(val_ids)} samples" if val_ids else "all"
        print(f"Evaluating {exp_name} ({split_label}) ...", end=" ", flush=True)
        metrics = evaluate_experiment(pred_dir, args.labels_dir, threshold=args.threshold, val_only_ids=val_ids)
        if metrics is None:
            print("no matched samples, skipping.")
            continue
        score, parts = compute_weighted_score(metrics)
        metrics["weighted_score"] = score
        metrics["score_parts"] = parts
        all_results[exp_name] = metrics
        print(f"done ({metrics['n_samples']} samples)")

    # --- Print results table ---
    split_mode = "val-only (20%, seed=42)" if args.val_only else "all samples"
    print("\n" + "=" * 90)
    print(f"  Evaluation Results  (threshold={args.threshold}, split={split_mode})")
    print("=" * 90)

    header = f"{'Experiment':<28} {'mIoU_bld':>9} {'mIoU_tree':>9} {'mIoU_wat':>9} {'RMSE_bH':>9} {'RMSE_vH':>9} {'Score':>8}"
    print(header)
    print("-" * 90)

    for exp, m in sorted(all_results.items(), key=lambda x: -x[1].get("weighted_score", 0)):
        line = (
            f"{exp:<28} "
            f"{m['mIoU_buildings']:>9.4f} "
            f"{m['mIoU_trees']:>9.4f} "
            f"{m['mIoU_water']:>9.4f} "
            f"{m['RMSE_building_height']:>9.4f} "
            f"{m['RMSE_vegetation_height']:>9.4f} "
            f"{m['weighted_score']:>8.4f}"
        )
        print(line)

    print("-" * 90)
    print(f"  Weights: mIoU_bld={WEIGHTS['mIoU_buildings']:.0%}  mIoU_tree={WEIGHTS['mIoU_trees']:.0%}  "
          f"mIoU_wat={WEIGHTS['mIoU_water']:.0%}  RMSE_bH={WEIGHTS['RMSE_building_height']:.0%}  "
          f"RMSE_vH={WEIGHTS['RMSE_vegetation_height']:.0%}")
    print(f"  Score = sum(mIoU_i * w_i) + sum((1 - RMSE_i/30) * w_i)  [higher is better]")
    print("=" * 90)


if __name__ == "__main__":
    main()
