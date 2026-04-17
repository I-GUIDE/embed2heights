"""
Evaluate baseline predictions against ground-truth labels.

Computes the 5 leaderboard metrics:
  - iou_buildings   (weight 25%)
  - iou_trees       (weight 15%)
  - iou_water       (weight 15%)
  - RMSE_building_height   (weight 25%)
  - RMSE_vegetation_height (weight 20%)

Metric definitions are derived from the 2026-04-17 all-zero probe submission
(see logs/METRIC_PROBE_REPORT.md):
  - IoU: per-image positive-only Jaccard, binarized with label > 0 and
    pred > pred_threshold, empty/empty -> 1.0 (sklearn zero_division=1.0
    convention). Averaged over samples.
  - RMSE: per-image RMSE on pixels where label_class > label_threshold,
    averaged over samples.

Usage:
    python evaluate.py                            # evaluate all baselines
    python evaluate.py --only alphaearth          # one experiment
    python evaluate.py --pred-threshold 0.3       # custom pred binarization
    python evaluate.py --label-threshold 0.5      # probe alternate label convention
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
    
    "alphaearth_refiner_softplus_bs16_lr1e4_aux005": "alphaearth_emb",
    "alphaearth_hrnet_w32_softplus_bs16_lr5e5_aux005": "alphaearth_emb",
    "alphaearth_hrnet_w18_softplus_bs16_lr1e4_aux005": "alphaearth_emb",
    
    "lightunet_v2head": "alphaearth_emb",  
    "hrnet_w18_v2head": "alphaearth_emb",    
      
    "lightunet_alphaearth": "alphaearth_emb",
    
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

# Leaderboard weights (official key names use the IoU_* prefix, matching the
# leaderboard columns iou_build / iou_veg / iou_water).
WEIGHTS = {
    "iou_buildings": 0.25,
    "iou_trees": 0.15,
    "iou_water": 0.15,
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
    """
    Per-image positive-class IoU with the official empty-case convention:
      - Both pred and gt empty -> 1.0 (perfect agreement that class is absent)
      - Exactly one empty     -> 0.0 (total disagreement)
      - Otherwise              -> |pred ∩ gt| / |pred ∪ gt|

    Matches sklearn's jaccard_score(zero_division=1.0). Derived from the
    2026-04-17 dummy-probe submission, which returned non-zero IoU values
    for an all-zero prediction — only consistent with the empty/empty -> 1
    convention.
    """
    pred_any = bool(pred_mask.any())
    true_any = bool(true_mask.any())
    if not pred_any and not true_any:
        return 1.0
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred_mask, true_mask).sum()
    return float(inter / union)


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


def evaluate_experiment(pred_dir, labels_dir, pred_threshold=0.5,
                        label_threshold=0.0, val_only_ids=None):
    """
    Compute the 5 metrics for one experiment.

    Aggregation matches the leaderboard (derived from the 2026-04-17 probe):
      - IoU: per-image positive-only Jaccard (empty/empty -> 1), averaged over samples.
      - RMSE: per-image RMSE on pixels where label_class > label_threshold,
              averaged over samples (NOT pixel-level global RMSE).
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    if not pred_files:
        return None

    # Build label lookup
    label_files = glob.glob(os.path.join(labels_dir, "**", "label_*.tif"), recursive=True)
    label_map = {normalize_core_id(f): f for f in label_files}

    # Accumulators — all per-image
    iou_building_list = []
    iou_trees_list = []
    iou_water_list = []
    rmse_building_list = []   # per-image RMSE on building pixels
    rmse_vegetation_list = [] # per-image RMSE on vegetation pixels
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

        # --- Binarize for IoU ---
        pred_bld = pred[CH_BUILDING] > pred_threshold
        true_bld = label[CH_BUILDING] > label_threshold
        pred_veg = pred[CH_VEGETATION] > pred_threshold
        true_veg = label[CH_VEGETATION] > label_threshold
        pred_wat = pred[CH_WATER] > pred_threshold
        true_wat = label[CH_WATER] > label_threshold

        iou_building_list.append(binary_iou(pred_bld, true_bld))
        iou_trees_list.append(binary_iou(pred_veg, true_veg))
        iou_water_list.append(binary_iou(pred_wat, true_wat))

        # --- RMSE on height channel, per-image, conditioned on GT class presence ---
        bld_mask = label[CH_BUILDING] > label_threshold
        veg_mask = label[CH_VEGETATION] > label_threshold

        if bld_mask.any():
            diff = pred[CH_HEIGHT][bld_mask] - label[CH_HEIGHT][bld_mask]
            rmse_building_list.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))

        if veg_mask.any():
            diff = pred[CH_HEIGHT][veg_mask] - label[CH_HEIGHT][veg_mask]
            rmse_vegetation_list.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))

    if matched == 0:
        return None

    def safe_nanmean(arr):
        vals = [v for v in arr if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')

    metrics = {
        "iou_buildings": safe_nanmean(iou_building_list),
        "iou_trees": safe_nanmean(iou_trees_list),
        "iou_water": safe_nanmean(iou_water_list),
        "RMSE_building_height": safe_nanmean(rmse_building_list),
        "RMSE_vegetation_height": safe_nanmean(rmse_vegetation_list),
        "n_samples": matched,
    }
    return metrics


def compute_weighted_score(metrics):
    """
    Combine into a single leaderboard-style score.

    IoU metrics: higher is better (0-1), contribute directly.
    RMSE metrics: `max(0, 1 - RMSE / MAX_HEIGHT)` then weighted. Higher is better.

    NOTE on MAX_HEIGHT: the 2026-04-17 probe confirmed the formula uses
    `max(0, 1 - RMSE/X)` with a class-specific X that is SMALL (≤4m for
    building, ≤10.9m for vegetation — our dummy clamped both RMSE terms to 0).
    The exact X per class is still unknown and requires a follow-up probe with
    a better height prediction. Using 30m below is a placeholder that
    over-estimates the RMSE contribution — the absolute score here will NOT
    match the leaderboard's total, though the IoU part does.
    """
    MAX_HEIGHT = 30.0  # placeholder; true X per class is smaller, TBD by probe.
    parts = {}
    for k, w in WEIGHTS.items():
        v = metrics.get(k, float('nan'))
        if np.isnan(v):
            parts[k] = float('nan')
        elif "RMSE" in k:
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
    p.add_argument("--pred-threshold", "--threshold", dest="pred_threshold", type=float, default=0.5,
                   help="Binarization threshold for PREDICTION channels (default 0.5). "
                        "The legacy alias --threshold is accepted.")
    p.add_argument("--label-threshold", type=float, default=0.0,
                   help="Binarization threshold for LABEL channels in both IoU and RMSE "
                        "pixel selection. Default 0.0 matches the leaderboard "
                        "(any non-zero fraction counts as positive).")
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

    # --- Print results table ---
    split_mode = "val-only (20%, seed=42)" if args.val_only else "all samples"
    print("\n" + "=" * 90)
    print(f"  Evaluation Results  (pred>{args.pred_threshold}, label>{args.label_threshold}, split={split_mode})")
    print("=" * 90)

    header = f"{'Experiment':<28} {'iou_bld':>9} {'iou_tree':>9} {'iou_wat':>9} {'RMSE_bH':>9} {'RMSE_vH':>9} {'Score':>8}"
    print(header)
    print("-" * 90)

    for exp, m in sorted(all_results.items(), key=lambda x: -x[1].get("weighted_score", 0)):
        line = (
            f"{exp:<28} "
            f"{m['iou_buildings']:>9.4f} "
            f"{m['iou_trees']:>9.4f} "
            f"{m['iou_water']:>9.4f} "
            f"{m['RMSE_building_height']:>9.4f} "
            f"{m['RMSE_vegetation_height']:>9.4f} "
            f"{m['weighted_score']:>8.4f}"
        )
        print(line)

    print("-" * 90)
    print(f"  Weights: iou_bld={WEIGHTS['iou_buildings']:.0%}  iou_tree={WEIGHTS['iou_trees']:.0%}  "
          f"iou_wat={WEIGHTS['iou_water']:.0%}  RMSE_bH={WEIGHTS['RMSE_building_height']:.0%}  "
          f"RMSE_vH={WEIGHTS['RMSE_vegetation_height']:.0%}")
    print(f"  Score = sum(iou_i * w_i) + sum(max(0, 1 - RMSE_i/X_i) * w_i)   [higher is better]")
    print(f"  NOTE: X_i (RMSE normalization) is unknown; placeholder=30 over-estimates RMSE contribution.")
    print("=" * 90)


if __name__ == "__main__":
    main()
