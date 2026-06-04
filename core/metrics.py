"""
Leaderboard metric helpers shared by evaluation and tuning scripts.

Metric formulas were verified by the 2026-04-17 all-zero probe (see
logs/METRIC_PROBE_REPORT.md) and are the single source of truth for both
offline evaluation and ensemble/threshold tuning.
"""

import glob
import json
import os

import numpy as np

from core.data.discovery import normalize_core_id


# Channel indices (same layout for pred and label)
CH_BUILDING = 0
CH_VEGETATION = 1
CH_WATER = 2
CH_HEIGHT = 3

# Leaderboard weights
WEIGHTS = {
    "iou_buildings":          0.25,
    "iou_trees":              0.15,
    "iou_water":              0.15,
    "RMSE_building_height":   0.25,
    "RMSE_vegetation_height": 0.20,
}

# Label binarization threshold used by the server (any nonzero fraction is
# positive). See METRIC_PROBE_REPORT.md.
LABEL_THRESHOLD = 0.0

# Per-class RMSE normalization for `max(0, 1 - RMSE / X)`.
RMSE_NORMALIZATION = {
    "RMSE_building_height":   3.0,
    "RMSE_vegetation_height": 5.0,
}


def pred_threshold_to_triplet(pred_threshold):
    """
    Scalar or (bld, veg, wat) iterable; matches evaluate.evaluate_experiment().
    """
    if np.isscalar(pred_threshold):
        t = float(pred_threshold)
        return t, t, t
    b, v, w = pred_threshold
    return float(b), float(v), float(w)


def new_leaderboard_accumulator():
    """Mutable accumulator for patch-wise IoU lists and conditional RMSE lists."""
    return {
        "iou_buildings": [],
        "iou_trees": [],
        "iou_water": [],
        "rmse_building_pixels": [],
        "rmse_vegetation_pixels": [],
    }


def append_patch_leaderboard_accumulators(
    pred_chw,
    label_chw,
    *,
    pred_threshold,
    label_threshold=LABEL_THRESHOLD,
    acc,
):
    """
    One patch worth of leaderboard logic (evaluate.py inner loop).

    pred_chw, label_chw: (4, H, W) floats; heights must match (e.g. meters).
    pred_threshold: scalar or length-3 (bld, veg, wat).

    Mutates ``acc`` lists from new_leaderboard_accumulator().
    """
    pred_chw = np.asarray(pred_chw, dtype=np.float32)
    label_chw = np.asarray(label_chw, dtype=np.float32)
    thr_b, thr_v, thr_w = pred_threshold_to_triplet(pred_threshold)

    h = min(pred_chw.shape[1], label_chw.shape[1])
    w = min(pred_chw.shape[2], label_chw.shape[2])
    pred_chw = pred_chw[:, :h, :w]
    label_chw = label_chw[:, :h, :w]

    acc["iou_buildings"].append(
        binary_iou(pred_chw[CH_BUILDING] > thr_b, label_chw[CH_BUILDING] > label_threshold)
    )
    acc["iou_trees"].append(
        binary_iou(
            pred_chw[CH_VEGETATION] > thr_v, label_chw[CH_VEGETATION] > label_threshold
        )
    )
    acc["iou_water"].append(
        binary_iou(pred_chw[CH_WATER] > thr_w, label_chw[CH_WATER] > label_threshold)
    )

    bld_mask = label_chw[CH_BUILDING] > label_threshold
    veg_mask = label_chw[CH_VEGETATION] > label_threshold
    if bld_mask.any():
        diff = pred_chw[CH_HEIGHT][bld_mask] - label_chw[CH_HEIGHT][bld_mask]
        acc["rmse_building_pixels"].append(
            float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
        )
    if veg_mask.any():
        diff = pred_chw[CH_HEIGHT][veg_mask] - label_chw[CH_HEIGHT][veg_mask]
        acc["rmse_vegetation_pixels"].append(
            float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
        )


def summarize_leaderboard_accumulator(acc, *, n_samples=None):
    """
    Compress accumulator lists into the five metric keys (+ n_samples).

    If ``n_samples`` is None, ``len(acc['iou_buildings'])`` is used (every patch
    appends exactly three IoU values).
    """
    def safe_nanmean(arr):
        vals = [v for v in arr if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    n = len(acc["iou_buildings"]) if n_samples is None else int(n_samples)

    return {
        "iou_buildings": safe_nanmean(acc["iou_buildings"]),
        "iou_trees": safe_nanmean(acc["iou_trees"]),
        "iou_water": safe_nanmean(acc["iou_water"]),
        "RMSE_building_height": safe_nanmean(acc["rmse_building_pixels"]),
        "RMSE_vegetation_height": safe_nanmean(acc["rmse_vegetation_pixels"]),
        "n_samples": n,
    }


def binary_iou(pred_mask, true_mask):
    """
    Per-image positive-class IoU with the leaderboard empty-case convention:
      - Both pred and gt empty -> 1.0 (perfect agreement on absence)
      - Exactly one empty      -> 0.0
      - Otherwise              -> |pred intersection gt| / |pred union|

    Matches sklearn's jaccard_score(zero_division=1.0).
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


def compute_weighted_score(metrics, max_height=RMSE_NORMALIZATION):
    """
    Combine the 5 leaderboard metrics into a single score.

    IoU metrics: higher is better (0-1), weighted directly.
    RMSE metrics: `max(0, 1 - RMSE / max_height[key])` then weighted.
    `max_height` can be a dict keyed by metric name, or a scalar applied to all.
    """
    parts = {}
    for key, weight in WEIGHTS.items():
        value = metrics.get(key, float("nan"))
        if np.isnan(value):
            parts[key] = float("nan")
        elif "RMSE" in key:
            norm = max_height[key] if isinstance(max_height, dict) else max_height
            parts[key] = max(0.0, 1.0 - value / norm) * weight
        else:
            parts[key] = value * weight
    score = sum(v for v in parts.values() if not np.isnan(v))
    return score, parts


def build_label_map(labels_dir):
    """Map `normalize_core_id(label_path) -> label_path` for all labels on disk."""
    label_files = glob.glob(os.path.join(labels_dir, "**", "label_*.tif"), recursive=True)
    return {normalize_core_id(path): path for path in label_files}


def load_val_ids(split_file):
    """Load the val split's normalized core ids from a JSON split file."""
    with open(split_file) as f:
        return set(json.load(f)["val"])
