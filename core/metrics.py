"""
Leaderboard metric helpers — shared by evaluate.py and the tools/ scripts.

Metric formulas were verified by the 2026-04-17 all-zero probe (see
logs/METRIC_PROBE_REPORT.md) and are the single source of truth for both
offline evaluation and ensemble/threshold tuning.
"""

import glob
import json
import os

import numpy as np

from .dataset import normalize_core_id


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

# Placeholder for `max(0, 1 - RMSE / X)` — X_class is unknown (bounded by
# X_bld < 4m, X_veg < 10.9m from the probe). Using 30 over-estimates the RMSE
# contribution, but IoU portion and model ranking are correct.
RMSE_NORMALIZATION = 30.0


def binary_iou(pred_mask, true_mask):
    """
    Per-image positive-class IoU with the leaderboard empty-case convention:
      - Both pred and gt empty -> 1.0 (perfect agreement on absence)
      - Exactly one empty      -> 0.0
      - Otherwise              -> |pred ∩ gt| / |pred ∪ gt|

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
    RMSE metrics: `max(0, 1 - RMSE / max_height)` then weighted.
    """
    parts = {}
    for key, weight in WEIGHTS.items():
        value = metrics.get(key, float("nan"))
        if np.isnan(value):
            parts[key] = float("nan")
        elif "RMSE" in key:
            parts[key] = max(0.0, 1.0 - value / max_height) * weight
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
