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

# Kept for backward compatibility (threshold sweep reports, etc.)
LABEL_THRESHOLD = 0.0


def label_gt_mask(label, channel):
    """Argmax GT mask: a pixel is positive for `channel` iff it is the dominant class.

    Among the three presence channels (0=building, 1=veg, 2=water), the pixel is
    assigned to whichever has the largest fraction. Empty pixels (all fractions == 0)
    are negative for every class.
    """
    presence = label[:3]
    dominant = np.argmax(presence, axis=0)
    has_any = presence.max(axis=0) > 0
    return (dominant == channel) & has_any

# Per-class RMSE normalization for `max(0, 1 - RMSE / X)`.
RMSE_NORMALIZATION = {
    "RMSE_building_height":   3.0,
    "RMSE_vegetation_height": 5.0,
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
