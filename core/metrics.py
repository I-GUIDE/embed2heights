"""Shared metric helpers: channel layout, per-tile IoU, and label lookup.

The official evaluation (presence = coverage > 0.10, plus building/vegetation
height RMSE) lives in ``evaluate.py``, which consumes these helpers.
"""

import glob
import os

import numpy as np

from core.data.discovery import normalize_core_id


# Channel indices (same layout for pred and label)
CH_BUILDING = 0
CH_VEGETATION = 1
CH_WATER = 2
CH_HEIGHT = 3


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


def build_label_map(labels_dir):
    """Map `normalize_core_id(label_path) -> label_path` for all labels on disk."""
    label_files = glob.glob(os.path.join(labels_dir, "**", "label_*.tif"), recursive=True)
    return {normalize_core_id(path): path for path in label_files}
