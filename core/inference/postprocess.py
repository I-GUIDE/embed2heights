"""Prediction post-processing: water-CC filter, threshold sweep, seg binarisation,
height calibration, and submission zipping — everything applied AFTER the model
forward, shared by ``assemble_final.py``.
"""

import glob
import os
import zipfile

import numpy as np

from core.metrics import binary_iou

# Channel layout of a prediction / label: 0=building, 1=vegetation, 2=water, 3=height.
_WATER = 2


def largest_component_size(mask):
    """Return largest 8-connected component size for a boolean 2D mask."""
    if not mask.any():
        return 0
    try:
        from scipy.ndimage import label as cc_label

        structure = np.ones((3, 3), dtype=np.uint8)
        comps, n_comp = cc_label(mask, structure=structure)
        if n_comp == 0:
            return 0
        sizes = np.bincount(comps.ravel())[1:]
        return int(sizes.max()) if len(sizes) else 0
    except ImportError:
        visited = np.zeros(mask.shape, dtype=bool)
        h, w = mask.shape
        best = 0
        ys, xs = np.nonzero(mask)
        for sy, sx in zip(ys, xs):
            if visited[sy, sx]:
                continue
            visited[sy, sx] = True
            stack = [(int(sy), int(sx))]
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for ny in range(max(0, y - 1), min(h, y + 2)):
                    for nx in range(max(0, x - 1), min(w, x + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            best = max(best, size)
        return best


def apply_water_cc_filter(mask, min_size):
    """Zero out the water mask if its largest blob is smaller than ``min_size`` px."""
    if int(min_size) <= 0 or not mask.any():
        return mask
    if largest_component_size(mask) < int(min_size):
        return np.zeros_like(mask, dtype=bool)
    return mask


def sweep_class_thresholds(seg_label_pairs, probs, *, gt_cov=0.10, water_k=4):
    """Pick the per-class probability threshold that maximises mean per-tile IoU
    against the official GT (coverage > gt_cov), over out-of-fold val tiles.

    seg_label_pairs: iterable of (seg_probs[3,H,W], label[>=3,H,W]) aligned arrays.
    Returns {0: bld_thr, 1: veg_thr, 2: wat_thr}. Water uses the CC filter.
    """
    probs = list(probs)
    acc = {c: [[] for _ in probs] for c in range(3)}
    for seg, lab in seg_label_pairs:
        for c in range(3):
            gt = lab[c] > gt_cov
            for j, p in enumerate(probs):
                m = seg[c] >= p
                if c == _WATER:
                    m = apply_water_cc_filter(m, water_k)
                acc[c][j].append(binary_iou(m, gt))
    best = {}
    for c in range(3):
        means = [float(np.mean(v)) if v else 0.0 for v in acc[c]]
        best[c] = float(probs[int(np.argmax(means))])
    return best


def binarize_seg(seg_probs, thresholds, *, water_k=4):
    """Threshold the 3 class probability maps to {0,1}; water gets the CC filter.
    thresholds: (bld, veg, wat). Returns float32 [3,H,W]."""
    out = np.zeros((3,) + seg_probs.shape[1:], np.float32)
    for c in range(3):
        m = seg_probs[c] >= thresholds[c]
        if c == _WATER:
            m = apply_water_cc_filter(m, water_k)
        out[c] = np.asarray(m, dtype=np.float32)
    return out


def calibrate_height(height, bld_mask, veg_mask, *, a_bld, b_bld, veg_shift):
    """Building pixels: de-compress h -> a_bld*h + b_bld; veg pixels: h -> h + veg_shift.
    Masks are disjoint (veg excludes building); both read the original height."""
    h = height.copy()
    h[bld_mask] = a_bld * height[bld_mask] + b_bld
    h[veg_mask] = height[veg_mask] + veg_shift
    return h


def assemble_tile(seg_mean, height_mean, thresholds, *, water_k, a_bld, b_bld, veg_shift):
    """Combine mean seg + mean height into the final [4,H,W] prediction:
    binarise seg (ch0-2) and calibrate height (ch3)."""
    seg = binarize_seg(seg_mean, thresholds, water_k=water_k)
    out = np.zeros((4,) + seg_mean.shape[1:], np.float32)
    out[:3] = seg
    bld_mask = seg[0] > 0.5
    veg_mask = (seg[1] > 0.5) & (~bld_mask)
    out[3] = calibrate_height(height_mean, bld_mask, veg_mask,
                              a_bld=a_bld, b_bld=b_bld, veg_shift=veg_shift)
    return out


def write_submission_zip(pred_dir, zip_path):
    """Zip every <pred_dir>/*.npy under a top-level predictions/ folder. Returns count."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(glob.glob(os.path.join(pred_dir, "*.npy"))):
            z.write(p, os.path.join("predictions", os.path.basename(p)))
    with zipfile.ZipFile(zip_path) as z:
        return len(z.namelist())
