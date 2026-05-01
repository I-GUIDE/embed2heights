"""Sweep (water_threshold, connected-component K) on a predictions dir.

For each combination, compute:
  - sample-averaged water IoU (per-image binary IoU, averaged over val)
  - total tuned score with bld/tree thresholds and height channel held FIXED
    (only the water IoU term changes)

Picks the (thr, K) that maximizes the sample-averaged water IoU AND reports
the corresponding estimated total score.

The connected-component filter is 8-connected: if the largest predicted
water component on a patch has fewer than K pixels, the entire water mask
on that patch is cleared. This recovers leaderboard sample-IoU on
empty-water patches whose predicted water support is too small to be
credible.

Usage:
    python tools/sweep_water_cc_filter.py \\
        --pred-dir runs/ensemble_5way.../predictions \\
        --bld-thr 0.46 --tree-thr 0.55 \\
        --water-thrs 0.50,0.55,...,0.95 \\
        --ks 0,4,8,12,16,24,32
"""
import argparse
import glob
import os
import sys

import numpy as np
import rasterio
from scipy.ndimage import label

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from core.metrics import (
    LABEL_THRESHOLD, CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT,
    binary_iou, build_label_map, load_val_ids, RMSE_NORMALIZATION, WEIGHTS,
)
from core.dataset import normalize_core_id


_CC8 = np.ones((3, 3), dtype=np.uint8)


def _cc_filter(mask, min_size):
    """If largest 8-connected component < min_size pixels, clear the mask."""
    if min_size <= 0:
        return mask
    if not mask.any():
        return mask
    components, n = label(mask, structure=_CC8)
    if n == 0:
        return mask
    sizes = np.bincount(components.ravel())[1:]  # skip background label 0
    if sizes.max() < min_size:
        return np.zeros_like(mask)
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--labels-dir",
                   default=os.path.abspath(os.path.join(ROOT_DIR, "..", "data", "train", "labels")))
    p.add_argument("--split-file",
                   default=os.path.join(ROOT_DIR, "splits", "split.json"))
    p.add_argument("--bld-thr", type=float, required=True,
                   help="Fixed building threshold (use the val-tuned value).")
    p.add_argument("--tree-thr", type=float, required=True,
                   help="Fixed tree threshold (use the val-tuned value).")
    p.add_argument("--water-thrs",
                   default="0.50,0.55,0.60,0.65,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94")
    p.add_argument("--ks", default="0,4,8,12,16,24,32")
    args = p.parse_args()

    val_ids = load_val_ids(args.split_file)
    label_map = build_label_map(args.labels_dir)
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npy")))

    # Cache per-image: water_prob, water_gt, bld_iou, tree_iou, rmse_b, rmse_v
    water_probs = []
    water_gts = []
    iou_b_list, iou_v_list = [], []
    rmse_b_list, rmse_v_list = [], []

    for pf in pred_files:
        cid = normalize_core_id(pf)
        if cid not in label_map or cid not in val_ids:
            continue
        pred = np.load(pf)
        with rasterio.open(label_map[cid]) as src:
            label_arr = src.read().astype(np.float32)
        h = min(pred.shape[1], label_arr.shape[1])
        w = min(pred.shape[2], label_arr.shape[2])
        pred = pred[:, :h, :w]
        label_arr = label_arr[:, :h, :w]

        gt_b = label_arr[CH_BUILDING] > LABEL_THRESHOLD
        gt_v = label_arr[CH_VEGETATION] > LABEL_THRESHOLD
        gt_w = label_arr[CH_WATER] > LABEL_THRESHOLD
        iou_b_list.append(binary_iou(pred[CH_BUILDING] > args.bld_thr, gt_b))
        iou_v_list.append(binary_iou(pred[CH_VEGETATION] > args.tree_thr, gt_v))

        if gt_b.any():
            d = pred[CH_HEIGHT][gt_b] - label_arr[CH_HEIGHT][gt_b]
            rmse_b_list.append(float(np.sqrt(np.mean(d.astype(np.float64) ** 2))))
        if gt_v.any():
            d = pred[CH_HEIGHT][gt_v] - label_arr[CH_HEIGHT][gt_v]
            rmse_v_list.append(float(np.sqrt(np.mean(d.astype(np.float64) ** 2))))

        water_probs.append(pred[CH_WATER].astype(np.float32))
        water_gts.append(gt_w)

    iou_bld_mean = float(np.nanmean(iou_b_list))
    iou_tree_mean = float(np.nanmean(iou_v_list))
    rmse_b_mean = float(np.nanmean(rmse_b_list))
    rmse_v_mean = float(np.nanmean(rmse_v_list))

    base_score_no_water = (
        WEIGHTS["iou_buildings"] * iou_bld_mean
        + WEIGHTS["iou_trees"] * iou_tree_mean
        + WEIGHTS["RMSE_building_height"] * max(0.0, 1.0 - rmse_b_mean / RMSE_NORMALIZATION["RMSE_building_height"])
        + WEIGHTS["RMSE_vegetation_height"] * max(0.0, 1.0 - rmse_v_mean / RMSE_NORMALIZATION["RMSE_vegetation_height"])
    )
    print()
    print(f"  Held-fixed thresholds: bld={args.bld_thr}, tree={args.tree_thr}")
    print(f"  Held-fixed metrics:    iou_bld={iou_bld_mean:.4f} iou_tree={iou_tree_mean:.4f} "
          f"RMSE_bH={rmse_b_mean:.4f} RMSE_vH={rmse_v_mean:.4f}")
    print(f"  Base score without water term: {base_score_no_water:.4f}")
    print()

    water_thrs = [float(x) for x in args.water_thrs.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    print(f"{'water_thr':>10} {'K':>4} {'sIoU_w':>8} {'eFP':>8} {'tot':>8}")
    print("-" * 50)

    best = None
    for thr in water_thrs:
        for K in ks:
            ious = []
            efp = 0
            n_empty = 0
            for prob, gt in zip(water_probs, water_gts):
                pb = prob > thr
                pb = _cc_filter(pb, K)
                ious.append(binary_iou(pb, gt))
                if not gt.any():
                    n_empty += 1
                    if pb.any():
                        efp += 1
            iou_w = float(np.nanmean(ious))
            total = base_score_no_water + WEIGHTS["iou_water"] * iou_w
            print(f"{thr:>10.2f} {K:>4d} {iou_w:>8.4f} {f'{efp}/{n_empty}':>8} {total:>8.4f}")
            if best is None or total > best["total"]:
                best = {"thr": thr, "K": K, "iou_w": iou_w, "total": total, "efp": efp, "n_empty": n_empty}

    print()
    print(f"BEST: water_thr={best['thr']:.2f}  K={best['K']}  "
          f"iou_w={best['iou_w']:.4f}  empty FP={best['efp']}/{best['n_empty']}  total={best['total']:.4f}")


if __name__ == "__main__":
    main()
