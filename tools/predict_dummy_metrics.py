"""
Predict what the leaderboard will return for a dummy (constant-value) submission
under each candidate metric formula, using the training labels as proxy for the
test set distribution.

Usage (default probe = all zeros):
    python tools/predict_dummy_metrics.py

Change the probe:
    python tools/predict_dummy_metrics.py --class-value 0.5 --height-value 0.0

Then compare the printed table against the actual leaderboard numbers after
uploading submission from make_dummy_submission.py. The candidate row whose 5
values best match the server's output is the official metric implementation.

Candidates covered:
  - mIoU: positive-only vs mean(pos, neg); GT threshold 0.5 vs 0; per-image
    macro-average vs global pixel-accumulated.
  - RMSE: pixel selection with label > 0.5 vs label > 0; per-image macro vs
    global pixel-accumulated; raw vs /30-normalized.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_LABELS_DIR = Path("/u/dingqi2/workspace/esa/data/train/labels")

CH_BUILDING = 0
CH_VEGETATION = 1
CH_WATER = 2
CH_HEIGHT = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    p.add_argument("--class-value", type=float, default=0.0,
                   help="Dummy prediction value for channels 0-2.")
    p.add_argument("--height-value", type=float, default=0.0,
                   help="Dummy prediction value for channel 3 (meters).")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all labels.")
    return p.parse_args()


def binary_iou(pred_mask, true_mask):
    inter = np.logical_and(pred_mask, true_mask).sum(dtype=np.int64)
    union = np.logical_or(pred_mask, true_mask).sum(dtype=np.int64)
    if union == 0:
        return np.nan
    return inter / union


def nanmean(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def soft_iou(pred, label):
    """Continuous IoU: sum(min(p, t)) / sum(max(p, t))."""
    num = np.minimum(pred, label).sum(dtype=np.float64)
    den = np.maximum(pred, label).sum(dtype=np.float64)
    if den == 0:
        return np.nan
    return num / den


def main():
    args = parse_args()

    label_files = sorted(glob.glob(str(args.labels_dir / "label_*.tif")))
    if args.max_samples > 0:
        label_files = label_files[:args.max_samples]
    if not label_files:
        raise RuntimeError(f"No labels in {args.labels_dir}")

    print(f"Using {len(label_files)} training labels as proxy for test distribution")
    print(f"Probe: class_value={args.class_value}, height_value={args.height_value}")

    class_channels = [("building", CH_BUILDING), ("trees", CH_VEGETATION), ("water", CH_WATER)]
    height_pair_channels = [("building_h", CH_BUILDING), ("vegetation_h", CH_VEGETATION)]

    # ----- per-image IoU accumulators (macro-average variants) -----
    per_img_iou = {
        f"{cls}_posonly_th05": [] for cls, _ in class_channels
    }
    per_img_iou.update({f"{cls}_meanpn_th05": [] for cls, _ in class_channels})
    per_img_iou.update({f"{cls}_posonly_th0": [] for cls, _ in class_channels})
    per_img_iou.update({f"{cls}_meanpn_th0": [] for cls, _ in class_channels})
    per_img_iou.update({f"{cls}_soft": [] for cls, _ in class_channels})

    # ----- global pixel-accumulated counters -----
    glob_ctr = {}
    for cls, _ in class_channels:
        for thresh_label in ("th05", "th0"):
            glob_ctr[f"{cls}_inter_{thresh_label}"] = 0
            glob_ctr[f"{cls}_union_{thresh_label}"] = 0
            glob_ctr[f"{cls}_neg_inter_{thresh_label}"] = 0
            glob_ctr[f"{cls}_neg_union_{thresh_label}"] = 0
        glob_ctr[f"{cls}_soft_num"] = 0.0
        glob_ctr[f"{cls}_soft_den"] = 0.0

    # ----- RMSE accumulators -----
    rmse_acc = {}
    for name, _ in height_pair_channels:
        for thresh_label in ("th05", "th0"):
            rmse_acc[f"{name}_se_{thresh_label}"] = 0.0
            rmse_acc[f"{name}_n_{thresh_label}"] = 0
            rmse_acc[f"{name}_per_img_rmse_{thresh_label}"] = []

    # Dummy arrays (created once, re-used per patch — spatial dims match label)
    # Class channel threshold conventions: assume pred > 0.5 and label > 0.5
    pred_cls_val = float(args.class_value)
    pred_h_val = float(args.height_value)

    # Precompute pred masks for the two conventional thresholds
    pred_mask_gt05 = pred_cls_val > 0.5
    pred_mask_gt0 = pred_cls_val > 0.0

    for lf in tqdm(label_files, desc="Accumulating over training labels"):
        with rasterio.open(lf) as src:
            label = src.read().astype(np.float32)  # (4, H, W)

        h, w = label.shape[1], label.shape[2]

        for cls, ch in class_channels:
            lbl = label[ch]

            for thresh_label, thresh in (("th05", 0.5), ("th0", 0.0)):
                if thresh_label == "th05":
                    true_pos = lbl > 0.5
                    pred_pos = np.full_like(true_pos, pred_mask_gt05)
                else:
                    # thresh 0: label > 0
                    true_pos = lbl > 0.0
                    pred_pos = np.full_like(true_pos, pred_mask_gt0)

                true_neg = ~true_pos
                pred_neg = ~pred_pos

                # per-image
                iou_p = binary_iou(pred_pos, true_pos)
                iou_n = binary_iou(pred_neg, true_neg)
                per_img_iou[f"{cls}_posonly_{thresh_label}"].append(iou_p)
                vals = [v for v in (iou_p, iou_n) if not np.isnan(v)]
                per_img_iou[f"{cls}_meanpn_{thresh_label}"].append(
                    float(np.mean(vals)) if vals else np.nan
                )

                # global
                glob_ctr[f"{cls}_inter_{thresh_label}"] += int(np.logical_and(pred_pos, true_pos).sum())
                glob_ctr[f"{cls}_union_{thresh_label}"] += int(np.logical_or(pred_pos, true_pos).sum())
                glob_ctr[f"{cls}_neg_inter_{thresh_label}"] += int(np.logical_and(pred_neg, true_neg).sum())
                glob_ctr[f"{cls}_neg_union_{thresh_label}"] += int(np.logical_or(pred_neg, true_neg).sum())

            # soft IoU (continuous)
            pred_cont = np.full_like(lbl, pred_cls_val)
            soft = soft_iou(pred_cont, lbl)
            per_img_iou[f"{cls}_soft"].append(soft)
            glob_ctr[f"{cls}_soft_num"] += float(np.minimum(pred_cont, lbl).sum())
            glob_ctr[f"{cls}_soft_den"] += float(np.maximum(pred_cont, lbl).sum())

        # ---- RMSE ----
        for name, ch in height_pair_channels:
            lbl_cls = label[ch]
            lbl_h = label[CH_HEIGHT]
            for thresh_label, thresh in (("th05", 0.5), ("th0", 0.0)):
                mask = lbl_cls > thresh
                if mask.any():
                    diff = pred_h_val - lbl_h[mask]
                    sse = float(np.sum(diff.astype(np.float64) ** 2))
                    n = int(mask.sum())
                    rmse_acc[f"{name}_se_{thresh_label}"] += sse
                    rmse_acc[f"{name}_n_{thresh_label}"] += n
                    rmse_acc[f"{name}_per_img_rmse_{thresh_label}"].append(float(np.sqrt(sse / n)))

    # --------- Compose candidate tables ----------
    print("\n" + "=" * 95)
    print(f"  Predicted leaderboard values under each candidate formula")
    print(f"  Probe: class_value={pred_cls_val}, height_value={pred_h_val}")
    print("=" * 95)

    iou_candidates = []
    for name, label_key in [
        ("A1. positive-only IoU, GT>0.5, per-image",  "posonly_th05"),
        ("A2. positive-only IoU, GT>0.5, global",     "posonly_th05_global"),
        ("B1. mean(pos,neg) IoU,  GT>0.5, per-image", "meanpn_th05"),
        ("B2. mean(pos,neg) IoU,  GT>0.5, global",    "meanpn_th05_global"),
        ("C1. positive-only IoU, GT>0,   per-image",  "posonly_th0"),
        ("C2. positive-only IoU, GT>0,   global",     "posonly_th0_global"),
        ("D1. mean(pos,neg) IoU,  GT>0,   per-image", "meanpn_th0"),
        ("D2. mean(pos,neg) IoU,  GT>0,   global",    "meanpn_th0_global"),
        ("E.  soft IoU (continuous), per-image",      "soft"),
        ("F.  soft IoU (continuous), global",         "soft_global"),
    ]:
        row = {"name": name}
        for cls, _ in class_channels:
            if label_key.endswith("_global"):
                base = label_key.replace("_global", "")
                if base == "soft":
                    n = glob_ctr[f"{cls}_soft_num"]
                    d = glob_ctr[f"{cls}_soft_den"]
                    v = n / d if d > 0 else float("nan")
                else:
                    # posonly_th05, posonly_th0, meanpn_th05, meanpn_th0
                    kind, th = base.split("_")
                    inter = glob_ctr[f"{cls}_inter_{th}"]
                    union = glob_ctr[f"{cls}_union_{th}"]
                    iou_p = inter / union if union > 0 else float("nan")
                    if kind == "posonly":
                        v = iou_p
                    else:
                        ni = glob_ctr[f"{cls}_neg_inter_{th}"]
                        nu = glob_ctr[f"{cls}_neg_union_{th}"]
                        iou_n = ni / nu if nu > 0 else float("nan")
                        vals = [x for x in (iou_p, iou_n) if not np.isnan(x)]
                        v = float(np.mean(vals)) if vals else float("nan")
            else:
                v = nanmean(per_img_iou[f"{cls}_{label_key}"])
            row[cls] = v
        iou_candidates.append(row)

    print(f"\nIoU candidates  (columns: buildings | trees | water)")
    print("-" * 95)
    print(f"{'Candidate':<50} {'bld':>10} {'tree':>10} {'water':>10}")
    print("-" * 95)
    for r in iou_candidates:
        print(f"{r['name']:<50} "
              f"{r['building']:>10.4f} {r['trees']:>10.4f} {r['water']:>10.4f}")

    print("\nRMSE candidates  (columns: raw meters | (1 - RMSE/30) normalized)")
    print("-" * 95)
    print(f"{'Candidate':<50} {'bld_h':>10} {'veg_h':>10} {'bldN':>8} {'vegN':>8}")
    print("-" * 95)
    rmse_rows = []
    for name, label_key in [
        ("R1. RMSE on GT>0.5 pixels, global pixel-accum", "se_th05"),
        ("R2. RMSE on GT>0   pixels, global pixel-accum", "se_th0"),
        ("R3. RMSE on GT>0.5 pixels, per-image macro",    "per_img_rmse_th05"),
        ("R4. RMSE on GT>0   pixels, per-image macro",    "per_img_rmse_th0"),
    ]:
        vals = {}
        for hname, _ in height_pair_channels:
            if label_key.startswith("se_"):
                sse = rmse_acc[f"{hname}_{label_key}"]
                n = rmse_acc[f"{hname}_n_{label_key[3:]}"]
                vals[hname] = float(np.sqrt(sse / n)) if n > 0 else float("nan")
            else:
                vals[hname] = nanmean(rmse_acc[f"{hname}_{label_key}"])
        bld = vals["building_h"]
        veg = vals["vegetation_h"]
        bldN = max(0.0, 1.0 - bld / 30.0) if not np.isnan(bld) else float("nan")
        vegN = max(0.0, 1.0 - veg / 30.0) if not np.isnan(veg) else float("nan")
        rmse_rows.append((name, bld, veg, bldN, vegN))
        print(f"{name:<50} {bld:>10.4f} {veg:>10.4f} {bldN:>8.4f} {vegN:>8.4f}")

    print("\n" + "=" * 95)
    print("How to read this:")
    print("  After submitting the dummy, the leaderboard returns 5 numbers:")
    print("    iou_build, iou_veg, iou_water, rmse_h_build, rmse_h_veg")
    print("  Scan the tables above — the IoU candidate that matches the first 3 numbers")
    print("  (within ~20% due to train/test distribution gap) is the official IoU formula.")
    print("  Similarly for RMSE candidates R1-R4.")
    print("  The final 'weighted mean' score on the leaderboard will also tell us whether")
    print("  RMSE is normalized by /30 or not — compare raw vs bldN/vegN columns.")
    print("=" * 95)


if __name__ == "__main__":
    main()
