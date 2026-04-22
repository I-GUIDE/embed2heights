"""
Height RMSE diagnostic — why is RMSE_H_VEG so much higher than RMSE_H_BUILD?

For the champion run (E_specialist_d2), computes on the val split:
  1) Leaderboard RMSE_bH / RMSE_vH (sanity-check reproduction).
  2) Full-image RMSE (all valid pixels) and "other" RMSE (valid minus bld/veg).
  3) Height-distribution stats per GT class (building / vegetation / other).
  4) Per-image RMSE histograms — how many outlier patches are driving the mean?

Run as a single-process script (no torch needed, label+prediction i/o only):
    python tools/diagnostic_height_rmse.py

Outputs a plain-text summary to stdout and saves a JSON with raw numbers next
to the script for later consumption.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import rasterio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, REPO_DIR)

# Inlined from core.dataset / core.metrics to avoid torch import on the login node.
CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT = 0, 1, 2, 3
LABEL_THRESHOLD = 0.0
_EMB_PREFIXES = ("gee_emb_", "tessera_emb_", "s2_", "emb_")
_EMB_SUFFIXES = ("_embeddings", "_embedding", "_quantized")
_YEAR_RE = re.compile(r"_(20\d{2})")


def _strip_prefixes(s, prefs):
    for p in prefs:
        if s.startswith(p):
            return s[len(p):]
    return s


def _strip_suffixes(s, sufs):
    for suf in sufs:
        if s.endswith(suf):
            return s[:-len(suf)]
    return s


def normalize_core_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    base = _strip_prefixes(base, ("label_", "pred_"))
    base = _strip_prefixes(base, _EMB_PREFIXES)
    base = _strip_suffixes(base, _EMB_SUFFIXES)
    return _YEAR_RE.sub("", base)


def build_label_map(labels_dir):
    return {
        normalize_core_id(p): p
        for p in glob.glob(os.path.join(labels_dir, "**", "label_*.tif"), recursive=True)
    }


def load_val_ids(split_file):
    with open(split_file) as f:
        return set(json.load(f)["val"])


def rmse(diff_vals):
    if diff_vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff_vals.astype(np.float64) ** 2)))


def summarize(name, arr):
    if arr.size == 0:
        return f"  {name:<28}: empty"
    p = np.percentile(arr, [5, 25, 50, 75, 95])
    return (f"  {name:<28}: n={arr.size:>12,d}  mean={arr.mean():7.3f}  "
            f"std={arr.std():7.3f}  p5/50/95={p[0]:6.2f}/{p[2]:6.2f}/{p[4]:6.2f}")


def main():
    exp_name = "alphaearth_tessera_iou_fusion_E_specialist_d2"
    pred_dir = os.path.join(REPO_DIR, "runs", exp_name, "predictions")
    labels_dir = "/u/dingqi2/workspace/esa/data/train/labels"
    split_file = os.path.join(REPO_DIR, "splits", "split.json")

    val_ids = load_val_ids(split_file)
    label_map = build_label_map(labels_dir)
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    print(f"exp       : {exp_name}")
    print(f"val_ids   : {len(val_ids)}")
    print(f"pred_files: {len(pred_files)}")

    # Buckets (concatenated pixel values across images)
    bld_diff, veg_diff, other_diff, all_diff = [], [], [], []
    bld_gt, veg_gt, other_gt, all_gt = [], [], [], []

    # Per-image RMSE lists (mean of these is the leaderboard number)
    rmse_b_per_img, rmse_v_per_img, rmse_other_per_img, rmse_all_per_img = [], [], [], []

    matched = 0
    for pf in pred_files:
        core_id = normalize_core_id(pf)
        if core_id not in label_map or core_id not in val_ids:
            continue
        matched += 1

        pred = np.load(pf).astype(np.float32)
        with rasterio.open(label_map[core_id]) as src:
            label = src.read().astype(np.float32)

        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        pred, label = pred[:, :h, :w], label[:, :h, :w]

        # Validity: label nodata = all 4 bands 0
        valid = ~np.all(label == 0, axis=0)
        # Height validity: drop pixels where label_height is exactly 0 with some
        # class present (nDSM hole). Same convention as losses.py.
        any_class = (label[CH_BUILDING] + label[CH_VEGETATION] + label[CH_WATER]) > 0
        height_hole = (label[CH_HEIGHT] == 0) & any_class
        height_valid = valid & ~height_hole

        bld_mask = (label[CH_BUILDING] > LABEL_THRESHOLD) & height_valid
        veg_mask = (label[CH_VEGETATION] > LABEL_THRESHOLD) & height_valid
        # "Other" = valid pixels that are NEITHER building NOR vegetation
        # (= bare ground / water / other). These are usually low-height.
        other_mask = height_valid & ~bld_mask & ~veg_mask

        p_h = pred[CH_HEIGHT]
        l_h = label[CH_HEIGHT]

        if bld_mask.any():
            d = p_h[bld_mask] - l_h[bld_mask]
            bld_diff.append(d)
            bld_gt.append(l_h[bld_mask])
            rmse_b_per_img.append(rmse(d))
        if veg_mask.any():
            d = p_h[veg_mask] - l_h[veg_mask]
            veg_diff.append(d)
            veg_gt.append(l_h[veg_mask])
            rmse_v_per_img.append(rmse(d))
        if other_mask.any():
            d = p_h[other_mask] - l_h[other_mask]
            other_diff.append(d)
            other_gt.append(l_h[other_mask])
            rmse_other_per_img.append(rmse(d))
        if height_valid.any():
            d = p_h[height_valid] - l_h[height_valid]
            all_diff.append(d)
            all_gt.append(l_h[height_valid])
            rmse_all_per_img.append(rmse(d))

    print(f"matched   : {matched}")
    print()

    # Concatenate
    bld_diff = np.concatenate(bld_diff) if bld_diff else np.empty(0, dtype=np.float64)
    veg_diff = np.concatenate(veg_diff) if veg_diff else np.empty(0, dtype=np.float64)
    other_diff = np.concatenate(other_diff) if other_diff else np.empty(0, dtype=np.float64)
    all_diff = np.concatenate(all_diff) if all_diff else np.empty(0, dtype=np.float64)
    bld_gt = np.concatenate(bld_gt) if bld_gt else np.empty(0, dtype=np.float64)
    veg_gt = np.concatenate(veg_gt) if veg_gt else np.empty(0, dtype=np.float64)
    other_gt = np.concatenate(other_gt) if other_gt else np.empty(0, dtype=np.float64)
    all_gt = np.concatenate(all_gt) if all_gt else np.empty(0, dtype=np.float64)

    print("=" * 78)
    print("RMSE (two aggregations: per-image mean [leaderboard] vs global pixel-pool)")
    print("=" * 78)

    def rmse_pair(name, per_img_list, global_diff):
        pi = np.array(per_img_list, dtype=np.float64) if per_img_list else np.empty(0)
        per_img = float(pi.mean()) if pi.size else float("nan")
        glob = rmse(global_diff)
        print(f"  {name:<28}: per_image={per_img:6.4f}   global={glob:6.4f}   "
              f"n_images={pi.size:>4d}  n_pixels={global_diff.size:>11,d}")
        return per_img, glob

    rb_i, rb_g = rmse_pair("RMSE building (label>0)", rmse_b_per_img, bld_diff)
    rv_i, rv_g = rmse_pair("RMSE vegetation (label>0)", rmse_v_per_img, veg_diff)
    ro_i, ro_g = rmse_pair("RMSE other (~bld,~veg)", rmse_other_per_img, other_diff)
    ra_i, ra_g = rmse_pair("RMSE full image (all valid)", rmse_all_per_img, all_diff)

    print()
    print("=" * 78)
    print("GT height distribution per class (what the model is actually asked to predict)")
    print("=" * 78)
    print(summarize("GT height on building px", bld_gt))
    print(summarize("GT height on vegetation px", veg_gt))
    print(summarize("GT height on other px", other_gt))
    print(summarize("GT height on all valid px", all_gt))

    print()
    print("=" * 78)
    print("Prediction error distribution per class")
    print("=" * 78)
    print(summarize("err building (pred-label)", bld_diff))
    print(summarize("err vegetation (pred-label)", veg_diff))
    print(summarize("err other (pred-label)", other_diff))
    print(summarize("err all valid (pred-label)", all_diff))

    print()
    print("=" * 78)
    print("Per-image RMSE distribution — what drives the sample-averaged metric?")
    print("=" * 78)

    def dump_per_img(name, per_img_list):
        if not per_img_list:
            print(f"  {name}: empty"); return
        arr = np.array(per_img_list, dtype=np.float64)
        p = np.percentile(arr, [5, 25, 50, 75, 90, 95, 99])
        print(f"  {name}: n={arr.size:>4d}  mean={arr.mean():5.3f}  "
              f"p5/50/95/99 = {p[0]:5.2f}/{p[2]:5.2f}/{p[5]:5.2f}/{p[6]:5.2f}  "
              f"max={arr.max():5.2f}")
        # Fraction of mean contributed by the top-5% worst patches
        thr = np.percentile(arr, 95)
        tail_contrib = float(arr[arr >= thr].sum() / arr.sum()) if arr.sum() > 0 else 0.0
        print(f"      top-5% tail contributes {tail_contrib*100:5.1f}% of the sum (i.e. of the mean)")

    dump_per_img("building per-image RMSE", rmse_b_per_img)
    dump_per_img("vegetation per-image RMSE", rmse_v_per_img)
    dump_per_img("full-image per-image RMSE", rmse_all_per_img)

    # Save JSON
    out = {
        "experiment": exp_name,
        "n_val_images": matched,
        "rmse": {
            "building_leaderboard":   rb_i,
            "building_global":        rb_g,
            "vegetation_leaderboard": rv_i,
            "vegetation_global":      rv_g,
            "other_leaderboard":      ro_i,
            "other_global":           ro_g,
            "full_leaderboard":       ra_i,
            "full_global":            ra_g,
        },
        "gt_height_stats": {
            "building":   dict(n=int(bld_gt.size), mean=float(bld_gt.mean() if bld_gt.size else 0),
                               std=float(bld_gt.std() if bld_gt.size else 0)),
            "vegetation": dict(n=int(veg_gt.size), mean=float(veg_gt.mean() if veg_gt.size else 0),
                               std=float(veg_gt.std() if veg_gt.size else 0)),
            "other":      dict(n=int(other_gt.size), mean=float(other_gt.mean() if other_gt.size else 0),
                               std=float(other_gt.std() if other_gt.size else 0)),
            "all_valid":  dict(n=int(all_gt.size), mean=float(all_gt.mean() if all_gt.size else 0),
                               std=float(all_gt.std() if all_gt.size else 0)),
        },
    }
    out_path = os.path.join(REPO_DIR, "runs", exp_name, "height_rmse_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
