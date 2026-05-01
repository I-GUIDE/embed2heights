"""
Per-pixel height-RMSE attribution diagnostic.

The bin-level RMSE table (`tools/diagnostic_height_rmse.py`) tells you which
GT-height bin has high error. This script tells you *what kind* of error
inside each bin: bias vs variance, edge vs interior pixels, which patches
contribute the most, and what the leaderboard-weighted ("score-contribution")
breakdown looks like. It is meant to be the answer to "where's the residual
RMSE actually coming from, after head/loss tuning has plateaued".

Three sections:

  1) RMSE² = bias² + variance, per (class, bin). Tells you whether the bin's
     error is *correctable by recalibration* (bias dominates) or whether the
     model genuinely has no signal there (variance dominates).
  2) Edge vs interior split, per (class, bin). Edge = within `--edge-width`
     pixels of a class-mask boundary. Tells you whether the residual is
     concentrated at object boundaries (where pixel-aligned embeddings
     fundamentally smear two classes' features) or in interiors.
  3) Score contribution per bin: each bin's contribution to the per-image
     RMSE that the leaderboard scores against. (Per-bin RMSE alone over-
     reads tall bins, since they are sparse; score contribution is the
     metric that matters.)
  4) Top-K worst patches, with their per-class RMSE and a region tag pulled
     from the filename. Tells you whether the residual mean is dominated by
     a handful of pathological patches.

Run as:
    python tools/diagnostic_height_attribution.py <experiment_name>

Default <experiment_name> is the current raw champion N.
"""
from __future__ import annotations

import collections
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

from tools.diagnostic_height_rmse import (  # noqa: E402
    CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT,
    LABEL_THRESHOLD, build_label_map, load_val_ids, normalize_core_id,
)

BINS = [0, 2, 5, 10, 20, 40, np.inf]
BIN_LABELS = ["0-2", "2-5", "5-10", "10-20", "20-40", "40+"]
RMSE_NORM = {"building": 3.0, "vegetation": 5.0}  # leaderboard ceilings
SCORE_WEIGHT = {"building": 0.25, "vegetation": 0.20}

REGION_RE = re.compile(r"_([A-Z]{2})(?:_|$)")


def _try_scipy_erosion():
    try:
        from scipy.ndimage import binary_erosion
        return binary_erosion
    except Exception:
        return None


_BINARY_EROSION = _try_scipy_erosion()


def edge_mask(mask, width=2):
    """Boolean array — True for mask pixels within `width` of the mask edge.

    Uses scipy.ndimage.binary_erosion when available (fast); falls back to
    a numpy roll-OR implementation otherwise. The fallback is O(width^2)
    in patch-pixel work and fine for width<=3 at 256x256.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    if _BINARY_EROSION is not None:
        eroded = _BINARY_EROSION(mask, iterations=int(width))
        return mask & ~eroded
    # numpy fallback: a pixel is interior if all neighbours within ±width
    # are also in the mask. Use rolling AND.
    interior = mask.copy()
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))
            interior &= shifted
    return mask & ~interior


def region_from_id(core_id):
    m = REGION_RE.search(core_id)
    return m.group(1) if m else "??"


def fmt_pct(x):
    return f"{x*100:5.1f}%"


def main():
    exp_name = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "alphaearth_tessera_iou_fusion_N_base48"
    )
    edge_width = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    pred_dir = os.path.join(REPO_DIR, "runs", exp_name, "predictions")
    labels_dir = "/u/dingqi2/workspace/esa/data/train/labels"
    split_file = os.path.join(REPO_DIR, "splits", "split.json")

    val_ids = load_val_ids(split_file)
    label_map = build_label_map(labels_dir)
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    print(f"exp        : {exp_name}")
    print(f"edge_width : {edge_width} px")
    print(f"val_ids    : {len(val_ids)}  matched preds: ~filtered below")

    # Aggregators. Keyed by (class, bin_idx, region) where region ∈ {edge, int, all}.
    # Store sum, sum_sq, n for diff = pred - label, and sum for label, label_sq
    # (latter only used to validate variance computation).
    classes = ("building", "vegetation")
    regions = ("all", "edge", "int")
    n_bins = len(BIN_LABELS)
    sum_d   = {(c, b, r): 0.0 for c in classes for b in range(n_bins) for r in regions}
    sum_d2  = {(c, b, r): 0.0 for c in classes for b in range(n_bins) for r in regions}
    cnt     = {(c, b, r): 0   for c in classes for b in range(n_bins) for r in regions}

    # Per-image RMSE pools (for top-K + score contribution).
    per_image = []  # list of dict(core_id, region, rmse_b, rmse_v, n_b, n_v, mean_h_b, mean_h_v)

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

        valid = ~np.all(label == 0, axis=0)
        any_class = (label[CH_BUILDING] + label[CH_VEGETATION] + label[CH_WATER]) > 0
        ndsm_hole = (label[CH_HEIGHT] == 0) & any_class
        height_valid = valid & ~ndsm_hole

        bld_pos = (label[CH_BUILDING] > LABEL_THRESHOLD) & height_valid
        veg_pos = (label[CH_VEGETATION] > LABEL_THRESHOLD) & height_valid
        bld_edge = edge_mask(bld_pos, width=edge_width)
        veg_edge = edge_mask(veg_pos, width=edge_width)

        p_h = pred[CH_HEIGHT]
        l_h = label[CH_HEIGHT]
        diff = p_h - l_h

        per_img_entry = {
            "core_id": core_id,
            "region": region_from_id(core_id),
            "rmse_b": np.nan, "rmse_v": np.nan,
            "n_b": int(bld_pos.sum()), "n_v": int(veg_pos.sum()),
            "mean_h_b": float(l_h[bld_pos].mean()) if bld_pos.any() else np.nan,
            "mean_h_v": float(l_h[veg_pos].mean()) if veg_pos.any() else np.nan,
            "max_h_b": float(l_h[bld_pos].max()) if bld_pos.any() else np.nan,
            "max_h_v": float(l_h[veg_pos].max()) if veg_pos.any() else np.nan,
        }
        if bld_pos.any():
            d = diff[bld_pos]
            per_img_entry["rmse_b"] = float(np.sqrt(np.mean(d ** 2)))
        if veg_pos.any():
            d = diff[veg_pos]
            per_img_entry["rmse_v"] = float(np.sqrt(np.mean(d ** 2)))
        per_image.append(per_img_entry)

        # Per-bin per-region accumulators
        for cname, cmask, cedge in (
            ("building",   bld_pos, bld_edge),
            ("vegetation", veg_pos, veg_edge),
        ):
            if not cmask.any():
                continue
            cint = cmask & ~cedge
            l_class = l_h[cmask]
            d_class = diff[cmask]
            d_edge_full = diff[cedge]
            d_int_full  = diff[cint]
            l_edge_full = l_h[cedge]
            l_int_full  = l_h[cint]

            for b_idx, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
                # ALL pixels in class
                m = (l_class >= lo) & (l_class < hi)
                if m.any():
                    d = d_class[m]
                    sum_d[(cname, b_idx, "all")]  += float(d.sum())
                    sum_d2[(cname, b_idx, "all")] += float(np.sum(d * d))
                    cnt[(cname, b_idx, "all")]    += int(d.size)
                # EDGE pixels in class
                me = (l_edge_full >= lo) & (l_edge_full < hi)
                if me.any():
                    d = d_edge_full[me]
                    sum_d[(cname, b_idx, "edge")]  += float(d.sum())
                    sum_d2[(cname, b_idx, "edge")] += float(np.sum(d * d))
                    cnt[(cname, b_idx, "edge")]    += int(d.size)
                # INTERIOR pixels in class
                mi = (l_int_full >= lo) & (l_int_full < hi)
                if mi.any():
                    d = d_int_full[mi]
                    sum_d[(cname, b_idx, "int")]  += float(d.sum())
                    sum_d2[(cname, b_idx, "int")] += float(np.sum(d * d))
                    cnt[(cname, b_idx, "int")]    += int(d.size)

    print(f"matched     : {matched}\n")

    # ---- Section 1: Bias² vs Variance per (class, bin) -----------------------
    print("=" * 100)
    print("Bias² vs Variance per bin     (RMSE² = bias² + variance; "
          "bias-dominated = recalibration helps; variance-dominated = no signal)")
    print("=" * 100)
    print(f"  {'class':<11} {'bin':<6} {'n':>11}  {'rmse':>7}  {'bias':>8}  "
          f"{'sigma':>7}  {'bias²':>7}  {'var':>7}  {'bias²/rmse²':>11}")
    print("  " + "-" * 96)
    section1 = {}
    for cname in classes:
        for b_idx, lab in enumerate(BIN_LABELS):
            n = cnt[(cname, b_idx, "all")]
            if n == 0:
                continue
            sd  = sum_d[(cname, b_idx, "all")]
            sd2 = sum_d2[(cname, b_idx, "all")]
            mean = sd / n
            mse  = sd2 / n
            bias  = mean
            var   = max(mse - bias * bias, 0.0)
            sigma = float(np.sqrt(var))
            rmse  = float(np.sqrt(mse))
            frac_bias = (bias * bias) / mse if mse > 0 else 0.0
            print(f"  {cname:<11} {lab:<6} {n:>11,d}  {rmse:>7.3f}  "
                  f"{bias:>8.3f}  {sigma:>7.3f}  {bias*bias:>7.3f}  "
                  f"{var:>7.3f}  {frac_bias*100:>10.1f}%")
            section1[(cname, lab)] = {
                "n": int(n), "rmse": rmse, "bias": bias, "sigma": sigma,
                "bias_sq_frac": frac_bias,
            }
        print()

    # ---- Section 2: Edge vs Interior per (class, bin) ------------------------
    print("=" * 100)
    print("Edge vs Interior split        (edge = within "
          f"{edge_width} px of class boundary; "
          "edge-dominated → boundary smearing in pixel embeddings)")
    print("=" * 100)
    print(f"  {'class':<11} {'bin':<6}  {'edge n':>10}  {'edge rmse':>10}  "
          f"{'int n':>10}  {'int rmse':>10}  {'edge/all n':>11}  "
          f"{'edge SS share':>14}")
    print("  " + "-" * 96)
    section2 = {}
    for cname in classes:
        for b_idx, lab in enumerate(BIN_LABELS):
            n_a = cnt[(cname, b_idx, "all")]
            n_e = cnt[(cname, b_idx, "edge")]
            n_i = cnt[(cname, b_idx, "int")]
            if n_a == 0:
                continue
            ss_a = sum_d2[(cname, b_idx, "all")]
            ss_e = sum_d2[(cname, b_idx, "edge")]
            ss_i = sum_d2[(cname, b_idx, "int")]
            rmse_e = float(np.sqrt(ss_e / n_e)) if n_e > 0 else float("nan")
            rmse_i = float(np.sqrt(ss_i / n_i)) if n_i > 0 else float("nan")
            edge_n_frac  = n_e / n_a
            edge_ss_frac = ss_e / ss_a if ss_a > 0 else 0.0
            print(f"  {cname:<11} {lab:<6}  {n_e:>10,d}  {rmse_e:>10.3f}  "
                  f"{n_i:>10,d}  {rmse_i:>10.3f}  {edge_n_frac*100:>10.1f}%  "
                  f"{edge_ss_frac*100:>13.1f}%")
            section2[(cname, lab)] = {
                "edge_n_frac": edge_n_frac, "edge_ss_frac": edge_ss_frac,
                "rmse_edge": rmse_e, "rmse_int": rmse_i,
            }
        print()

    # ---- Section 3: Score contribution per bin -------------------------------
    # Per-image RMSE is computed by binning. Per-image SS contribution from
    # bin b = sum(d² in bin b) / N_img. The leaderboard mean is over images,
    # not pixels, so a clean attribution is messy. We approximate by computing
    # the "if this bin's pixels were the only ones contributing, the per-image
    # RMSE would be sqrt(SS_bin / total_class_n_for_that_image)" — and then
    # report the share of total class SS each bin holds, weighted across all
    # images. This matches what the score-mean is actually averaging over,
    # within ε.
    print("=" * 100)
    print("Score contribution per bin    (squared-error share of the "
          "leaderboard RMSE_class; sums to ~100% per class)")
    print("=" * 100)
    print(f"  {'class':<11} {'bin':<6}  {'n':>11}  {'n share':>9}  "
          f"{'SS share':>10}  {'rmse if alone':>14}  "
          f"{'score contrib':>14}")
    print("  " + "-" * 96)
    section3 = {}
    for cname in classes:
        # Use global pool (n × rmse² gives the per-bin SS contribution).
        all_ss = sum(sum_d2[(cname, b, "all")] for b in range(n_bins))
        all_n  = sum(cnt[(cname, b, "all")]    for b in range(n_bins))
        if all_n == 0 or all_ss <= 0:
            print(f"  {cname:<11}: empty"); print(); continue
        for b_idx, lab in enumerate(BIN_LABELS):
            n   = cnt[(cname, b_idx, "all")]
            ss  = sum_d2[(cname, b_idx, "all")]
            if n == 0:
                continue
            share_n  = n / all_n
            share_ss = ss / all_ss
            # If only this bin contributed, the global-pool RMSE for that
            # class would be sqrt(ss/n) — the same as the bin RMSE shown in
            # diagnostic_height_rmse.py. We re-derive the global-pool RMSE
            # for the class (not per-image) and read off the fraction.
            class_rmse_global = float(np.sqrt(all_ss / all_n))
            # Score contribution: how much score-pts does this bin's SS share
            # cost us, relative to the bin being magically zero-error?
            # ΔRMSE_class ≈ class_rmse_global × (1 - sqrt(1 - share_ss))
            # Δscore = ΔRMSE_class / RMSE_NORM[class] × SCORE_WEIGHT[class]
            d_rmse = class_rmse_global * (1.0 - np.sqrt(max(1.0 - share_ss, 0.0)))
            d_score = d_rmse / RMSE_NORM[cname] * SCORE_WEIGHT[cname]
            print(f"  {cname:<11} {lab:<6}  {n:>11,d}  {share_n*100:>8.2f}%  "
                  f"{share_ss*100:>9.2f}%  {np.sqrt(ss/n):>14.3f}  "
                  f"{d_score*100:>13.3f}%")
            section3[(cname, lab)] = {
                "n_share": share_n, "ss_share": share_ss,
                "delta_score_pct": d_score * 100,
            }
        print()

    # ---- Section 4: Top-K worst patches --------------------------------------
    print("=" * 100)
    print("Top-15 patches by per-image building RMSE (>= 200 building px only)")
    print("=" * 100)
    bld_sortable = [p for p in per_image if not np.isnan(p["rmse_b"]) and p["n_b"] >= 200]
    bld_sortable.sort(key=lambda p: -p["rmse_b"])
    print(f"  {'core_id':<25} {'region':>6}  {'n_bld':>7}  {'rmse_b':>7}  "
          f"{'mean_h':>7}  {'max_h':>7}")
    for p in bld_sortable[:15]:
        print(f"  {p['core_id']:<25} {p['region']:>6}  {p['n_b']:>7,d}  "
              f"{p['rmse_b']:>7.2f}  {p['mean_h_b']:>7.2f}  "
              f"{p['max_h_b']:>7.2f}")

    print()
    print("=" * 100)
    print("Top-15 patches by per-image vegetation RMSE (>= 1000 vegetation px only)")
    print("=" * 100)
    veg_sortable = [p for p in per_image if not np.isnan(p["rmse_v"]) and p["n_v"] >= 1000]
    veg_sortable.sort(key=lambda p: -p["rmse_v"])
    print(f"  {'core_id':<25} {'region':>6}  {'n_veg':>8}  {'rmse_v':>7}  "
          f"{'mean_h':>7}  {'max_h':>7}")
    for p in veg_sortable[:15]:
        print(f"  {p['core_id']:<25} {p['region']:>6}  {p['n_v']:>8,d}  "
              f"{p['rmse_v']:>7.2f}  {p['mean_h_v']:>7.2f}  "
              f"{p['max_h_v']:>7.2f}")

    # ---- Section 5: Region distribution of worst patches --------------------
    print()
    print("=" * 100)
    print("Region split — top-25% worst patches per class vs full val")
    print("=" * 100)
    full_region_b = collections.Counter(p["region"] for p in per_image if p["n_b"] >= 200)
    full_region_v = collections.Counter(p["region"] for p in per_image if p["n_v"] >= 1000)
    n_top_b = max(1, len(bld_sortable) // 4)
    n_top_v = max(1, len(veg_sortable) // 4)
    bad_region_b = collections.Counter(p["region"] for p in bld_sortable[:n_top_b])
    bad_region_v = collections.Counter(p["region"] for p in veg_sortable[:n_top_v])

    def _share(c, total):
        return {k: v / total for k, v in c.items()} if total else {}
    print("  Building (val n_b>=200):")
    for region in sorted(full_region_b):
        full = full_region_b[region] / max(sum(full_region_b.values()), 1)
        bad  = bad_region_b.get(region, 0) / max(n_top_b, 1)
        delta = bad - full
        marker = " ❗" if delta > 0.05 else ("  ✓" if delta < -0.03 else "")
        print(f"    {region:>4} : full={fmt_pct(full)}  worst25%={fmt_pct(bad)}  "
              f"Δ={delta*100:+5.1f} pp{marker}")
    print("  Vegetation (val n_v>=1000):")
    for region in sorted(full_region_v):
        full = full_region_v[region] / max(sum(full_region_v.values()), 1)
        bad  = bad_region_v.get(region, 0) / max(n_top_v, 1)
        delta = bad - full
        marker = " ❗" if delta > 0.05 else ("  ✓" if delta < -0.03 else "")
        print(f"    {region:>4} : full={fmt_pct(full)}  worst25%={fmt_pct(bad)}  "
              f"Δ={delta*100:+5.1f} pp{marker}")

    # JSON dump
    out = {
        "experiment": exp_name,
        "edge_width": edge_width,
        "n_val_images": matched,
        "bias_var_per_bin":   {f"{c}|{l}": section1[(c, l)] for (c, l) in section1},
        "edge_int_per_bin":   {f"{c}|{l}": section2[(c, l)] for (c, l) in section2},
        "score_contrib_per_bin": {f"{c}|{l}": section3[(c, l)] for (c, l) in section3},
        "top_worst_building": [
            {k: v for k, v in p.items()} for p in bld_sortable[:15]
        ],
        "top_worst_vegetation": [
            {k: v for k, v in p.items()} for p in veg_sortable[:15]
        ],
    }
    out_path = os.path.join(REPO_DIR, "runs", exp_name, "height_attribution.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    main()
