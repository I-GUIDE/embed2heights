"""
Compare predicted height distributions between val and test sets.

Why: when the val/test split is held out from the same source, the *input*
(embedding) distribution can still drift between val and test (different
geographies / years / sensors). We can't read test-set GT heights, but we
*can* read the model's predicted heights. If pred_test deviates from
pred_val in a way that pred_val doesn't deviate from gt_val, that's a
signal of input drift; the model is being asked to predict on inputs it
was never trained near.

This script reads three height pools from a single experiment:
  - gt_val:   ground-truth heights on val patches (from labels/)
  - pred_val: model predictions on val patches (from runs/<exp>/predictions/)
  - pred_test: model predictions on test patches (from
               runs/<exp>/predictions_test/, produced by predict.py without
               --test-targets-dir)

and reports:
  1) Coarse summary stats (mean/median/p95/p99/max) for each pool.
  2) Fraction of pixels in each leaderboard-relevant bin (0-2/2-5/.../40+).
  3) KS distance between pred_val and pred_test (no labels needed).
  4) Fraction of pixels above tall-bin thresholds (10/20/40 m).
  5) Saves a 2-panel matplotlib figure (linear and log y axes).

Usage:
    python tools/diagnostic_height_histogram_drift.py <experiment_name>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import rasterio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, REPO_DIR)

# Reuse the same id-normalisation as the RMSE diagnostic.
from tools.diagnostic_height_rmse import (  # noqa: E402
    CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT,
    LABEL_THRESHOLD, build_label_map, load_val_ids, normalize_core_id,
)

BINS = [0, 2, 5, 10, 20, 40, np.inf]
BIN_LABELS = ["0-2", "2-5", "5-10", "10-20", "20-40", "40+"]


def collect_pred_heights(pred_files, valid_id_set=None):
    """Return a flat 1D float32 array of predicted heights across all files.

    A NumPy prediction file is shape (4, H, W); we read only channel 3 to
    keep memory low. If valid_id_set is supplied, only files whose
    normalised core-id is in the set contribute (used for filtering
    runs/<exp>/predictions/ down to val patches).
    """
    chunks = []
    for pf in pred_files:
        if valid_id_set is not None:
            cid = normalize_core_id(pf)
            if cid not in valid_id_set:
                continue
        arr = np.load(pf)
        h = arr[CH_HEIGHT].astype(np.float32).reshape(-1)
        chunks.append(h)
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


def collect_gt_heights(label_map, val_ids):
    """Return a flat 1D float32 array of GT heights on valid val pixels.

    "Valid" = global_valid (not full-zero label) AND not nDSM hole. Same
    convention as the RMSE diagnostic, so the histograms are comparable.
    """
    chunks = []
    for cid in sorted(val_ids):
        if cid not in label_map:
            continue
        with rasterio.open(label_map[cid]) as src:
            label = src.read().astype(np.float32)
        valid = ~np.all(label == 0, axis=0)
        any_class = (label[CH_BUILDING] + label[CH_VEGETATION] + label[CH_WATER]) > 0
        ndsm_hole = (label[CH_HEIGHT] == 0) & any_class
        height_valid = valid & ~ndsm_hole
        h = label[CH_HEIGHT][height_valid].reshape(-1)
        chunks.append(h)
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


def summarize(name, arr):
    if arr.size == 0:
        return {"name": name, "n": 0}
    p = np.percentile(arr, [5, 50, 90, 95, 99])
    return {
        "name": name,
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p5": float(p[0]),
        "p50": float(p[1]),
        "p90": float(p[2]),
        "p95": float(p[3]),
        "p99": float(p[4]),
        "max": float(arr.max()),
    }


def bin_fractions(arr):
    if arr.size == 0:
        return {lab: 0.0 for lab in BIN_LABELS}
    out = {}
    for lo, hi, lab in zip(BINS[:-1], BINS[1:], BIN_LABELS):
        m = (arr >= lo) & (arr < hi)
        out[lab] = float(m.mean())
    return out


def tall_fractions(arr):
    """Cumulative tail fractions — the tall-side P(>= t)."""
    if arr.size == 0:
        return {f">={t}m": 0.0 for t in (10, 20, 40)}
    return {f">={t}m": float((arr >= t).mean()) for t in (10, 20, 40)}


def ks_distance(a, b, n_grid=2001):
    """Empirical KS distance D = sup_x |F_a(x) - F_b(x)|, on a shared grid.

    SciPy isn't always available on the cluster; this is the same statistic
    via numpy.searchsorted on a shared evaluation grid. n_grid=2001 is more
    than enough resolution for height histograms in [0, 80].
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    grid = np.linspace(lo, hi, n_grid)
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    fa = np.searchsorted(a_sorted, grid, side="right") / a_sorted.size
    fb = np.searchsorted(b_sorted, grid, side="right") / b_sorted.size
    return float(np.max(np.abs(fa - fb)))


def plot_histograms(pools, out_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN: matplotlib unavailable; skipping plot: {exc}")
        return None

    edges = np.concatenate([np.linspace(0, 40, 81), np.array([50, 60, 80, 120])])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"gt_val": "k", "pred_val": "C0", "pred_test": "C3"}
    for ax, log in zip(axes, [False, True]):
        for name, arr in pools.items():
            if arr.size == 0:
                continue
            ax.hist(arr.clip(max=120), bins=edges, density=True, histtype="step",
                    linewidth=1.6, label=f"{name} (n={arr.size:,})",
                    color=colors.get(name, None))
        ax.set_xlabel("predicted / GT height (m)")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(("log-y, " if log else "") + title)
        if log:
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    if len(sys.argv) < 2:
        print("usage: python tools/diagnostic_height_histogram_drift.py <experiment_name>")
        sys.exit(1)
    exp_name = sys.argv[1]
    exp_dir = os.path.join(REPO_DIR, "runs", exp_name)
    pred_dir_val = os.path.join(exp_dir, "predictions")
    pred_dir_test = os.path.join(exp_dir, "predictions_test")
    labels_dir = "/u/dingqi2/workspace/esa/data/train/labels"
    split_file = os.path.join(REPO_DIR, "splits", "split.json")

    if not os.path.isdir(pred_dir_val):
        raise SystemExit(f"missing val predictions: {pred_dir_val}")
    if not os.path.isdir(pred_dir_test):
        raise SystemExit(
            f"missing test predictions: {pred_dir_test}\n"
            "Run predict.py without --test-targets-dir against the test "
            "embeddings dir, with --predictions-dir set to this path."
        )

    val_ids = load_val_ids(split_file)
    label_map = build_label_map(labels_dir)
    val_pred_files = sorted(glob.glob(os.path.join(pred_dir_val, "*.npy")))
    test_pred_files = sorted(glob.glob(os.path.join(pred_dir_test, "*.npy")))
    print(f"exp                : {exp_name}")
    print(f"val ids            : {len(val_ids)}")
    print(f"pred files (val dir): {len(val_pred_files)}  (filtering to val ids)")
    print(f"pred files (test)  : {len(test_pred_files)}")

    print("\nLoading pred_val ...")
    pred_val = collect_pred_heights(val_pred_files, valid_id_set=val_ids)
    print(f"  pred_val: {pred_val.size:,} pixels")
    print("Loading pred_test ...")
    pred_test = collect_pred_heights(test_pred_files)
    print(f"  pred_test: {pred_test.size:,} pixels")
    print("Loading gt_val ...")
    gt_val = collect_gt_heights(label_map, val_ids)
    print(f"  gt_val: {gt_val.size:,} pixels")

    pools = {"gt_val": gt_val, "pred_val": pred_val, "pred_test": pred_test}

    print("\n" + "=" * 78)
    print("Summary stats (meters)")
    print("=" * 78)
    summaries = {name: summarize(name, arr) for name, arr in pools.items()}
    for s in summaries.values():
        if s["n"] == 0:
            print(f"  {s['name']:<10}: empty")
            continue
        print(
            f"  {s['name']:<10}: n={s['n']:>13,d}  mean={s['mean']:6.3f}  "
            f"std={s['std']:6.3f}  p50={s['p50']:5.2f}  p95={s['p95']:6.2f}  "
            f"p99={s['p99']:6.2f}  max={s['max']:6.2f}"
        )

    print("\n" + "=" * 78)
    print("Bin-fraction (probability mass per leaderboard-relevant bin)")
    print("=" * 78)
    fracs = {name: bin_fractions(arr) for name, arr in pools.items()}
    header = f"  {'pool':<10} | " + " ".join(f"{lab:>7}" for lab in BIN_LABELS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, f in fracs.items():
        cells = " ".join(f"{f[lab]*100:6.2f}%" for lab in BIN_LABELS)
        print(f"  {name:<10} | {cells}")

    print("\n" + "=" * 78)
    print("Tall-side cumulative tail fractions")
    print("=" * 78)
    tails = {name: tall_fractions(arr) for name, arr in pools.items()}
    for name, t in tails.items():
        cells = "  ".join(f"{k}={v*100:5.3f}%" for k, v in t.items())
        print(f"  {name:<10}: {cells}")

    print("\n" + "=" * 78)
    print("Distribution distance (label-free drift signal)")
    print("=" * 78)
    ks_pv_pt = ks_distance(pred_val, pred_test)
    ks_gv_pv = ks_distance(gt_val, pred_val)
    ks_gv_pt = ks_distance(gt_val, pred_test)
    print(f"  KS(pred_val, pred_test) = {ks_pv_pt:.4f}   <- pure model-side distribution shift")
    print(f"  KS(gt_val, pred_val)    = {ks_gv_pv:.4f}   <- val calibration error baseline")
    print(f"  KS(gt_val, pred_test)   = {ks_gv_pt:.4f}   <- compound (calibration + drift)")
    if not np.isnan(ks_pv_pt) and not np.isnan(ks_gv_pv):
        print(f"  drift / calibration ratio = {ks_pv_pt / max(ks_gv_pv, 1e-6):.3f}")
        print(
            "  Reading: ratio < 1 means val/test predictions are closer to each other "
            "than the model is to its own val labels — i.e. drift is small relative "
            "to known calibration error. Ratio > 1 is the warning sign."
        )

    fig_path = os.path.join(exp_dir, "height_histogram_drift.png")
    plot_path = plot_histograms(pools, fig_path, title=f"{exp_name} — height distribution")
    if plot_path:
        print(f"\nSaved figure: {plot_path}")

    # JSON dump for downstream consumption
    out = {
        "experiment": exp_name,
        "summaries": summaries,
        "bin_fractions": fracs,
        "tall_fractions": tails,
        "ks": {
            "pred_val__pred_test": ks_pv_pt,
            "gt_val__pred_val": ks_gv_pv,
            "gt_val__pred_test": ks_gv_pt,
        },
    }
    out_json = os.path.join(exp_dir, "height_histogram_drift.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved JSON  : {out_json}")


if __name__ == "__main__":
    main()
