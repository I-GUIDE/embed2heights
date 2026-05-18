"""Per-region (tile-suffix) threshold sweep.

Hypothesis: tiles from different regions (KE, JE, etc.) have systematically
different optimal binarization thresholds. If true, applying per-region
thresholds at submission time should improve overall score vs a single
global per-class triple. KE-region is known to dominate the worst-tile list
on the 6-seed e100 ensemble fold0 val.

Aggregation matches the leaderboard convention:
  - macro-IoU per class (mean of per-tile IoU with empty-case=1.0)
  - macro-RMSE per class (mean of per-tile RMSE on masked pixels)

Optimization: greedy per-class threshold sweep, maximizing the local
overall score within each region's tiles. Output: per-region (t_b,t_v,t_w)
plus aggregated overall score using those thresholds.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.inference.calibration import (  # noqa: E402
    apply_water_cc_filter,
    load_labeled_predictions,
    mean_valid,
)
from core.metrics import LABEL_THRESHOLD, WEIGHTS, binary_iou  # noqa: E402


SUFFIX_RE = re.compile(r"_([A-Z]+)(?:_\d{4})?$")


def get_region(core_id):
    m = SUFFIX_RE.search(core_id)
    return m.group(1) if m else "??"


def score_from_components(iou_b, iou_t, iou_w, rmse_b, rmse_v):
    return (
        WEIGHTS["iou_buildings"] * iou_b
        + WEIGHTS["iou_trees"] * iou_t
        + WEIGHTS["iou_water"] * iou_w
        + WEIGHTS["RMSE_building_height"] * max(0.0, 1.0 - rmse_b / 3.0)
        + WEIGHTS["RMSE_vegetation_height"] * max(0.0, 1.0 - rmse_v / 5.0)
    )


def macro_evaluate(predictions, labels, ids, thresholds, k_water=0):
    """Macro-IoU + macro-RMSE matching the leaderboard scoring convention."""
    iou_lists = [[], [], []]
    rmse_bld, rmse_veg = [], []
    for cid in ids:
        pred = predictions[cid]; label = labels[cid]
        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        pred = pred[:, :h, :w]; label = label[:, :h, :w]
        for c in range(3):
            pm = pred[c] > thresholds[c]
            if c == 2 and k_water > 0:
                pm = apply_water_cc_filter(pm, k_water)
            gt = label[c] > LABEL_THRESHOLD
            iou_lists[c].append(binary_iou(pm, gt))
        for ch, bucket in ((0, rmse_bld), (1, rmse_veg)):
            mask = label[ch] > LABEL_THRESHOLD
            if mask.any():
                diff = pred[3][mask] - label[3][mask]
                bucket.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))
    iou_b = mean_valid(iou_lists[0])
    iou_t = mean_valid(iou_lists[1])
    iou_w = mean_valid(iou_lists[2])
    rmse_b = mean_valid(rmse_bld) if rmse_bld else 3.0
    rmse_v = mean_valid(rmse_veg) if rmse_veg else 5.0
    return (iou_b, iou_t, iou_w, rmse_b, rmse_v,
            score_from_components(iou_b, iou_t, iou_w, rmse_b, rmse_v))


def sweep_region(predictions, labels, ids, k_water=14, grid=None):
    """Greedy per-class sweep to find optimal (t_b, t_v, t_w) for one region."""
    if grid is None:
        grid = np.round(np.arange(0.30, 0.95 + 1e-9, 0.01), 3)
    best_t = [0.5, 0.5, 0.5]
    for c in range(3):
        best_score = -1.0
        best_th = best_t[c]
        for th in grid:
            cand = list(best_t); cand[c] = float(th)
            kw = k_water if c == 2 else 0
            *_, s = macro_evaluate(predictions, labels, ids, cand, k_water=kw)
            if s > best_score:
                best_score = s; best_th = float(th)
        best_t[c] = best_th
    return tuple(best_t)


def evaluate_with_per_region_thresholds(predictions, labels, per_region_thresh,
                                       region_ids, k_water=14):
    """Score using region-specific thresholds, aggregated macro across all val tiles."""
    iou_lists = [[], [], []]
    rmse_bld, rmse_veg = [], []
    for reg, ids in region_ids.items():
        t = per_region_thresh[reg]
        for cid in ids:
            pred = predictions[cid]; label = labels[cid]
            h = min(pred.shape[1], label.shape[1])
            w = min(pred.shape[2], label.shape[2])
            pred = pred[:, :h, :w]; label = label[:, :h, :w]
            for c in range(3):
                pm = pred[c] > t[c]
                if c == 2 and k_water > 0:
                    pm = apply_water_cc_filter(pm, k_water)
                gt = label[c] > LABEL_THRESHOLD
                iou_lists[c].append(binary_iou(pm, gt))
            for ch, bucket in ((0, rmse_bld), (1, rmse_veg)):
                mask = label[ch] > LABEL_THRESHOLD
                if mask.any():
                    diff = pred[3][mask] - label[3][mask]
                    bucket.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))
    iou_b = mean_valid(iou_lists[0])
    iou_t = mean_valid(iou_lists[1])
    iou_w = mean_valid(iou_lists[2])
    rmse_b = mean_valid(rmse_bld) if rmse_bld else 3.0
    rmse_v = mean_valid(rmse_veg) if rmse_veg else 5.0
    return (iou_b, iou_t, iou_w, rmse_b, rmse_v,
            score_from_components(iou_b, iou_t, iou_w, rmse_b, rmse_v))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--k-water", type=int, default=14)
    p.add_argument("--min-region-tiles", type=int, default=10,
                   help="Skip regions with fewer than this many val tiles.")
    p.add_argument("--global-t", nargs=3, type=float, default=[0.640, 0.570, 0.820],
                   help="Global per-class threshold baseline (default: 0.640/0.570/0.820 — known fold0 optimum).")
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    print(f"Loading predictions from {args.pred_dir}")
    predictions, labels = load_labeled_predictions(args.pred_dir, args.labels_dir, args.split_file)
    print(f"  {len(predictions)} val tiles")

    region_ids = defaultdict(list)
    for cid in predictions:
        region_ids[get_region(cid)].append(cid)
    print(f"\nVal region distribution:")
    for reg, ids in sorted(region_ids.items(), key=lambda x: -len(x[1])):
        print(f"  {reg}: {len(ids)}")

    GLOBAL_T = tuple(args.global_t)
    iou_b, iou_t, iou_w, rb, rv, gs = macro_evaluate(
        predictions, labels, list(predictions.keys()), GLOBAL_T, k_water=args.k_water)
    print(f"\nGlobal baseline (t={GLOBAL_T}, K_water={args.k_water}):")
    print(f"  iou_b={iou_b:.4f} iou_t={iou_t:.4f} iou_w={iou_w:.4f}  RMSE_b={rb:.3f} RMSE_v={rv:.3f}  score={gs:.4f}")

    print(f"\nPer-region sweep (regions with >={args.min_region_tiles} tiles):")
    per_region_thresh = {}
    for reg, ids in sorted(region_ids.items(), key=lambda x: -len(x[1])):
        if len(ids) < args.min_region_tiles:
            per_region_thresh[reg] = GLOBAL_T
            continue
        t = sweep_region(predictions, labels, ids, k_water=args.k_water)
        per_region_thresh[reg] = t
        # Per-region scores under global vs region thresholds
        *_, sg = macro_evaluate(predictions, labels, ids, GLOBAL_T, k_water=args.k_water)
        *_, sr = macro_evaluate(predictions, labels, ids, t, k_water=args.k_water)
        print(f"  {reg} (n={len(ids)}): t=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f}) "
              f"Δt=({t[0]-GLOBAL_T[0]:+.2f},{t[1]-GLOBAL_T[1]:+.2f},{t[2]-GLOBAL_T[2]:+.2f}) "
              f"local_score: {sg:.4f}→{sr:.4f} (Δ{sr-sg:+.4f})")

    iou_b, iou_t, iou_w, rb, rv, ps = evaluate_with_per_region_thresholds(
        predictions, labels, per_region_thresh, region_ids, k_water=args.k_water)
    print(f"\nPer-region thresholds (overall):")
    print(f"  iou_b={iou_b:.4f} iou_t={iou_t:.4f} iou_w={iou_w:.4f}  RMSE_b={rb:.3f} RMSE_v={rv:.3f}  score={ps:.4f}")
    print(f"\nDelta vs global: {ps - gs:+.4f}")

    if args.output_json:
        out = {
            "global_thresholds": list(GLOBAL_T),
            "global_score": float(gs),
            "per_region_thresholds": {r: list(t) for r, t in per_region_thresh.items()},
            "per_region_score": float(ps),
            "k_water": args.k_water,
            "delta": float(ps - gs),
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
