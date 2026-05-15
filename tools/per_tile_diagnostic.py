"""Per-tile diagnostic: compute per-tile score for a prediction directory at
fixed thresholds + optional K-water, and report the worst N tiles, the score
distribution, and the per-class precision/recall at chosen thresholds.

Use after sweep_thresholds.py reveals the optimal config — this drills into
which tiles are bottlenecking the score.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.data.discovery import normalize_core_id  # noqa: E402
from core.metrics import WEIGHTS  # noqa: E402


def remove_small_components(mask, min_size):
    if min_size <= 0:
        return mask
    labels, n = label(mask.astype(np.uint8))
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = counts >= min_size
    return keep[labels].astype(mask.dtype)


def compute_metrics(pred, label_arr, thresholds, water_k=0):
    """pred: (4, H, W) probabilities + height. label: (4, H, W) fractions + height."""
    # Binarize predictions for classes
    binp = np.zeros((3, *pred.shape[1:]), dtype=np.float32)
    for c, t in enumerate(thresholds):
        binp[c] = (pred[c] > t).astype(np.float32)
    if water_k > 0:
        binp[2] = remove_small_components(binp[2], water_k)
    # Label: leaderboard convention LABEL_THRESHOLD=0
    binl = (label_arr[:3] > 0).astype(np.float32)

    metrics = {}
    for cls, name in enumerate(["bld", "tree", "wat"]):
        tp = (binp[cls] * binl[cls]).sum()
        fp = (binp[cls] * (1 - binl[cls])).sum()
        fn = ((1 - binp[cls]) * binl[cls]).sum()
        denom = tp + fp + fn
        metrics[f"iou_{name}"] = float(tp / (denom + 1e-6)) if denom > 0 else 1.0
        metrics[f"prec_{name}"] = float(tp / (tp + fp + 1e-6)) if (tp + fp) > 0 else 0.0
        metrics[f"rec_{name}"] = float(tp / (tp + fn + 1e-6)) if (tp + fn) > 0 else 0.0

    # Height RMSE on positive-truth pixels
    h_pred = pred[3]
    h_gt = label_arr[3]
    for cls, name in [(0, "bld"), (1, "tree")]:
        mask = (label_arr[cls] > 0).astype(np.float32)
        if mask.sum() == 0:
            metrics[f"rmse_{name}H"] = float("nan")
            continue
        diff2 = ((h_pred - h_gt) ** 2 * mask).sum() / mask.sum()
        metrics[f"rmse_{name}H"] = float(np.sqrt(diff2))
    return metrics


def overall_score(m, rmse_b_ceiling=3.0, rmse_v_ceiling=5.0):
    """Approximate leaderboard score from per-tile metrics."""
    rb = m.get("rmse_bldH", rmse_b_ceiling)
    rv = m.get("rmse_treeH", rmse_v_ceiling)
    if np.isnan(rb):
        rb = rmse_b_ceiling
    if np.isnan(rv):
        rv = rmse_v_ceiling
    return (
        WEIGHTS["iou_buildings"] * m["iou_bld"]
        + WEIGHTS["iou_trees"] * m["iou_tree"]
        + WEIGHTS["iou_water"] * m["iou_wat"]
        + WEIGHTS["RMSE_building_height"] * max(0, 1.0 - rb / rmse_b_ceiling)
        + WEIGHTS["RMSE_vegetation_height"] * max(0, 1.0 - rv / rmse_v_ceiling)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--split-file", default=None, help="Optional JSON with 'val' key (core ids).")
    p.add_argument("--thresholds", nargs=3, type=float, required=True, metavar=("B", "V", "W"))
    p.add_argument("--water-k", type=int, default=0)
    p.add_argument("--top-n-worst", type=int, default=20)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npy")))
    if not pred_files:
        sys.exit(f"No .npy in {args.pred_dir}")
    label_files = glob.glob(os.path.join(args.labels_dir, "**", "label_*.tif"), recursive=True)
    label_map = {normalize_core_id(p): p for p in label_files}

    val_ids = None
    if args.split_file:
        with open(args.split_file) as f:
            val_ids = set(json.load(f).get("val", []))

    import rasterio
    rows = []
    n_skipped = 0
    for pf in pred_files:
        core_id = normalize_core_id(pf)
        if val_ids is not None and core_id not in val_ids:
            continue
        if core_id not in label_map:
            n_skipped += 1
            continue
        pred = np.load(pf).astype(np.float32)
        with rasterio.open(label_map[core_id]) as src:
            lab = src.read().astype(np.float32)
        # Pad label to match pred if needed
        if lab.shape != pred.shape:
            n_skipped += 1
            continue
        m = compute_metrics(pred, lab, args.thresholds, water_k=args.water_k)
        m["score"] = overall_score(m)
        m["core_id"] = core_id
        rows.append(m)

    if not rows:
        sys.exit("No tiles evaluated.")
    rows_sorted = sorted(rows, key=lambda r: r["score"])

    # Summary
    print(f"Evaluated {len(rows)} tiles (skipped {n_skipped})")
    print(f"Thresholds: bld={args.thresholds[0]} veg={args.thresholds[1]} wat={args.thresholds[2]} | K_water={args.water_k}")
    print()
    scores = np.array([r["score"] for r in rows])
    print(f"Score: mean={scores.mean():.4f}  std={scores.std():.4f}  min={scores.min():.4f}  max={scores.max():.4f}")
    print(f"       p10={np.percentile(scores, 10):.4f}  median={np.percentile(scores, 50):.4f}  p90={np.percentile(scores, 90):.4f}")

    # Per-class summary
    for name in ["bld", "tree", "wat"]:
        ious = np.array([r[f"iou_{name}"] for r in rows])
        precs = np.array([r[f"prec_{name}"] for r in rows])
        recs = np.array([r[f"rec_{name}"] for r in rows])
        print(f"\n{name.upper()}: iou mean={ious.mean():.4f} | prec mean={precs.mean():.4f} | rec mean={recs.mean():.4f}")
        print(f"     iou worst-10%: {np.percentile(ious, 10):.4f}  best-10%: {np.percentile(ious, 90):.4f}")

    # Worst N
    print(f"\n=== Worst {args.top_n_worst} tiles by score ===")
    print(f"{'core_id':<22} {'score':>7} {'iou_b':>7} {'iou_t':>7} {'iou_w':>7} {'rmse_b':>7} {'rmse_v':>7}")
    for r in rows_sorted[:args.top_n_worst]:
        rb = r.get("rmse_bldH", float("nan"))
        rv = r.get("rmse_treeH", float("nan"))
        print(f"{r['core_id']:<22} {r['score']:>7.4f} {r['iou_bld']:>7.4f} {r['iou_tree']:>7.4f} {r['iou_wat']:>7.4f} {rb:>7.3f} {rv:>7.3f}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(rows, indent=2))
        print(f"\nDetailed JSON: {args.output_json}")


if __name__ == "__main__":
    main()
