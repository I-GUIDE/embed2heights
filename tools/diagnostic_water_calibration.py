"""Water-channel calibration diagnostic.

For each run directory, computes:
  - best sample-averaged water IoU and the threshold that achieves it
  - pooled-pixel water IoU / precision / recall at that threshold
  - empty-water false-positive count: val patches with no GT water that have
    any predicted water pixel at the threshold
  - background water-probability statistics (mean, p99) over non-water pixels

Mirrors the diagnostic table in logs/XFUSION_CROSSLEVEL_PLAN.md.
"""
import argparse
import glob
import os
import sys

import numpy as np
import rasterio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from core.metrics import (
    LABEL_THRESHOLD, CH_WATER, build_label_map, load_val_ids,
)
from core.dataset import normalize_core_id


def _per_image_iou(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return inter / union


def _diagnose(pred_dir, labels_dir, val_ids, thr_sweep):
    label_map = build_label_map(labels_dir)
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))

    water_probs_by_image = []
    water_gt_by_image = []
    bg_probs_chunks = []

    for pf in pred_files:
        core_id = normalize_core_id(pf)
        if core_id not in label_map or core_id not in val_ids:
            continue
        pred = np.load(pf)
        with rasterio.open(label_map[core_id]) as src:
            label = src.read().astype(np.float32)
        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        prob = pred[CH_WATER, :h, :w].astype(np.float32)
        gt = (label[CH_WATER, :h, :w] > LABEL_THRESHOLD)

        water_probs_by_image.append(prob)
        water_gt_by_image.append(gt)
        bg_probs_chunks.append(prob[~gt])

    if not water_probs_by_image:
        raise SystemExit(f"no matched val predictions in {pred_dir}")

    bg_all = np.concatenate(bg_probs_chunks).astype(np.float64)
    bg_mean = float(bg_all.mean())
    bg_p99 = float(np.percentile(bg_all, 99))
    bg_p999 = float(np.percentile(bg_all, 99.9))

    n_total = len(water_probs_by_image)
    n_empty_water = int(sum(int(not g.any()) for g in water_gt_by_image))

    rows = []
    for thr in thr_sweep:
        per_image_iou = []
        empty_fp = 0
        tp = fp = fn = 0
        for prob, gt in zip(water_probs_by_image, water_gt_by_image):
            pb = prob > thr
            per_image_iou.append(_per_image_iou(pb, gt))
            if not gt.any() and pb.any():
                empty_fp += 1
            tp += int(np.logical_and(pb, gt).sum())
            fp += int(np.logical_and(pb, ~gt).sum())
            fn += int(np.logical_and(~pb, gt).sum())
        sample_iou = float(np.mean(per_image_iou))
        pooled_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        rows.append({
            "thr": thr,
            "sample_iou": sample_iou,
            "pooled_iou": pooled_iou,
            "precision": precision,
            "recall": recall,
            "empty_fp": empty_fp,
        })

    best = max(rows, key=lambda r: r["sample_iou"])
    return {
        "n_total": n_total,
        "n_empty_water": n_empty_water,
        "bg_mean": bg_mean,
        "bg_p99": bg_p99,
        "bg_p999": bg_p999,
        "rows": rows,
        "best": best,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", action="append", required=True,
                   help="One or more run directories (predictions/ subdir is read).")
    p.add_argument("--labels-dir",
                   default=os.path.abspath(os.path.join(ROOT_DIR, "..", "data", "train", "labels")))
    p.add_argument("--split-file",
                   default=os.path.join(ROOT_DIR, "splits", "split.json"))
    p.add_argument("--thr-grid", default="0.50,0.60,0.70,0.75,0.80,0.82,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96",
                   help="Comma-separated water thresholds to sweep.")
    return p.parse_args()


def main():
    args = parse_args()
    val_ids = load_val_ids(args.split_file)
    thr_sweep = [float(x) for x in args.thr_grid.split(",")]

    print()
    print(f"  Val ids: {len(val_ids)}    Labels dir: {args.labels_dir}")
    print(f"  Thresholds swept: {thr_sweep[0]:.2f} .. {thr_sweep[-1]:.2f} ({len(thr_sweep)} pts)")
    print()
    print(f"{'Run':<60} {'thr':>5} {'sIoU':>7} {'pIoU':>7} {'prec':>7} {'rec':>7} {'eFP':>8} {'bgmean':>9} {'bgp99':>9}")
    print("-" * 140)

    summaries = []
    for rd in args.run_dir:
        pred_dir = os.path.join(rd, "predictions")
        if not os.path.isdir(pred_dir):
            print(f"  [skip] no predictions/ under {rd}")
            continue
        d = _diagnose(pred_dir, args.labels_dir, val_ids, thr_sweep)
        b = d["best"]
        run_name = os.path.basename(rd.rstrip("/"))
        print(f"{run_name:<60} {b['thr']:>5.2f} {b['sample_iou']:>7.4f} "
              f"{b['pooled_iou']:>7.4f} {b['precision']:>7.4f} {b['recall']:>7.4f} "
              f"{str(b['empty_fp'])+'/'+str(d['n_empty_water']):>8} "
              f"{d['bg_mean']:>9.5f} {d['bg_p99']:>9.4f}")
        summaries.append((run_name, d))

    # Per-run threshold sweep table for the most relevant region.
    print()
    for run_name, d in summaries:
        print()
        print(f"== {run_name} threshold sweep ==")
        print(f"{'thr':>5} {'sIoU':>7} {'pIoU':>7} {'prec':>7} {'rec':>7} {'eFP':>8}")
        for r in d["rows"]:
            print(f"{r['thr']:>5.2f} {r['sample_iou']:>7.4f} {r['pooled_iou']:>7.4f} "
                  f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
                  f"{str(r['empty_fp'])+'/'+str(d['n_empty_water']):>8}")
        print(f"   bg_mean={d['bg_mean']:.5f}  bg_p99={d['bg_p99']:.4f}  bg_p999={d['bg_p999']:.4f}")


if __name__ == "__main__":
    main()
