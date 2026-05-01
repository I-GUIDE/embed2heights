"""
Sweep simple post-hoc height calibration on validation predictions.

The calibration is applied only to pixels that look building-like according to
the model:

    if pred_building > mask_threshold and pred_height > min_pred_height:
        pred_height = pred_height * scale + bias

This is an in-memory diagnostic by default; prediction files are not modified.
It reports leaderboard-aligned IoUs/RMSEs and total score, so the tradeoff
between building RMSE improvement and vegetation RMSE damage is visible.

Typical fold-0 probe:

    python tools/sweep_height_calibration.py \
        --pred-dir runs/xfusion_019_tm_s2_ae_tessera_presence_3way_deep_groupcode_f0/predictions \
        --split-file splits/group_code_5fold_seed42/fold_0/split.json \
        --thresholds 0.67 0.60 0.87
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.dataset import normalize_core_id  # noqa: E402
from core.metrics import (  # noqa: E402
    LABEL_THRESHOLD,
    CH_BUILDING,
    CH_VEGETATION,
    CH_WATER,
    CH_HEIGHT,
    binary_iou,
    build_label_map,
    compute_weighted_score,
    load_val_ids,
)

DEFAULT_LABELS_DIR = SCRIPT_DIR.parent / "data" / "train" / "labels"
DEFAULT_SPLIT_FILE = SCRIPT_DIR / "splits" / "split.json"


def frange(start, stop, step):
    values = []
    x = float(start)
    while x <= float(stop) + 1e-9:
        values.append(round(x, 10))
        x += float(step)
    return values


def safe_mean(values):
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def load_pairs(pred_dir, labels_dir, split_file):
    val_ids = load_val_ids(str(split_file))
    label_map = build_label_map(str(labels_dir))
    pred_files = sorted(glob.glob(str(pred_dir / "*.npy")))
    if not pred_files:
        raise FileNotFoundError("No .npy files found in {}".format(pred_dir))

    pairs = []
    for pred_path in pred_files:
        core_id = normalize_core_id(pred_path)
        if core_id not in val_ids or core_id not in label_map:
            continue
        pairs.append((core_id, pred_path, label_map[core_id]))
    if not pairs:
        raise RuntimeError("No matched pred/label pairs after split filtering.")
    return pairs


def collect_stats(pairs, pred_thresholds, mask_thresholds, min_pred_heights):
    """Collect IoUs and RMSE sufficient statistics for every mask setting.

    For each image/class/mask setting, store terms that let us evaluate
    sum((d0 + I * ((scale - 1) * h + bias))^2) without revisiting pixels.
    """
    iou_lists = [[], [], []]
    base_rmse = {"building": [], "vegetation": []}
    stats = {}
    for mask_thr in mask_thresholds:
        for min_h in min_pred_heights:
            stats[(mask_thr, min_h)] = {
                "building": [],
                "vegetation": [],
            }

    for _, pred_path, label_path in pairs:
        pred = np.load(pred_path).astype(np.float32)
        with rasterio.open(label_path) as src:
            label = src.read().astype(np.float32)

        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        pred = pred[:, :h, :w]
        label = label[:, :h, :w]

        iou_lists[0].append(binary_iou(
            pred[CH_BUILDING] > pred_thresholds[0],
            label[CH_BUILDING] > LABEL_THRESHOLD,
        ))
        iou_lists[1].append(binary_iou(
            pred[CH_VEGETATION] > pred_thresholds[1],
            label[CH_VEGETATION] > LABEL_THRESHOLD,
        ))
        iou_lists[2].append(binary_iou(
            pred[CH_WATER] > pred_thresholds[2],
            label[CH_WATER] > LABEL_THRESHOLD,
        ))

        pred_height = pred[CH_HEIGHT].astype(np.float64)
        build_prob = pred[CH_BUILDING]
        label_height = label[CH_HEIGHT].astype(np.float64)

        for class_name, channel in (("building", CH_BUILDING), ("vegetation", CH_VEGETATION)):
            gt_mask = label[channel] > LABEL_THRESHOLD
            if not gt_mask.any():
                continue

            h_gt = pred_height[gt_mask]
            d0 = h_gt - label_height[gt_mask]
            base_ss = float(np.sum(d0 * d0))
            n = int(d0.size)
            base_rmse[class_name].append(float(np.sqrt(base_ss / n)))

            prob_gt = build_prob[gt_mask]
            for mask_thr in mask_thresholds:
                for min_h in min_pred_heights:
                    apply = (prob_gt > mask_thr) & (h_gt > min_h)
                    if apply.any():
                        h_apply = h_gt[apply]
                        d_apply = d0[apply]
                        record = {
                            "n": n,
                            "base_ss": base_ss,
                            "a_count": int(apply.sum()),
                            "sum_dh": float(np.sum(d_apply * h_apply)),
                            "sum_d": float(np.sum(d_apply)),
                            "sum_h2": float(np.sum(h_apply * h_apply)),
                            "sum_h": float(np.sum(h_apply)),
                        }
                    else:
                        record = {
                            "n": n,
                            "base_ss": base_ss,
                            "a_count": 0,
                            "sum_dh": 0.0,
                            "sum_d": 0.0,
                            "sum_h2": 0.0,
                            "sum_h": 0.0,
                        }
                    stats[(mask_thr, min_h)][class_name].append(record)

    base_metrics = {
        "iou_buildings": safe_mean(iou_lists[0]),
        "iou_trees": safe_mean(iou_lists[1]),
        "iou_water": safe_mean(iou_lists[2]),
        "RMSE_building_height": safe_mean(base_rmse["building"]),
        "RMSE_vegetation_height": safe_mean(base_rmse["vegetation"]),
        "n_samples": len(pairs),
    }
    base_score, base_parts = compute_weighted_score(base_metrics)
    base_metrics["weighted_score"] = base_score
    base_metrics["score_parts"] = base_parts
    return base_metrics, stats


def rmse_from_records(records, scale, bias):
    alpha = float(scale) - 1.0
    beta = float(bias)
    rmses = []
    apply_fracs = []
    for rec in records:
        ss = rec["base_ss"]
        if rec["a_count"]:
            ss += (
                2.0 * alpha * rec["sum_dh"]
                + 2.0 * beta * rec["sum_d"]
                + alpha * alpha * rec["sum_h2"]
                + 2.0 * alpha * beta * rec["sum_h"]
                + beta * beta * rec["a_count"]
            )
        ss = max(ss, 0.0)
        rmses.append(float(np.sqrt(ss / rec["n"])))
        apply_fracs.append(rec["a_count"] / max(1, rec["n"]))
    return safe_mean(rmses), safe_mean(apply_fracs)


def evaluate_calibration(base_metrics, records, mask_thr, min_h, scale, bias):
    rmse_b, b_apply = rmse_from_records(records["building"], scale, bias)
    rmse_v, v_apply = rmse_from_records(records["vegetation"], scale, bias)
    metrics = dict(base_metrics)
    metrics["RMSE_building_height"] = rmse_b
    metrics["RMSE_vegetation_height"] = rmse_v
    score, parts = compute_weighted_score(metrics)
    metrics["weighted_score"] = score
    metrics["score_parts"] = parts
    return {
        "mask_threshold": float(mask_thr),
        "min_pred_height": float(min_h),
        "scale": float(scale),
        "bias": float(bias),
        "building_apply_frac_on_gt": float(b_apply),
        "vegetation_apply_frac_on_gt": float(v_apply),
        "metrics": metrics,
        "score_delta": float(score - base_metrics["weighted_score"]),
        "rmse_b_delta": float(rmse_b - base_metrics["RMSE_building_height"]),
        "rmse_v_delta": float(rmse_v - base_metrics["RMSE_vegetation_height"]),
    }


def fmt_row(row):
    m = row["metrics"]
    return (
        "{score:7.4f} {delta:+8.4f}  {mask:5.2f} {minh:5.2f} "
        "{scale:5.2f} {bias:5.2f}  {rb:7.4f} {drb:+7.4f} "
        "{rv:7.4f} {drv:+7.4f}  {ba:6.3f} {va:6.3f}"
    ).format(
        score=m["weighted_score"],
        delta=row["score_delta"],
        mask=row["mask_threshold"],
        minh=row["min_pred_height"],
        scale=row["scale"],
        bias=row["bias"],
        rb=m["RMSE_building_height"],
        drb=row["rmse_b_delta"],
        rv=m["RMSE_vegetation_height"],
        drv=row["rmse_v_delta"],
        ba=row["building_apply_frac_on_gt"],
        va=row["vegetation_apply_frac_on_gt"],
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    p.add_argument("--thresholds", type=float, nargs=3, default=(0.5, 0.5, 0.5),
                   metavar=("BLD", "VEG", "WAT"),
                   help="Presence thresholds used for IoU score reporting.")
    p.add_argument("--mask-thresholds", type=float, nargs="*", default=None,
                   help="Explicit building-prob thresholds for deciding where to calibrate height.")
    p.add_argument("--mask-threshold-start", type=float, default=0.20)
    p.add_argument("--mask-threshold-stop", type=float, default=0.80)
    p.add_argument("--mask-threshold-step", type=float, default=0.10)
    p.add_argument("--min-pred-heights", type=float, nargs="*", default=(0.0, 2.0, 5.0))
    p.add_argument("--scale-start", type=float, default=1.00)
    p.add_argument("--scale-stop", type=float, default=1.80)
    p.add_argument("--scale-step", type=float, default=0.05)
    p.add_argument("--bias-start", type=float, default=0.00)
    p.add_argument("--bias-stop", type=float, default=2.00)
    p.add_argument("--bias-step", type=float, default=0.25)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-rmse-v-delta", type=float, default=None,
                   help="Optional filter: only rank rows where vegetation RMSE delta <= this value.")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    mask_thresholds = (
        [float(v) for v in args.mask_thresholds]
        if args.mask_thresholds is not None
        else frange(args.mask_threshold_start, args.mask_threshold_stop, args.mask_threshold_step)
    )
    min_pred_heights = [float(v) for v in args.min_pred_heights]
    scales = frange(args.scale_start, args.scale_stop, args.scale_step)
    biases = frange(args.bias_start, args.bias_stop, args.bias_step)

    pairs = load_pairs(args.pred_dir, args.labels_dir, args.split_file)
    print("Matched {} pred/label pairs".format(len(pairs)))
    print("Collecting sufficient statistics ...")
    base_metrics, stats = collect_stats(
        pairs,
        tuple(float(v) for v in args.thresholds),
        mask_thresholds,
        min_pred_heights,
    )

    rows = []
    for mask_thr in mask_thresholds:
        for min_h in min_pred_heights:
            records = stats[(mask_thr, min_h)]
            for scale in scales:
                for bias in biases:
                    row = evaluate_calibration(base_metrics, records, mask_thr, min_h, scale, bias)
                    if args.max_rmse_v_delta is not None and row["rmse_v_delta"] > args.max_rmse_v_delta:
                        continue
                    rows.append(row)

    rows.sort(key=lambda row: row["metrics"]["weighted_score"], reverse=True)

    print("\nBase metrics:")
    print(
        "  iou_bld={:.4f} iou_tree={:.4f} iou_wat={:.4f} "
        "RMSE_bH={:.4f} RMSE_vH={:.4f} score={:.4f}".format(
            base_metrics["iou_buildings"],
            base_metrics["iou_trees"],
            base_metrics["iou_water"],
            base_metrics["RMSE_building_height"],
            base_metrics["RMSE_vegetation_height"],
            base_metrics["weighted_score"],
        )
    )

    print("\nTop calibrations:")
    print("  score    delta   mask  minH scale  bias  RMSE_b  dRMSEb  RMSE_v  dRMSEv  b_app  v_app")
    print("  " + "-" * 94)
    for row in rows[:args.top_k]:
        print("  " + fmt_row(row))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "pred_dir": str(args.pred_dir),
            "labels_dir": str(args.labels_dir),
            "split_file": str(args.split_file),
            "thresholds": list(map(float, args.thresholds)),
            "base_metrics": base_metrics,
            "grid": {
                "mask_thresholds": mask_thresholds,
                "min_pred_heights": min_pred_heights,
                "scales": scales,
                "biases": biases,
                "max_rmse_v_delta": args.max_rmse_v_delta,
            },
            "top_rows": rows[: max(args.top_k, 100)],
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print("\nWrote {}".format(args.output_json))


if __name__ == "__main__":
    main()
