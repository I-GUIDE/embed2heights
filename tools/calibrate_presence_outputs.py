"""
Calibrate continuous presence outputs so a server-side threshold of 0.5
matches the best validation threshold found locally.

This tool does NOT binarize channels 0-2. Instead it applies a per-class
logit shift:

    p' = sigmoid(scale * (logit(p) - logit(t)))

so that `p > t` is equivalent to `p' > 0.5`.

Typical use:

    # 1. Tune thresholds on val predictions with leaderboard-aligned metric
    python tools/calibrate_presence_outputs.py \
        --val-pred-dir runs/my_exp/predictions \
        --output-json runs/my_exp/presence_calibration.json

    # 2. Apply the calibration to raw test predictions
    python tools/calibrate_presence_outputs.py \
        --thresholds 0.72 0.54 0.92 \
        --apply-pred-dir runs/my_exp/test_predictions_alphaearth \
        --apply-output-dir runs/my_exp/test_predictions_alphaearth_calibrated
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
    binary_iou,
    build_label_map,
    compute_weighted_score,
    load_val_ids,
)

DEFAULT_LABELS_DIR = SCRIPT_DIR.parent / "data" / "train" / "labels"
DEFAULT_SPLIT_FILE = SCRIPT_DIR / "splits" / "split.json"
CLASS_NAMES = ("building", "vegetation", "water")


def load_inputs(pred_dir, labels_dir, split_file):
    val_ids = load_val_ids(str(split_file))
    label_map = build_label_map(str(labels_dir))
    pred_files = sorted(glob.glob(str(pred_dir / "*.npy")))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")

    predictions, labels = {}, {}
    for pf in pred_files:
        core_id = normalize_core_id(pf)
        if core_id not in val_ids or core_id not in label_map:
            continue
        predictions[core_id] = np.load(pf).astype(np.float32)
        with rasterio.open(label_map[core_id]) as src:
            labels[core_id] = src.read().astype(np.float32)

    if not predictions:
        raise RuntimeError("No matched val pred/label pairs after filtering.")
    return predictions, labels


def evaluate_thresholds(labels, predictions, pred_thresholds):
    iou_lists = [[], [], []]
    rmse_bld, rmse_veg = [], []

    for core_id, pred in predictions.items():
        label = labels[core_id]
        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        pred = pred[:, :h, :w]
        label = label[:, :h, :w]

        for c in range(3):
            iou_lists[c].append(binary_iou(
                pred[c] > pred_thresholds[c],
                label[c] > LABEL_THRESHOLD,
            ))

        for ch, bucket in ((0, rmse_bld), (1, rmse_veg)):
            mask = label[ch] > LABEL_THRESHOLD
            if mask.any():
                diff = pred[3][mask] - label[3][mask]
                bucket.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))

    def safe_mean(values):
        vals = [v for v in values if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    metrics = {
        "iou_buildings": safe_mean(iou_lists[0]),
        "iou_trees": safe_mean(iou_lists[1]),
        "iou_water": safe_mean(iou_lists[2]),
        "RMSE_building_height": safe_mean(rmse_bld),
        "RMSE_vegetation_height": safe_mean(rmse_veg),
        "n_samples": len(predictions),
    }
    score, parts = compute_weighted_score(metrics)
    metrics["weighted_score"] = score
    metrics["score_parts"] = parts
    return metrics


def find_best_thresholds(labels, predictions, grid_start, grid_stop, grid_step):
    grid = np.arange(grid_start, grid_stop + 1e-9, grid_step, dtype=np.float32)

    per_class = [0.5, 0.5, 0.5]
    for c, key in enumerate(("iou_buildings", "iou_trees", "iou_water")):
        best_t, best_val = 0.5, -1.0
        for t in grid:
            trial = per_class.copy()
            trial[c] = float(t)
            metrics = evaluate_thresholds(labels, predictions, tuple(trial))
            if metrics[key] > best_val:
                best_t, best_val = float(t), float(metrics[key])
        per_class[c] = best_t

    base_metrics = evaluate_thresholds(labels, predictions, (0.5, 0.5, 0.5))
    best_metrics = evaluate_thresholds(labels, predictions, tuple(per_class))
    return tuple(per_class), base_metrics, best_metrics


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def calibrate_array(arr, thresholds, scales, eps):
    out = arr.astype(np.float32, copy=True)
    for c in range(3):
        z = logit(out[c], eps)
        z = scales[c] * (z - logit(np.float32(thresholds[c]), eps))
        out[c] = sigmoid(z).astype(np.float32)
    return out


def apply_calibration(pred_dir, output_dir, thresholds, scales, eps):
    pred_files = sorted(pred_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for pred_path in pred_files:
        arr = np.load(pred_path).astype(np.float32, copy=False)
        calibrated = calibrate_array(arr, thresholds, scales, eps)
        np.save(output_dir / pred_path.name, calibrated)


def parse_scale_args(scale, scales):
    if scales is not None:
        return tuple(float(v) for v in scales)
    return (float(scale), float(scale), float(scale))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-pred-dir", type=Path, default=None,
                        help="Validation prediction dir used to tune thresholds and report calibrated val score.")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--grid-start", type=float, default=0.05)
    parser.add_argument("--grid-stop", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--thresholds", type=float, nargs=3, default=None,
                        metavar=("BLD", "VEG", "WAT"),
                        help="Use explicit thresholds instead of tuning them from --val-pred-dir.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Global logit-scale after threshold centering. "
                             "scale=1 preserves exact binary equivalence at 0.5.")
    parser.add_argument("--scales", type=float, nargs=3, default=None,
                        metavar=("BLD", "VEG", "WAT"),
                        help="Per-class logit scales. Overrides --scale.")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Numerical clamp before logit.")
    parser.add_argument("--apply-pred-dir", type=Path, default=None,
                        help="Prediction dir to transform with the calibrated continuous mapping.")
    parser.add_argument("--apply-output-dir", type=Path, default=None,
                        help="Output dir for transformed predictions.")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional JSON report path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.apply_pred_dir is not None and args.apply_output_dir is None:
        raise ValueError("--apply-output-dir is required when --apply-pred-dir is set")
    if args.thresholds is None and args.val_pred_dir is None:
        raise ValueError("Provide either --thresholds or --val-pred-dir")

    scales = parse_scale_args(args.scale, args.scales)
    report = {
        "scale": scales,
        "eps": args.eps,
    }

    if args.val_pred_dir is not None:
        predictions, labels = load_inputs(args.val_pred_dir, args.labels_dir, args.split_file)
        if args.thresholds is None:
            thresholds, base_metrics, best_metrics = find_best_thresholds(
                labels,
                predictions,
                args.grid_start,
                args.grid_stop,
                args.grid_step,
            )
        else:
            thresholds = tuple(float(v) for v in args.thresholds)
            base_metrics = evaluate_thresholds(labels, predictions, (0.5, 0.5, 0.5))
            best_metrics = evaluate_thresholds(labels, predictions, thresholds)

        calibrated_predictions = {
            core_id: calibrate_array(pred, thresholds, scales, args.eps)
            for core_id, pred in predictions.items()
        }
        calibrated_metrics = evaluate_thresholds(labels, calibrated_predictions, (0.5, 0.5, 0.5))

        report.update({
            "val_pred_dir": str(args.val_pred_dir),
            "labels_dir": str(args.labels_dir),
            "split_file": str(args.split_file),
            "n_samples": len(predictions),
            "selected_thresholds": dict(zip(CLASS_NAMES, thresholds)),
            "base_metrics_at_0.5": base_metrics,
            "best_metrics_at_selected_thresholds": best_metrics,
            "calibrated_metrics_at_0.5": calibrated_metrics,
        })

        print(f"Matched {len(predictions)} validation samples")
        print(f"Selected thresholds: bld={thresholds[0]:.3f} veg={thresholds[1]:.3f} wat={thresholds[2]:.3f}")
        print(f"Logit scales:        bld={scales[0]:.3f} veg={scales[1]:.3f} wat={scales[2]:.3f}")
        print(
            "Base @0.5:          "
            f"{base_metrics['iou_buildings']:.4f} {base_metrics['iou_trees']:.4f} {base_metrics['iou_water']:.4f} "
            f"{base_metrics['RMSE_building_height']:.4f} {base_metrics['RMSE_vegetation_height']:.4f} "
            f"{base_metrics['weighted_score']:.4f}"
        )
        print(
            "Best @selected:      "
            f"{best_metrics['iou_buildings']:.4f} {best_metrics['iou_trees']:.4f} {best_metrics['iou_water']:.4f} "
            f"{best_metrics['RMSE_building_height']:.4f} {best_metrics['RMSE_vegetation_height']:.4f} "
            f"{best_metrics['weighted_score']:.4f}"
        )
        print(
            "Calibrated @0.5:     "
            f"{calibrated_metrics['iou_buildings']:.4f} {calibrated_metrics['iou_trees']:.4f} {calibrated_metrics['iou_water']:.4f} "
            f"{calibrated_metrics['RMSE_building_height']:.4f} {calibrated_metrics['RMSE_vegetation_height']:.4f} "
            f"{calibrated_metrics['weighted_score']:.4f}"
        )
    else:
        thresholds = tuple(float(v) for v in args.thresholds)
        report["selected_thresholds"] = dict(zip(CLASS_NAMES, thresholds))

    if args.apply_pred_dir is not None:
        apply_calibration(args.apply_pred_dir, args.apply_output_dir, thresholds, scales, args.eps)
        report["apply_pred_dir"] = str(args.apply_pred_dir)
        report["apply_output_dir"] = str(args.apply_output_dir)
        print(f"Wrote calibrated predictions: {args.apply_output_dir}")

    output_json = args.output_json
    if output_json is None:
        base_dir = args.val_pred_dir.parent if args.val_pred_dir is not None else args.apply_output_dir.parent
        output_json = base_dir / "presence_calibration.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote report: {output_json}")


if __name__ == "__main__":
    main()
