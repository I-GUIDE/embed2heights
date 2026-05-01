"""Fit and estimate a lightweight post-processor for xfusion_019 OOF folds.

The script treats each group-code fold model as the only valid predictor for
its held-out validation ids. It then runs a second-level cross-validation over
those five OOF folds:

  - fit class thresholds on four folds,
  - fit a water connected-component filter on four folds,
  - optionally fit a simple class-conditioned height affine correction,
  - evaluate the learned parameters on the held-out fold.

This gives a less biased estimate than fitting post-processing on all OOF
predictions and reporting the same samples.
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rasterio
try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - fallback for minimal environments.
    ndimage = None

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.dataset import normalize_core_id  # noqa: E402
from core.metrics import (  # noqa: E402
    LABEL_THRESHOLD,
    binary_iou,
    build_label_map,
    compute_weighted_score,
)

DEFAULT_LABELS_DIR = SCRIPT_DIR.parent / "data" / "train" / "labels"
DEFAULT_FOLD_PATTERN = (
    SCRIPT_DIR
    / "runs"
    / "xfusion_019_tm_s2_ae_tessera_presence_3way_deep_groupcode_f{fold}"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs" / "xfusion_019_oof_postprocess"

CLASS_KEYS = ("iou_buildings", "iou_trees", "iou_water")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prediction_map(pred_dir):
    paths = sorted(pred_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy predictions found in {pred_dir}")
    return {normalize_core_id(str(path)): path for path in paths}


def collect_records(fold_pattern, labels_dir):
    label_map = build_label_map(str(labels_dir))
    records = []
    fold_dirs = []
    seen = set()

    for fold in range(5):
        fold_dir = Path(str(fold_pattern).format(fold=fold))
        split_file = fold_dir / "split.json"
        pred_dir = fold_dir / "predictions"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        pred_map = _prediction_map(pred_dir)
        split = _load_json(split_file)
        val_ids = sorted(split["val"])
        fold_dirs.append(str(fold_dir))

        for core_id in val_ids:
            if core_id in seen:
                raise RuntimeError(f"Duplicate held-out id across folds: {core_id}")
            if core_id not in pred_map:
                raise FileNotFoundError(f"Missing OOF prediction for {core_id} in {pred_dir}")
            if core_id not in label_map:
                raise FileNotFoundError(f"Missing label for {core_id} under {labels_dir}")
            seen.add(core_id)
            records.append({
                "core_id": core_id,
                "fold": fold,
                "pred_path": str(pred_map[core_id]),
                "label_path": label_map[core_id],
            })

    return records, fold_dirs


def _read_pair(record):
    pred = np.load(record["pred_path"]).astype(np.float32)
    with rasterio.open(record["label_path"]) as src:
        label = src.read().astype(np.float32)
    h = min(pred.shape[1], label.shape[1])
    w = min(pred.shape[2], label.shape[2])
    return pred[:, :h, :w], label[:, :h, :w]


def largest_component_size(mask):
    """Return largest 8-connected component size for a boolean 2D mask."""
    if not mask.any():
        return 0
    if ndimage is not None:
        structure = np.ones((3, 3), dtype=np.int8)
        labels, n_labels = ndimage.label(mask, structure=structure)
        if n_labels == 0:
            return 0
        counts = np.bincount(labels.ravel())
        return int(counts[1:].max()) if len(counts) > 1 else 0
    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    best = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        visited[sy, sx] = True
        q = deque([(int(sy), int(sx))])
        size = 0
        while q:
            y, x = q.pop()
            size += 1
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            for ny in range(y0, y1):
                for nx in range(x0, x1):
                    if not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
        if size > best:
            best = size
    return best


def fit_height_affine(records, shrink):
    sums = {
        "building": {"n": 0, "x": 0.0, "y": 0.0, "xx": 0.0, "xy": 0.0},
        "vegetation": {"n": 0, "x": 0.0, "y": 0.0, "xx": 0.0, "xy": 0.0},
    }
    for record in records:
        pred, label = _read_pair(record)
        for name, ch in (("building", 0), ("vegetation", 1)):
            mask = label[ch] > LABEL_THRESHOLD
            if not mask.any():
                continue
            x = pred[3][mask].astype(np.float64)
            y = label[3][mask].astype(np.float64)
            bucket = sums[name]
            bucket["n"] += int(mask.sum())
            bucket["x"] += float(x.sum())
            bucket["y"] += float(y.sum())
            bucket["xx"] += float((x * x).sum())
            bucket["xy"] += float((x * y).sum())

    params = {}
    for name, bucket in sums.items():
        n = bucket["n"]
        if n == 0:
            a_fit, b_fit = 1.0, 0.0
        else:
            denom = bucket["xx"] - (bucket["x"] * bucket["x"] / n)
            if abs(denom) < 1e-9:
                a_fit = 1.0
                b_fit = bucket["y"] / n - bucket["x"] / n
            else:
                a_fit = (bucket["xy"] - bucket["x"] * bucket["y"] / n) / denom
                b_fit = bucket["y"] / n - a_fit * bucket["x"] / n
        params[name] = {
            "a_fit": float(a_fit),
            "b_fit": float(b_fit),
            "a": float(1.0 + shrink * (a_fit - 1.0)),
            "b": float(shrink * b_fit),
            "n_pixels": int(n),
        }
    return params


def apply_height(pred, params):
    if not params.get("height_affine", True):
        return pred[3]
    b = params["height_affine_params"]["building"]
    v = params["height_affine_params"]["vegetation"]
    h_b = np.maximum(0.0, b["a"] * pred[3] + b["b"])
    h_v = np.maximum(0.0, v["a"] * pred[3] + v["b"])
    p_b = np.clip(pred[0], 0.0, 1.0)
    p_v = np.clip(pred[1], 0.0, 1.0)
    fg = 1.0 - (1.0 - p_b) * (1.0 - p_v)
    denom = p_b + p_v + 1e-6
    h_fg = (p_b * h_b + p_v * h_v) / denom
    return fg * h_fg + (1.0 - fg) * pred[3]


def eval_records(records, params):
    ious = [[], [], []]
    rmse_b, rmse_v = [], []
    empty_water_fp = 0
    empty_water_total = 0

    thresholds = params["thresholds"]
    water_k = int(params.get("water_min_component", 0))

    for record in records:
        pred, label = _read_pair(record)
        masks = [
            pred[0] > thresholds[0],
            pred[1] > thresholds[1],
            pred[2] > thresholds[2],
        ]
        if water_k > 0 and largest_component_size(masks[2]) < water_k:
            masks[2] = np.zeros_like(masks[2], dtype=bool)

        for ch in range(3):
            gt = label[ch] > LABEL_THRESHOLD
            ious[ch].append(binary_iou(masks[ch], gt))

        gt_water = label[2] > LABEL_THRESHOLD
        if not gt_water.any():
            empty_water_total += 1
            if masks[2].any():
                empty_water_fp += 1

        height = apply_height(pred, params)
        for ch, bucket in ((0, rmse_b), (1, rmse_v)):
            gt = label[ch] > LABEL_THRESHOLD
            if gt.any():
                diff = height[gt].astype(np.float64) - label[3][gt].astype(np.float64)
                bucket.append(float(np.sqrt(np.mean(diff * diff))))

    def mean(xs):
        vals = [x for x in xs if not np.isnan(x)]
        return float(np.mean(vals)) if vals else float("nan")

    metrics = {
        "iou_buildings": mean(ious[0]),
        "iou_trees": mean(ious[1]),
        "iou_water": mean(ious[2]),
        "RMSE_building_height": mean(rmse_b),
        "RMSE_vegetation_height": mean(rmse_v),
        "n_samples": len(records),
        "empty_water_fp": empty_water_fp,
        "empty_water_total": empty_water_total,
    }
    score, parts = compute_weighted_score(metrics)
    metrics["weighted_score"] = score
    metrics["score_parts"] = parts
    return metrics


def tune_class_threshold(records, channel, grid):
    sums = np.zeros(len(grid), dtype=np.float64)
    for record in records:
        pred, label = _read_pair(record)
        gt = label[channel] > LABEL_THRESHOLD
        prob = pred[channel]
        for i, threshold in enumerate(grid):
            sums[i] += binary_iou(prob > threshold, gt)
    means = sums / max(1, len(records))
    idx = int(np.argmax(means))
    return float(grid[idx]), float(means[idx])


def tune_water(records, water_grid, k_grid, base_thresholds):
    threshold_sums = np.zeros(len(water_grid), dtype=np.float64)
    for record in records:
        pred, label = _read_pair(record)
        gt = label[2] > LABEL_THRESHOLD
        prob = pred[2]
        for i, threshold in enumerate(water_grid):
            threshold_sums[i] += binary_iou(prob > threshold, gt)
    threshold_means = threshold_sums / max(1, len(records))

    # Connected-component filtering is comparatively expensive. Keep the best
    # threshold-only candidates plus 0.5 as a calibration anchor.
    candidate_idxs = set(np.argsort(threshold_means)[-5:].tolist())
    nearest_05 = int(np.argmin(np.abs(water_grid - 0.5)))
    candidate_idxs.add(nearest_05)
    candidate_thresholds = [float(water_grid[i]) for i in sorted(candidate_idxs)]

    best = {"threshold": float(water_grid[int(np.argmax(threshold_means))]), "k": 0,
            "iou": float(np.max(threshold_means)), "metrics": None}
    combo_sums = {
        (threshold, int(k)): 0.0
        for threshold in candidate_thresholds
        for k in k_grid
    }
    for record in records:
        pred, label = _read_pair(record)
        gt = label[2] > LABEL_THRESHOLD
        prob = pred[2]
        gt_empty = not gt.any()
        for threshold in candidate_thresholds:
            mask = prob > threshold
            comp = largest_component_size(mask)
            for k in k_grid:
                if comp < int(k):
                    # The filter clears the whole water mask for this patch.
                    combo_sums[(threshold, int(k))] += 1.0 if gt_empty else 0.0
                else:
                    combo_sums[(threshold, int(k))] += binary_iou(mask, gt)

    for threshold in candidate_thresholds:
        for k in k_grid:
            value = combo_sums[(threshold, int(k))] / max(1, len(records))
            if value > best["iou"]:
                best = {
                    "threshold": float(threshold),
                    "k": int(k),
                    "iou": float(value),
                    "metrics": None,
                }
    return best


def fit_params(records, args):
    threshold_grid = np.arange(args.threshold_start, args.threshold_stop + 1e-9, args.threshold_step)
    water_grid = np.arange(args.water_threshold_start, args.water_threshold_stop + 1e-9, args.water_threshold_step)
    k_grid = [int(x) for x in args.water_k_grid.split(",") if x.strip()]

    b_t, b_iou = tune_class_threshold(records, 0, threshold_grid)
    v_t, v_iou = tune_class_threshold(records, 1, threshold_grid)
    water = tune_water(records, water_grid, k_grid, (b_t, v_t))

    params = {
        "thresholds": [b_t, v_t, water["threshold"]],
        "water_min_component": water["k"],
        "height_affine": not args.no_height_affine,
        "fit_details": {
            "building_iou": b_iou,
            "vegetation_iou": v_iou,
            "water_iou": water["iou"],
        },
    }
    if params["height_affine"]:
        params["height_affine_params"] = fit_height_affine(records, shrink=args.height_shrink)
    return params


def fold_records(records, fold):
    return [r for r in records if r["fold"] == fold]


def not_fold_records(records, fold):
    return [r for r in records if r["fold"] != fold]


def aggregate_metrics(rows):
    weighted = sum(row["n_samples"] * row["weighted_score"] for row in rows)
    n = sum(row["n_samples"] for row in rows)
    out = {"n_samples": n, "weighted_score": weighted / n}
    for key in (
        "iou_buildings",
        "iou_trees",
        "iou_water",
        "RMSE_building_height",
        "RMSE_vegetation_height",
    ):
        out[key] = sum(row["n_samples"] * row[key] for row in rows) / n
    out["empty_water_fp"] = sum(row["empty_water_fp"] for row in rows)
    out["empty_water_total"] = sum(row["empty_water_total"] for row in rows)
    return out


def print_metrics(label, metrics):
    print(
        f"{label:<24} "
        f"{metrics['iou_buildings']:.4f} {metrics['iou_trees']:.4f} {metrics['iou_water']:.4f} "
        f"{metrics['RMSE_building_height']:.4f} {metrics['RMSE_vegetation_height']:.4f} "
        f"{metrics['weighted_score']:.4f} "
        f"water_eFP={metrics['empty_water_fp']}/{metrics['empty_water_total']}"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold-pattern", type=Path, default=DEFAULT_FOLD_PATTERN)
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--threshold-start", type=float, default=0.40)
    p.add_argument("--threshold-stop", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.03)
    p.add_argument("--water-threshold-start", type=float, default=0.45)
    p.add_argument("--water-threshold-stop", type=float, default=0.95)
    p.add_argument("--water-threshold-step", type=float, default=0.03)
    p.add_argument("--water-k-grid", default="0,4,8,12,16,24,32")
    p.add_argument("--height-shrink", type=float, default=1.0)
    p.add_argument("--no-height-affine", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    records, fold_dirs = collect_records(args.fold_pattern, args.labels_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "thresholds": [0.5, 0.5, 0.5],
        "water_min_component": 0,
        "height_affine": False,
    }
    base_metrics = eval_records(records, base_params)

    cv_rows = []
    cv_params = []
    for fold in range(5):
        train = not_fold_records(records, fold)
        valid = fold_records(records, fold)
        params = fit_params(train, args)
        metrics = eval_records(valid, params)
        cv_rows.append(metrics)
        cv_params.append({"fold": fold, "params": params, "metrics": metrics})

    nested_metrics = aggregate_metrics(cv_rows)
    full_params = fit_params(records, args)
    full_fit_metrics = eval_records(records, full_params)

    report = {
        "fold_dirs": fold_dirs,
        "labels_dir": str(args.labels_dir),
        "n_oof_records": len(records),
        "base_default_oof_metrics": base_metrics,
        "nested_cv_metrics": nested_metrics,
        "nested_cv_folds": cv_params,
        "full_fit_params": full_params,
        "full_fit_oof_metrics_optimistic": full_fit_metrics,
        "notes": [
            "nested_cv_metrics is the less biased estimate.",
            "full_fit_oof_metrics_optimistic fits and evaluates on the same OOF ids.",
        ],
    }
    out_json = args.output_dir / "x019_oof_postprocess_report.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"OOF records: {len(records)}")
    print(f"Wrote: {out_json}")
    print()
    print(f"{'row':<24} {'bIoU':>6} {'tIoU':>6} {'wIoU':>6} {'bRMSE':>7} {'vRMSE':>7} {'score':>7} water_eFP")
    print("-" * 90)
    print_metrics("base @0.5", base_metrics)
    print_metrics("nested learned", nested_metrics)
    print_metrics("full-fit optimistic", full_fit_metrics)
    print()
    print("Full-fit params:")
    print(json.dumps(full_params, indent=2))


if __name__ == "__main__":
    main()
