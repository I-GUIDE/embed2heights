"""Threshold and post-processing calibration helpers."""

import json
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import rasterio

from core.data.discovery import normalize_core_id
from core.inference.postprocess import (
    apply_height_channel,
    apply_water_cc_filter,
    largest_component_size,
)
from core.metrics import (
    LABEL_THRESHOLD,
    WEIGHTS,
    binary_iou,
    build_label_map,
    compute_weighted_score,
    load_val_ids,
)


CLASS_NAMES = ("building", "vegetation", "water")
CLASS_KEYS = ("iou_buildings", "iou_trees", "iou_water")


ThresholdSweepResult = namedtuple(
    "ThresholdSweepResult",
    [
        "base_metrics",
        "best_global_threshold",
        "best_global_metrics",
        "per_class_thresholds",
        "per_class_metrics",
        "best_water_cc_min_size",
        "water_cc_metrics",
        "n_samples",
    ],
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def prediction_map(pred_dir):
    paths = sorted(Path(pred_dir).glob("*.npy"))
    if not paths:
        raise FileNotFoundError("No .npy predictions found in {}".format(pred_dir))
    return {normalize_core_id(str(path)): path for path in paths}


def load_labeled_predictions(pred_dir, labels_dir, split_file=None):
    val_ids = load_val_ids(str(split_file)) if split_file else None
    label_map = build_label_map(str(labels_dir))
    predictions, labels = {}, {}
    missing_labels = []

    for path in sorted(Path(pred_dir).glob("*.npy")):
        cid = normalize_core_id(path)
        if val_ids is not None and cid not in val_ids:
            continue
        if cid not in label_map:
            missing_labels.append(cid)
            continue
        predictions[cid] = np.load(path).astype(np.float32)
        with rasterio.open(label_map[cid]) as src:
            labels[cid] = src.read().astype(np.float32)

    if missing_labels:
        print(f"WARN: {len(missing_labels)} predictions had no matching label and were skipped.", file=sys.stderr)
    if not predictions:
        raise RuntimeError("No matched pred/label pairs after filtering.")
    return predictions, labels


def collect_oof_records(fold_pattern, labels_dir, n_folds=5, predictions_subdir="predictions", split_name="split.json"):
    label_map = build_label_map(str(labels_dir))
    records = []
    fold_dirs = []
    seen = set()

    for fold in range(int(n_folds)):
        fold_dir = Path(str(fold_pattern).format(fold=fold))
        split_file = fold_dir / split_name
        pred_dir = fold_dir / predictions_subdir
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        pred_map = prediction_map(pred_dir)
        split = load_json(split_file)
        val_ids = sorted(split["val"])
        fold_dirs.append(str(fold_dir))

        for core_id in val_ids:
            if core_id in seen:
                raise RuntimeError("Duplicate held-out id across folds: {}".format(core_id))
            if core_id not in pred_map:
                raise FileNotFoundError("Missing OOF prediction for {} in {}".format(core_id, pred_dir))
            if core_id not in label_map:
                raise FileNotFoundError("Missing label for {} under {}".format(core_id, labels_dir))
            seen.add(core_id)
            records.append({
                "core_id": core_id,
                "fold": fold,
                "pred_path": str(pred_map[core_id]),
                "label_path": label_map[core_id],
            })

    return records, fold_dirs


def read_pair(record):
    pred = np.load(record["pred_path"]).astype(np.float32)
    with rasterio.open(record["label_path"]) as src:
        label = src.read().astype(np.float32)
    h = min(pred.shape[1], label.shape[1])
    w = min(pred.shape[2], label.shape[2])
    return pred[:, :h, :w], label[:, :h, :w]


def evaluate_at_thresholds(labels, predictions, thresholds, water_cc_min_size=0):
    iou_lists = [[], [], []]
    rmse_bld, rmse_veg = [], []

    for cid, pred in predictions.items():
        label = labels[cid]
        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        pred = pred[:, :h, :w]
        label = label[:, :h, :w]

        for c in range(3):
            pred_mask = pred[c] > thresholds[c]
            if c == 2 and water_cc_min_size > 0:
                pred_mask = apply_water_cc_filter(pred_mask, water_cc_min_size)
            iou_lists[c].append(binary_iou(pred_mask, label[c] > LABEL_THRESHOLD))

        for ch, bucket in ((0, rmse_bld), (1, rmse_veg)):
            mask = label[ch] > LABEL_THRESHOLD
            if mask.any():
                diff = pred[3][mask] - label[3][mask]
                bucket.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))

    metrics = {
        "iou_buildings": mean_valid(iou_lists[0]),
        "iou_trees": mean_valid(iou_lists[1]),
        "iou_water": mean_valid(iou_lists[2]),
        "RMSE_building_height": mean_valid(rmse_bld),
        "RMSE_vegetation_height": mean_valid(rmse_veg),
        "n_samples": len(predictions),
    }
    score, parts = compute_weighted_score(metrics)
    metrics["weighted_score"] = score
    metrics["score_parts"] = parts
    return metrics


def mean_valid(values):
    vals = [value for value in values if not np.isnan(value)]
    return float(np.mean(vals)) if vals else float("nan")


def sweep_water_cc(labels, predictions, thresholds, k_values):
    best_k = 0
    best_metrics = evaluate_at_thresholds(labels, predictions, thresholds, water_cc_min_size=0)
    rows = [{"k": 0, "metrics": best_metrics}]
    for k in k_values:
        k = int(k)
        if k == 0:
            continue
        metrics = evaluate_at_thresholds(labels, predictions, thresholds, water_cc_min_size=k)
        rows.append({"k": k, "metrics": metrics})
        if metrics["weighted_score"] > best_metrics["weighted_score"]:
            best_k = k
            best_metrics = metrics
    return best_k, best_metrics, rows


def sweep_thresholds(
    pred_dir,
    labels_dir,
    split_file=None,
    grid_start=0.05,
    grid_stop=0.90,
    grid_step=0.025,
    water_k_grid=None,
):
    predictions, labels = load_labeled_predictions(pred_dir, labels_dir, split_file)
    grid = np.arange(grid_start, grid_stop + 1e-9, grid_step)

    base = evaluate_at_thresholds(labels, predictions, (0.5, 0.5, 0.5))
    best_global_metrics, best_global_threshold = None, 0.5
    for threshold in grid:
        metrics = evaluate_at_thresholds(labels, predictions, (float(threshold),) * 3)
        if best_global_metrics is None or metrics["weighted_score"] > best_global_metrics["weighted_score"]:
            best_global_metrics, best_global_threshold = metrics, float(threshold)

    per_class = [0.5, 0.5, 0.5]
    for channel, key in enumerate(CLASS_KEYS):
        best_t, best_value = 0.5, -1.0
        for threshold in grid:
            trial = per_class.copy()
            trial[channel] = float(threshold)
            metrics = evaluate_at_thresholds(labels, predictions, tuple(trial))
            if metrics[key] > best_value:
                best_t, best_value = float(threshold), float(metrics[key])
        per_class[channel] = best_t
    per_class_metrics = evaluate_at_thresholds(labels, predictions, tuple(per_class))
    water_cc_best_k = 0
    water_cc_metrics = per_class_metrics
    if water_k_grid:
        water_cc_best_k, water_cc_metrics, _ = sweep_water_cc(
            labels,
            predictions,
            tuple(per_class),
            water_k_grid,
        )

    return ThresholdSweepResult(
        base_metrics=base,
        best_global_threshold=best_global_threshold,
        best_global_metrics=best_global_metrics,
        per_class_thresholds=tuple(per_class),
        per_class_metrics=per_class_metrics,
        best_water_cc_min_size=water_cc_best_k,
        water_cc_metrics=water_cc_metrics,
        n_samples=len(predictions),
    )


def write_threshold_report(result, output_json, pred_dir):
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pred_dir": str(pred_dir),
        "label_threshold": LABEL_THRESHOLD,
        "class_names": CLASS_NAMES,
        "weights": WEIGHTS,
        "n_samples": result.n_samples,
        "base_metrics_at_0_5": result.base_metrics,
        "best_global_threshold": result.best_global_threshold,
        "best_global_metrics": result.best_global_metrics,
        "per_class_thresholds": dict(zip(CLASS_NAMES, result.per_class_thresholds)),
        "per_class_metrics": result.per_class_metrics,
        "best_water_cc_min_size": result.best_water_cc_min_size,
        "water_cc_metrics": result.water_cc_metrics,
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def fit_height_affine(records, shrink):
    sums = {
        "building": {"n": 0, "x": 0.0, "y": 0.0, "xx": 0.0, "xy": 0.0},
        "vegetation": {"n": 0, "x": 0.0, "y": 0.0, "xx": 0.0, "xy": 0.0},
    }
    for record in records:
        pred, label = read_pair(record)
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


def eval_records(records, params):
    ious = [[], [], []]
    rmse_b, rmse_v = [], []
    empty_water_fp = 0
    empty_water_total = 0

    thresholds = params["thresholds"]
    water_k = int(params.get("water_min_component", 0))

    for record in records:
        pred, label = read_pair(record)
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

        height = apply_height_channel(pred, params)
        for ch, bucket in ((0, rmse_b), (1, rmse_v)):
            gt = label[ch] > LABEL_THRESHOLD
            if gt.any():
                diff = height[gt].astype(np.float64) - label[3][gt].astype(np.float64)
                bucket.append(float(np.sqrt(np.mean(diff * diff))))

    metrics = {
        "iou_buildings": mean_valid(ious[0]),
        "iou_trees": mean_valid(ious[1]),
        "iou_water": mean_valid(ious[2]),
        "RMSE_building_height": mean_valid(rmse_b),
        "RMSE_vegetation_height": mean_valid(rmse_v),
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
        pred, label = read_pair(record)
        gt = label[channel] > LABEL_THRESHOLD
        prob = pred[channel]
        for i, threshold in enumerate(grid):
            sums[i] += binary_iou(prob > threshold, gt)
    means = sums / max(1, len(records))
    idx = int(np.argmax(means))
    return float(grid[idx]), float(means[idx])


def tune_water(records, water_grid, k_grid):
    threshold_sums = np.zeros(len(water_grid), dtype=np.float64)
    for record in records:
        pred, label = read_pair(record)
        gt = label[2] > LABEL_THRESHOLD
        prob = pred[2]
        for i, threshold in enumerate(water_grid):
            threshold_sums[i] += binary_iou(prob > threshold, gt)
    threshold_means = threshold_sums / max(1, len(records))

    candidate_idxs = set(np.argsort(threshold_means)[-5:].tolist())
    nearest_05 = int(np.argmin(np.abs(water_grid - 0.5)))
    candidate_idxs.add(nearest_05)
    candidate_thresholds = [float(water_grid[i]) for i in sorted(candidate_idxs)]

    best = {
        "threshold": float(water_grid[int(np.argmax(threshold_means))]),
        "k": 0,
        "iou": float(np.max(threshold_means)),
        "metrics": None,
    }
    combo_sums = {
        (threshold, int(k)): 0.0
        for threshold in candidate_thresholds
        for k in k_grid
    }
    for record in records:
        pred, label = read_pair(record)
        gt = label[2] > LABEL_THRESHOLD
        prob = pred[2]
        gt_empty = not gt.any()
        for threshold in candidate_thresholds:
            mask = prob > threshold
            comp = largest_component_size(mask)
            for k in k_grid:
                if comp < int(k):
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


def fit_params(
    records,
    threshold_start=0.40,
    threshold_stop=0.95,
    threshold_step=0.03,
    water_threshold_start=0.45,
    water_threshold_stop=0.95,
    water_threshold_step=0.03,
    water_k_grid="0,4,8,12,16,24,32",
    height_shrink=1.0,
    height_affine=True,
):
    threshold_grid = np.arange(threshold_start, threshold_stop + 1e-9, threshold_step)
    water_grid = np.arange(water_threshold_start, water_threshold_stop + 1e-9, water_threshold_step)
    k_grid = [int(value) for value in str(water_k_grid).split(",") if value.strip()]

    b_t, b_iou = tune_class_threshold(records, 0, threshold_grid)
    v_t, v_iou = tune_class_threshold(records, 1, threshold_grid)
    water = tune_water(records, water_grid, k_grid)

    params = {
        "thresholds": [b_t, v_t, water["threshold"]],
        "water_min_component": water["k"],
        "height_affine": bool(height_affine),
        "fit_details": {
            "building_iou": b_iou,
            "vegetation_iou": v_iou,
            "water_iou": water["iou"],
        },
    }
    if params["height_affine"]:
        params["height_affine_params"] = fit_height_affine(records, shrink=height_shrink)
    return params


def fold_records(records, fold):
    return [record for record in records if record["fold"] == fold]


def not_fold_records(records, fold):
    return [record for record in records if record["fold"] != fold]


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


def run_nested_oof_cv(records, n_folds=5, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}
    base_params = {
        "thresholds": [0.5, 0.5, 0.5],
        "water_min_component": 0,
        "height_affine": False,
    }
    base_metrics = eval_records(records, base_params)

    cv_rows = []
    cv_params = []
    for fold in range(int(n_folds)):
        train = not_fold_records(records, fold)
        valid = fold_records(records, fold)
        params = fit_params(train, **fit_kwargs)
        metrics = eval_records(valid, params)
        cv_rows.append(metrics)
        cv_params.append({"fold": fold, "params": params, "metrics": metrics})

    nested_metrics = aggregate_metrics(cv_rows)
    full_params = fit_params(records, **fit_kwargs)
    full_fit_metrics = eval_records(records, full_params)

    return {
        "base_default_oof_metrics": base_metrics,
        "nested_cv_metrics": nested_metrics,
        "nested_cv_folds": cv_params,
        "full_fit_params": full_params,
        "full_fit_oof_metrics_optimistic": full_fit_metrics,
    }


def build_oof_report(fold_dirs, labels_dir, records, cv_result, notes=None):
    report = {
        "fold_dirs": fold_dirs,
        "labels_dir": str(labels_dir),
        "n_oof_records": len(records),
    }
    report.update(cv_result)
    report["notes"] = notes or [
        "nested_cv_metrics is the less biased estimate.",
        "full_fit_oof_metrics_optimistic fits and evaluates on the same OOF ids.",
    ]
    return report


def format_metrics(metrics):
    return (
        f"{metrics['iou_buildings']:.4f} {metrics['iou_trees']:.4f} {metrics['iou_water']:.4f} "
        f"{metrics['RMSE_building_height']:.4f} {metrics['RMSE_vegetation_height']:.4f} "
        f"{metrics['weighted_score']:.4f}"
    )
