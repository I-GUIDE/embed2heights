import argparse
import glob
import json
import os
import re

import numpy as np
import rasterio


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
DEFAULT_LABELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "train", "labels"))
DEFAULT_SPLIT_FILE = os.path.join(SCRIPT_DIR, "splits", "split.json")

HEIGHT_SCORE_MAX = 30.0
LABEL_THRESHOLD = 0.5
WEIGHTS = {
    "mIoU_buildings": 0.25,
    "mIoU_trees": 0.15,
    "mIoU_water": 0.15,
    "RMSE_building_height": 0.25,
    "RMSE_vegetation_height": 0.20,
}

EXPERIMENTS = {
    "w18": "alphaearth_hrnet_w18_softplus_bs16_lr1e4_aux005",
    "lightunet": "lightunet_alphaearth",
    "refiner": "alphaearth_refiner_softplus_bs16_lr1e4_aux005",
    "w32": "alphaearth_hrnet_w32_softplus_bs16_lr5e5_aux005",
}


def normalize_core_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.startswith("label_"):
        base = base[len("label_"):]
    if base.startswith("pred_"):
        base = base[len("pred_"):]
    for prefix in ("gee_emb_", "tessera_emb_", "emb_", "s2_", "s1_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    for suffix in ("_embedding", "_embeddings", "_quantized", "_merged"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return re.sub(r"_\d{4}$", "", base)


def binary_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return np.nan
    return intersection / union


def mean_iou(pred_mask, true_mask):
    pos = binary_iou(pred_mask, true_mask)
    neg = binary_iou(~pred_mask, ~true_mask)
    vals = [v for v in (pos, neg) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def weighted_score(metrics):
    score = 0.0
    for key, weight in WEIGHTS.items():
        value = metrics[key]
        if np.isnan(value):
            continue
        if "RMSE" in key:
            score += max(0.0, 1.0 - value / HEIGHT_SCORE_MAX) * weight
        else:
            score += value * weight
    return score


def load_val_ids(split_file):
    with open(split_file) as f:
        return set(json.load(f)["val"])


def load_labels(labels_dir, val_ids):
    label_files = glob.glob(os.path.join(labels_dir, "**", "label_*.tif"), recursive=True)
    label_map = {normalize_core_id(path): path for path in label_files}
    labels = {}
    missing = []
    for core_id in sorted(val_ids):
        path = label_map.get(core_id)
        if path is None:
            missing.append(core_id)
            continue
        with rasterio.open(path) as src:
            labels[core_id] = src.read().astype(np.float32)
    if missing:
        raise FileNotFoundError(f"Missing labels for {len(missing)} val ids, e.g. {missing[:5]}")
    return labels


def load_predictions(runs_dir, val_ids):
    predictions = {}
    for short_name, exp_name in EXPERIMENTS.items():
        pred_dir = os.path.join(runs_dir, exp_name, "predictions")
        pred_map = {
            normalize_core_id(path): path
            for path in glob.glob(os.path.join(pred_dir, "*.npy"))
        }
        model_preds = {}
        missing = []
        for core_id in sorted(val_ids):
            path = pred_map.get(core_id)
            if path is None:
                missing.append(core_id)
                continue
            model_preds[core_id] = np.load(path).astype(np.float32)
        if missing:
            raise FileNotFoundError(
                f"Missing predictions for {short_name}/{exp_name}: {len(missing)} ids, e.g. {missing[:5]}"
            )
        predictions[short_name] = model_preds
    return predictions


def blend_sample(model_preds, core_id, spec):
    if isinstance(spec, str):
        return model_preds[spec][core_id]

    if spec["type"] == "mean":
        arrs = [model_preds[name][core_id] for name in spec["models"]]
        return np.mean(arrs, axis=0).astype(np.float32)

    if spec["type"] == "weighted_channels":
        out = np.zeros_like(model_preds["w18"][core_id], dtype=np.float32)
        for channel, weights in spec["weights"].items():
            c = int(channel)
            total = sum(weights.values())
            for name, weight in weights.items():
                out[c] += model_preds[name][core_id][c] * (weight / total)
        out[:3] = np.clip(out[:3], 0.0, 1.0)
        out[3] = np.maximum(out[3], 0.0)
        return out

    raise ValueError(f"Unknown ensemble spec: {spec}")


def evaluate_spec(labels, predictions, spec, pred_thresholds=(0.5, 0.5, 0.5)):
    miou_lists = [[], [], []]
    height_se = [0.0, 0.0]
    height_n = [0, 0]

    for core_id, label in labels.items():
        pred = blend_sample(predictions, core_id, spec)
        h = min(pred.shape[1], label.shape[1])
        w = min(pred.shape[2], label.shape[2])
        pred = pred[:, :h, :w]
        label = label[:, :h, :w]

        for channel in range(3):
            pred_mask = pred[channel] > pred_thresholds[channel]
            true_mask = label[channel] > LABEL_THRESHOLD
            miou_lists[channel].append(mean_iou(pred_mask, true_mask))

        bld_mask = label[0] > LABEL_THRESHOLD
        veg_mask = label[1] > LABEL_THRESHOLD
        if bld_mask.any():
            diff = pred[3][bld_mask] - label[3][bld_mask]
            height_se[0] += float(np.sum(diff * diff))
            height_n[0] += int(bld_mask.sum())
        if veg_mask.any():
            diff = pred[3][veg_mask] - label[3][veg_mask]
            height_se[1] += float(np.sum(diff * diff))
            height_n[1] += int(veg_mask.sum())

    metrics = {
        "mIoU_buildings": float(np.nanmean(miou_lists[0])),
        "mIoU_trees": float(np.nanmean(miou_lists[1])),
        "mIoU_water": float(np.nanmean(miou_lists[2])),
        "RMSE_building_height": float(np.sqrt(height_se[0] / height_n[0])),
        "RMSE_vegetation_height": float(np.sqrt(height_se[1] / height_n[1])),
        "n_samples": len(labels),
    }
    metrics["weighted_score"] = weighted_score(metrics)
    return metrics


def sweep_thresholds(labels, predictions, spec, grid):
    base = evaluate_spec(labels, predictions, spec)

    best_global = None
    for threshold in grid:
        metrics = evaluate_spec(labels, predictions, spec, (threshold, threshold, threshold))
        row = (metrics["weighted_score"], threshold, metrics)
        if best_global is None or row[0] > best_global[0]:
            best_global = row

    per_class_thresholds = [0.5, 0.5, 0.5]
    for channel, key in enumerate(("mIoU_buildings", "mIoU_trees", "mIoU_water")):
        best = None
        for threshold in grid:
            trial = per_class_thresholds.copy()
            trial[channel] = threshold
            metrics = evaluate_spec(labels, predictions, spec, tuple(trial))
            row = (metrics[key], threshold)
            if best is None or row[0] > best[0]:
                best = row
        per_class_thresholds[channel] = best[1]

    best_per_class = evaluate_spec(labels, predictions, spec, tuple(per_class_thresholds))
    return base, best_global, tuple(per_class_thresholds), best_per_class


def fmt_metrics(metrics):
    return (
        f"{metrics['mIoU_buildings']:.4f} "
        f"{metrics['mIoU_trees']:.4f} "
        f"{metrics['mIoU_water']:.4f} "
        f"{metrics['RMSE_building_height']:.4f} "
        f"{metrics['RMSE_vegetation_height']:.4f} "
        f"{metrics['weighted_score']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--grid-start", type=float, default=0.10)
    parser.add_argument("--grid-stop", type=float, default=0.90)
    parser.add_argument("--grid-step", type=float, default=0.025)
    args = parser.parse_args()

    val_ids = load_val_ids(args.split_file)
    print(f"Loading {len(val_ids)} validation samples...")
    labels = load_labels(args.labels_dir, val_ids)
    predictions = load_predictions(args.runs_dir, val_ids)

    grid = np.arange(args.grid_start, args.grid_stop + 1e-9, args.grid_step)

    specs = {
        "w18": "w18",
        "lightunet": "lightunet",
        "refiner": "refiner",
        "w32": "w32",
        "avg_w18_lightunet_refiner": {"type": "mean", "models": ["w18", "lightunet", "refiner"]},
        "avg_w18_lightunet": {"type": "mean", "models": ["w18", "lightunet"]},
        "avg_w18_refiner": {"type": "mean", "models": ["w18", "refiner"]},
        "avg_all4": {"type": "mean", "models": ["w18", "lightunet", "refiner", "w32"]},
        "weighted_metric_v1": {
            "type": "weighted_channels",
            "weights": {
                "0": {"lightunet": 0.45, "refiner": 0.30, "w18": 0.25},
                "1": {"w18": 0.45, "refiner": 0.35, "lightunet": 0.20},
                "2": {"lightunet": 0.50, "w18": 0.30, "refiner": 0.20},
                "3": {"w18": 0.50, "refiner": 0.35, "lightunet": 0.15},
            },
        },
        "channel_best_v1": {
            "type": "weighted_channels",
            "weights": {
                "0": {"lightunet": 1.0},
                "1": {"w18": 1.0},
                "2": {"lightunet": 1.0},
                "3": {"w18": 0.65, "refiner": 0.35},
            },
        },
    }

    print("\nBase @ pred thresholds=(0.5,0.5,0.5), label threshold fixed at 0.5")
    print(f"{'name':<32} {'bIoU':>7} {'tIoU':>7} {'wIoU':>7} {'bRMSE':>8} {'vRMSE':>8} {'score':>7}")
    print("-" * 82)
    base_rows = []
    sweep_rows = []
    for name, spec in specs.items():
        base, best_global, per_class_thresholds, best_per_class = sweep_thresholds(labels, predictions, spec, grid)
        base_rows.append((base["weighted_score"], name, base))
        sweep_rows.append((best_per_class["weighted_score"], name, best_global, per_class_thresholds, best_per_class))
        print(f"{name:<32} {fmt_metrics(base)}")

    print("\nBest base scores")
    for score, name, metrics in sorted(base_rows, reverse=True)[:8]:
        print(f"{name:<32} score={score:.4f} metrics={fmt_metrics(metrics)}")

    print("\nThreshold sweep summary")
    print("Note: prediction thresholds vary, while GT masks stay at 0.5. This estimates calibration/post-processing upside.")
    print(f"{'name':<32} {'base':>7} {'global':>16} {'per_class_thr':>24} {'per_class_score':>16}")
    print("-" * 104)
    for score, name, best_global, per_class_thresholds, best_per_class in sorted(sweep_rows, reverse=True):
        global_score, global_threshold, _ = best_global
        base_score = next(row[0] for row in base_rows if row[1] == name)
        thr = ",".join(f"{x:.3f}" for x in per_class_thresholds)
        print(
            f"{name:<32} {base_score:>7.4f} "
            f"{global_score:>7.4f}@{global_threshold:<5.3f} "
            f"{thr:>24} {best_per_class['weighted_score']:>16.4f}"
        )


if __name__ == "__main__":
    main()
