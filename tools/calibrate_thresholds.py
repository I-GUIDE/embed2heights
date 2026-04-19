import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.dataset import normalize_core_id  # noqa: E402
from core.metrics import load_val_ids, build_label_map  # noqa: E402

DEFAULT_LABELS_DIR = SCRIPT_DIR.parent / "data" / "train" / "labels"
DEFAULT_SPLIT_FILE = SCRIPT_DIR / "splits" / "split.json"
CLASS_NAMES = ("building", "vegetation", "water")
CLASS_WEIGHTS = (0.25, 0.15, 0.15)
# calibrate_thresholds uses a stricter label binarization (0.5) — intentional,
# it's tuning against the old `mean(pos, neg)` IoU curve rather than the
# probe-verified leaderboard metric. Keep in sync with comments below.
LABEL_THRESHOLD = 0.5


def build_prediction_map(pred_dir):
    pred_files = sorted(pred_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files found in {pred_dir}")
    return {normalize_core_id(path): path for path in pred_files}


def load_pairs(pred_dir, labels_dir, split_file):
    val_ids = load_val_ids(split_file)
    pred_map = build_prediction_map(pred_dir)
    label_map = build_label_map(labels_dir)
    pairs = []
    missing_pred = []
    missing_label = []
    for core_id in sorted(val_ids):
        pred_path = pred_map.get(core_id)
        label_path = label_map.get(core_id)
        if pred_path is None:
            missing_pred.append(core_id)
            continue
        if label_path is None:
            missing_label.append(core_id)
            continue
        pairs.append((core_id, pred_path, label_path))
    if missing_pred:
        raise FileNotFoundError(f"Missing predictions for {len(missing_pred)} val ids, e.g. {missing_pred[:5]}")
    if missing_label:
        raise FileNotFoundError(f"Missing labels for {len(missing_label)} val ids, e.g. {missing_label[:5]}")
    return pairs


def sample_mean_iou_curve(pred_values, true_values, thresholds):
    pred_masks = pred_values[None, :] > thresholds[:, None]
    true_mask = true_values[None, :]

    pos_intersection = np.logical_and(pred_masks, true_mask).sum(axis=1)
    pos_union = np.logical_or(pred_masks, true_mask).sum(axis=1)
    neg_intersection = np.logical_and(~pred_masks, ~true_mask).sum(axis=1)
    neg_union = np.logical_or(~pred_masks, ~true_mask).sum(axis=1)

    pos = np.full(len(thresholds), np.nan, dtype=np.float32)
    neg = np.full(len(thresholds), np.nan, dtype=np.float32)
    pos_valid = pos_union > 0
    neg_valid = neg_union > 0
    pos[pos_valid] = pos_intersection[pos_valid] / pos_union[pos_valid]
    neg[neg_valid] = neg_intersection[neg_valid] / neg_union[neg_valid]
    return np.nanmean(np.stack([pos, neg], axis=0), axis=0)


def class_miou_curve(pairs, channel, thresholds):
    curves = []
    for _, pred_path, label_path in pairs:
        pred = np.load(pred_path, mmap_mode="r")
        with rasterio.open(label_path) as src:
            label = src.read(channel + 1).astype(np.float32)
        h = min(pred.shape[1], label.shape[0])
        w = min(pred.shape[2], label.shape[1])
        pred_values = np.asarray(pred[channel, :h, :w], dtype=np.float32).reshape(-1)
        true_values = (label[:h, :w] > LABEL_THRESHOLD).reshape(-1)
        curves.append(sample_mean_iou_curve(pred_values, true_values, thresholds))
    return np.nanmean(np.stack(curves, axis=0), axis=0)


def select_threshold(rows, epsilon, policy):
    best_score = max(row["miou"] for row in rows)
    candidates = [row for row in rows if row["miou"] >= best_score - epsilon]
    if policy == "best":
        return max(rows, key=lambda row: (row["miou"], -abs(row["threshold"] - 0.5)))
    if policy == "closest_to_0.5":
        return min(candidates, key=lambda row: (abs(row["threshold"] - 0.5), -row["miou"]))
    if policy == "lowest":
        return min(candidates, key=lambda row: row["threshold"])
    if policy == "highest":
        return max(candidates, key=lambda row: row["threshold"])
    raise ValueError(f"Unknown policy: {policy}")


def calibrate(pairs, thresholds, epsilon, policy):
    result = {}
    score_gain = 0.0
    for channel, (name, weight) in enumerate(zip(CLASS_NAMES, CLASS_WEIGHTS)):
        miou_curve = class_miou_curve(pairs, channel, thresholds)
        rows = [
            {"threshold": float(threshold), "miou": float(miou)}
            for threshold, miou in zip(thresholds, miou_curve)
        ]
        chosen = select_threshold(rows, epsilon=epsilon, policy=policy)
        default = min(rows, key=lambda row: abs(row["threshold"] - 0.5))
        score_gain += weight * (chosen["miou"] - default["miou"])
        result[name] = {
            "threshold": chosen["threshold"],
            "miou": chosen["miou"],
            "default_0.5_miou": default["miou"],
            "score_gain_vs_0.5": weight * (chosen["miou"] - default["miou"]),
            "curve": rows,
        }
    return result, score_gain


def materialize(pred_dir, output_dir, thresholds):
    pred_files = sorted(pred_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files found in {pred_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for pred_path in pred_files:
        pred = np.load(pred_path).astype(np.float32, copy=True)
        for channel, threshold in enumerate(thresholds):
            pred[channel] = (pred[channel] > threshold).astype(np.float32)
        np.save(output_dir / pred_path.name, pred)


def parse_args():
    parser = argparse.ArgumentParser(description="Automatically calibrate segmentation thresholds on the val split.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Validation predictions directory.")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--grid-start", type=float, default=0.05)
    parser.add_argument("--grid-stop", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.005)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0002,
        help="Treat thresholds within this class-mIoU margin as tied for robust selection.",
    )
    parser.add_argument(
        "--policy",
        choices=("best", "closest_to_0.5", "lowest", "highest"),
        default="closest_to_0.5",
        help="Tie-breaking policy inside the epsilon-best plateau.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--materialize-dir",
        type=Path,
        default=None,
        help="Optional output directory with validation fraction channels hard-thresholded to 0/1.",
    )
    parser.add_argument(
        "--apply-pred-dir",
        type=Path,
        default=None,
        help="Optional prediction directory to post-process with the selected thresholds, e.g. test predictions.",
    )
    parser.add_argument(
        "--apply-output-dir",
        type=Path,
        default=None,
        help="Output directory for --apply-pred-dir. Required when --apply-pred-dir is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.apply_pred_dir and args.apply_output_dir is None:
        raise ValueError("--apply-output-dir is required when --apply-pred-dir is set")

    thresholds = np.arange(args.grid_start, args.grid_stop + 1e-9, args.grid_step, dtype=np.float32)
    pairs = load_pairs(args.pred_dir, args.labels_dir, args.split_file)
    calibration, score_gain = calibrate(pairs, thresholds, args.epsilon, args.policy)

    selected = [calibration[name]["threshold"] for name in CLASS_NAMES]
    output = {
        "pred_dir": str(args.pred_dir),
        "labels_dir": str(args.labels_dir),
        "split_file": str(args.split_file),
        "n_val": len(pairs),
        "label_threshold": LABEL_THRESHOLD,
        "grid": {
            "start": args.grid_start,
            "stop": args.grid_stop,
            "step": args.grid_step,
        },
        "epsilon": args.epsilon,
        "policy": args.policy,
        "selected_thresholds": dict(zip(CLASS_NAMES, selected)),
        "segmentation_score_gain_vs_0.5": score_gain,
        "classes": calibration,
    }

    output_json = args.output_json or args.pred_dir.parent / "threshold_calibration.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Calibrated on {len(pairs)} validation samples")
    print(f"Policy: {args.policy}, epsilon={args.epsilon}")
    print(f"{'class':<12} {'threshold':>10} {'mIoU@thr':>10} {'mIoU@0.5':>10} {'score_gain':>11}")
    for name in CLASS_NAMES:
        row = calibration[name]
        print(
            f"{name:<12} {row['threshold']:>10.3f} {row['miou']:>10.4f} "
            f"{row['default_0.5_miou']:>10.4f} {row['score_gain_vs_0.5']:>11.5f}"
        )
    print(f"Segmentation score gain vs 0.5: {score_gain:.5f}")
    print(f"Wrote: {output_json}")

    if args.materialize_dir:
        materialize(args.pred_dir, args.materialize_dir, selected)
        print(f"Wrote calibrated predictions: {args.materialize_dir}")

    if args.apply_pred_dir:
        materialize(args.apply_pred_dir, args.apply_output_dir, selected)
        print(f"Applied thresholds to: {args.apply_pred_dir}")
        print(f"Wrote post-processed predictions: {args.apply_output_dir}")


if __name__ == "__main__":
    main()
