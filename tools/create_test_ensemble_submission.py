import argparse
import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_W18_DIR = (
    SCRIPT_DIR
    / "runs"
    / "alphaearth_hrnet_w18_softplus_bs16_lr1e4_aux005"
    / "test_predictions_alphaearth"
)
DEFAULT_LIGHTUNET_DIR = SCRIPT_DIR / "runs" / "lightunet_alphaearth" / "test_predictions_alphaearth"
DEFAULT_REFINER_DIR = (
    SCRIPT_DIR
    / "runs"
    / "alphaearth_refiner_softplus_bs16_lr1e4_aux005"
    / "test_predictions_alphaearth"
)
DEFAULT_RAW_OUTPUT_DIR = SCRIPT_DIR / "runs" / "alphaearth_weighted_metric_v1_test" / "predictions"
DEFAULT_CAL_OUTPUT_DIR = SCRIPT_DIR / "runs" / "alphaearth_weighted_metric_v1_calibrated_test" / "predictions"

WEIGHTED_METRIC_V1 = {
    "0": {"lightunet": 0.45, "refiner": 0.30, "w18": 0.25},
    "1": {"w18": 0.45, "refiner": 0.35, "lightunet": 0.20},
    "2": {"lightunet": 0.50, "w18": 0.30, "refiner": 0.20},
    "3": {"w18": 0.50, "refiner": 0.35, "lightunet": 0.15},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create AlphaEarth weighted ensemble test predictions for submission."
    )
    parser.add_argument("--w18-dir", type=Path, default=DEFAULT_W18_DIR)
    parser.add_argument("--lightunet-dir", type=Path, default=DEFAULT_LIGHTUNET_DIR)
    parser.add_argument("--refiner-dir", type=Path, default=DEFAULT_REFINER_DIR)
    parser.add_argument("--raw-output-dir", type=Path, default=DEFAULT_RAW_OUTPUT_DIR)
    parser.add_argument("--calibrated-output-dir", type=Path, default=DEFAULT_CAL_OUTPUT_DIR)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=3,
        default=(0.575, 0.900, 0.900),
        metavar=("BUILDING", "VEGETATION", "WATER"),
        help="Per-class prediction thresholds for the calibrated hard-mask output.",
    )
    return parser.parse_args()


def npy_map(path):
    if not path.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {path}")
    files = sorted(path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy predictions found in: {path}")
    return {file.name: file for file in files}


def check_same_files(named_maps):
    reference_name, reference = next(iter(named_maps.items()))
    reference_files = set(reference)
    for name, file_map in named_maps.items():
        missing = sorted(reference_files - set(file_map))
        extra = sorted(set(file_map) - reference_files)
        if missing or extra:
            raise RuntimeError(
                f"File mismatch for {name} vs {reference_name}: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
    return sorted(reference_files)


def load_prediction(path):
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.shape != (4, 256, 256):
        raise ValueError(f"Unexpected shape for {path}: {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {path}")
    return arr


def weighted_ensemble(preds):
    out = np.zeros_like(preds["w18"], dtype=np.float32)
    for channel, weights in WEIGHTED_METRIC_V1.items():
        channel_idx = int(channel)
        total = float(sum(weights.values()))
        for model_name, weight in weights.items():
            out[channel_idx] += preds[model_name][channel_idx] * np.float32(weight / total)
    out[:3] = np.clip(out[:3], 0.0, 1.0)
    out[3] = np.maximum(out[3], 0.0)
    return out.astype(np.float32, copy=False)


def calibrated_hard_masks(raw, thresholds):
    out = raw.copy()
    for channel, threshold in enumerate(thresholds):
        out[channel] = (raw[channel] > threshold).astype(np.float32)
    return out


def write_manifest(output_dir, source_dirs, thresholds, calibrated):
    manifest = {
        "scheme": "weighted_metric_v1",
        "sources": {name: str(path) for name, path in source_dirs.items()},
        "weights": WEIGHTED_METRIC_V1,
        "calibrated_hard_masks": calibrated,
        "thresholds": {
            "building": thresholds[0],
            "vegetation": thresholds[1],
            "water": thresholds[2],
        }
        if calibrated
        else None,
        "channels": ["building_fraction", "vegetation_fraction", "water_fraction", "height_m"],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with (output_dir.parent / "ensemble_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main():
    args = parse_args()
    source_dirs = {
        "w18": args.w18_dir,
        "lightunet": args.lightunet_dir,
        "refiner": args.refiner_dir,
    }
    file_maps = {name: npy_map(path) for name, path in source_dirs.items()}
    filenames = check_same_files(file_maps)

    args.raw_output_dir.mkdir(parents=True, exist_ok=True)
    args.calibrated_output_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        preds = {
            name: load_prediction(file_maps[name][filename])
            for name in ("w18", "lightunet", "refiner")
        }
        raw = weighted_ensemble(preds)
        np.save(args.raw_output_dir / filename, raw)
        np.save(args.calibrated_output_dir / filename, calibrated_hard_masks(raw, args.thresholds))

    write_manifest(args.raw_output_dir, source_dirs, args.thresholds, calibrated=False)
    write_manifest(args.calibrated_output_dir, source_dirs, args.thresholds, calibrated=True)

    print(f"Created raw ensemble predictions: {args.raw_output_dir}")
    print(f"Created calibrated ensemble predictions: {args.calibrated_output_dir}")
    print(f"Files: {len(filenames)}")


if __name__ == "__main__":
    main()
