"""
Sweep binarization thresholds for channels 0-2 on a labeled prediction dir.

This is intentionally independent from ensemble and submission packaging. It
prints the best global threshold and greedy per-class thresholds, and can write
a JSON report that `tools/make_submission.py` can also produce/use.
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.inference.calibration import (  # noqa: E402
    format_metrics,
    sweep_thresholds,
    write_threshold_report,
)
from core.metrics import LABEL_THRESHOLD, WEIGHTS  # noqa: E402


DEFAULT_LABELS_DIR = SCRIPT_DIR.parent / "data" / "train" / "labels"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Validation prediction dir containing [4,H,W] .npy files.")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--split-file", type=Path, default=None,
                        help="Optional JSON with a 'val' key.")
    parser.add_argument("--grid-start", type=float, default=0.05)
    parser.add_argument("--grid-stop", type=float, default=0.90)
    parser.add_argument("--grid-step", type=float, default=0.025)
    parser.add_argument("--water-k-grid", default=None,
                        help="Optional comma-separated water connected-component K values to test "
                             "after per-class threshold sweep, e.g. 0,4,8,12,16,24,32.")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional JSON report path.")
    return parser.parse_args()


def parse_k_grid(value):
    if value is None:
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def main():
    args = parse_args()
    print(f"Loading predictions from {args.pred_dir}")
    result = sweep_thresholds(
        pred_dir=args.pred_dir,
        labels_dir=args.labels_dir,
        split_file=args.split_file,
        grid_start=args.grid_start,
        grid_stop=args.grid_stop,
        grid_step=args.grid_step,
        water_k_grid=parse_k_grid(args.water_k_grid),
    )

    print("\n" + "=" * 95)
    print(f"  Threshold sweep for: {args.pred_dir}")
    print(f"  Label threshold fixed at {LABEL_THRESHOLD}  (leaderboard convention)")
    print("=" * 95)
    print(f"{'Row':<34} {'iou_bld':>8} {'iou_tree':>9} {'iou_wat':>8} {'RMSE_bH':>8} {'RMSE_vH':>8} {'Score':>8}")
    print("-" * 95)
    print(f"{'base (0.5, 0.5, 0.5)':<34} {format_metrics(result.base_metrics)}")
    label = f"global-best @{result.best_global_threshold:.3f}"
    print(f"{label:<34} {format_metrics(result.best_global_metrics)}")
    thresholds = result.per_class_thresholds
    label = f"per-class ({thresholds[0]:.3f},{thresholds[1]:.3f},{thresholds[2]:.3f})"
    print(f"{label:<34} {format_metrics(result.per_class_metrics)}")
    if args.water_k_grid is not None:
        label = f"per-class + water K={result.best_water_cc_min_size}"
        print(f"{label:<34} {format_metrics(result.water_cc_metrics)}")
    print("=" * 95)
    print(f"  Weights: iou_bld={WEIGHTS['iou_buildings']:.0%}  iou_tree={WEIGHTS['iou_trees']:.0%}  "
          f"iou_wat={WEIGHTS['iou_water']:.0%}  RMSE_bH={WEIGHTS['RMSE_building_height']:.0%}  "
          f"RMSE_vH={WEIGHTS['RMSE_vegetation_height']:.0%}")

    if args.output_json:
        write_threshold_report(result, args.output_json, args.pred_dir)
        print(f"Wrote threshold report: {args.output_json}")


if __name__ == "__main__":
    main()
