"""
Build a leaderboard submission zip from prediction files.

The script can run the complete post-processing path:
  1. optionally ensemble multiple test prediction dirs,
  2. optionally sweep thresholds on a labeled validation prediction dir,
  3. binarize channels 0-2 and save the binarized predictions,
  4. package the required `predictions/*.npy` zip layout.

Existing direct packaging still works:
    python tools/make_submission.py \\
        --pred-dir runs/my_exp/test_predictions_alphaearth \\
        --output runs/my_exp/submission.zip

Complete pipeline example:
    python tools/make_submission.py \\
        --ensemble-inputs runs/a/test_predictions runs/b/test_predictions \\
        --ensemble-output-dir runs/ens/test_predictions \\
        --sweep-ensemble-inputs runs/a/predictions runs/b/predictions \\
        --binarized-output-dir runs/ens/test_predictions_binary \\
        --output runs/ens/submission.zip
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.inference.calibration import (  # noqa: E402
    binarize_predictions,
    format_metrics,
    sweep_thresholds,
    write_threshold_report,
)
from core.inference.ensemble import (  # noqa: E402
    ensemble_mean,
    ensemble_weighted,
    load_weighted_ensemble_spec,
)
from core.inference.submission import (  # noqa: E402
    package_submission,
    validate_prediction_dir,
)


DEFAULT_LABELS_DIR = SCRIPT_DIR.parent / "data" / "train" / "labels"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    source = parser.add_argument_group("prediction source")
    source.add_argument("--pred-dir", type=Path, default=None,
                        help="Existing test prediction directory to package.")
    source.add_argument("--ensemble-inputs", type=Path, nargs="+", default=None,
                        help="Two or more test prediction dirs to mean-ensemble before packaging.")
    source.add_argument("--ensemble-spec", type=Path, default=None,
                        help="Weighted ensemble JSON spec. Mutually exclusive with --ensemble-inputs.")
    source.add_argument("--ensemble-output-dir", type=Path, default=None,
                        help="Where to save ensembled test predictions. Required for persistent ensemble output; "
                             "otherwise a temporary dir is used.")

    thresholds = parser.add_argument_group("thresholds and binarization")
    thresholds.add_argument("--thresholds", "--binarize-thresholds", type=float, nargs=3, default=None,
                            metavar=("BLD", "VEG", "WAT"),
                            help="Explicit per-class thresholds for channels 0-2.")
    thresholds.add_argument("--sweep-pred-dir", type=Path, default=None,
                            help="Validation prediction dir used to sweep thresholds when --thresholds is omitted.")
    thresholds.add_argument("--sweep-ensemble-inputs", type=Path, nargs="+", default=None,
                            help="Validation prediction dirs to mean-ensemble before threshold sweep.")
    thresholds.add_argument("--sweep-ensemble-spec", type=Path, default=None,
                            help="Weighted ensemble JSON spec for validation predictions before threshold sweep.")
    thresholds.add_argument("--sweep-ensemble-output-dir", type=Path, default=None,
                            help="Where to save the validation ensemble used for threshold sweep. "
                                 "If omitted, a temporary dir is used.")
    thresholds.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    thresholds.add_argument("--split-file", type=Path, default=None,
                            help="Optional JSON with a 'val' key for threshold sweep.")
    thresholds.add_argument("--grid-start", type=float, default=0.05)
    thresholds.add_argument("--grid-stop", type=float, default=0.90)
    thresholds.add_argument("--grid-step", type=float, default=0.025)
    thresholds.add_argument("--threshold-report", type=Path, default=None,
                            help="Optional JSON report path for swept thresholds.")
    thresholds.add_argument("--binarized-output-dir", type=Path, default=None,
                            help="Where to save binarized predictions. If thresholds are set and this is omitted, "
                                 "a temporary dir is used for packaging only.")
    thresholds.add_argument("--no-binarize", action="store_true",
                            help="Package continuous predictions even if threshold sweep inputs are provided.")
    thresholds.add_argument("--water-cc-min-size", type=int, default=0, metavar="K",
                            help="Optional 8-connected water-mask filter after binarization. K=0 disables.")

    package = parser.add_argument_group("package")
    package.add_argument("--output", type=Path, required=True, help="Output submission .zip path.")
    package.add_argument("--expected-count", type=int, default=946,
                         help="Expected number of .npy files. Pass 0 to skip.")
    package.add_argument("--skip-validation", action="store_true",
                         help="Skip sample shape/range validation before packaging.")
    return parser.parse_args()


def resolve_prediction_source(args, tmp_path: Path) -> Path:
    sources = sum(value is not None for value in (args.pred_dir, args.ensemble_inputs, args.ensemble_spec))
    if sources != 1:
        raise ValueError("Provide exactly one of --pred-dir, --ensemble-inputs, or --ensemble-spec.")
    if args.ensemble_inputs and args.ensemble_spec:
        raise ValueError("--ensemble-inputs and --ensemble-spec are mutually exclusive.")

    if args.pred_dir is not None:
        if not args.pred_dir.is_dir():
            raise FileNotFoundError(f"--pred-dir does not exist: {args.pred_dir}")
        return args.pred_dir

    output_dir = args.ensemble_output_dir or (tmp_path / "ensemble_predictions")
    if args.ensemble_inputs is not None:
        count = ensemble_mean(args.ensemble_inputs, output_dir)
        print(f"Ensembled {count} files by mean: {output_dir}")
        return output_dir

    inputs, channel_weights = load_weighted_ensemble_spec(args.ensemble_spec)
    count = ensemble_weighted(inputs, channel_weights, output_dir)
    print(f"Ensembled {count} files with weighted spec {args.ensemble_spec}: {output_dir}")
    return output_dir


def resolve_sweep_prediction_dir(args, tmp_path):
    sources = sum(value is not None for value in (
        args.sweep_pred_dir,
        args.sweep_ensemble_inputs,
        args.sweep_ensemble_spec,
    ))
    if sources == 0:
        return None
    if sources != 1:
        raise ValueError("Provide at most one of --sweep-pred-dir, --sweep-ensemble-inputs, or --sweep-ensemble-spec.")

    if args.sweep_pred_dir is not None:
        return args.sweep_pred_dir

    output_dir = args.sweep_ensemble_output_dir or (tmp_path / "sweep_ensemble_predictions")
    if args.sweep_ensemble_inputs is not None:
        count = ensemble_mean(args.sweep_ensemble_inputs, output_dir)
        print(f"Ensembled {count} validation files by mean for threshold sweep: {output_dir}")
        return output_dir

    inputs, channel_weights = load_weighted_ensemble_spec(args.sweep_ensemble_spec)
    count = ensemble_weighted(inputs, channel_weights, output_dir)
    print(f"Ensembled {count} validation files with weighted spec for threshold sweep: {output_dir}")
    return output_dir


def resolve_thresholds(args, tmp_path):
    if args.no_binarize:
        return None
    if args.thresholds is not None:
        return tuple(float(value) for value in args.thresholds)
    sweep_pred_dir = resolve_sweep_prediction_dir(args, tmp_path)
    if sweep_pred_dir is None:
        return None

    result = sweep_thresholds(
        pred_dir=sweep_pred_dir,
        labels_dir=args.labels_dir,
        split_file=args.split_file,
        grid_start=args.grid_start,
        grid_stop=args.grid_stop,
        grid_step=args.grid_step,
    )
    print(
        "Swept thresholds: "
        f"bld={result.per_class_thresholds[0]:.3f} "
        f"veg={result.per_class_thresholds[1]:.3f} "
        f"wat={result.per_class_thresholds[2]:.3f}"
    )
    print(f"Validation score @ thresholds: {format_metrics(result.per_class_metrics)}")
    if args.threshold_report:
        write_threshold_report(result, args.threshold_report, sweep_pred_dir)
        print(f"Wrote threshold report: {args.threshold_report}")
    return result.per_class_thresholds


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    args = parse_args()

    with tempfile.TemporaryDirectory(prefix="make_submission_") as tmpd:
        tmp_path = Path(tmpd)
        pred_dir = resolve_prediction_source(args, tmp_path)
        thresholds = resolve_thresholds(args, tmp_path)

        package_dir = pred_dir
        if thresholds is not None:
            package_dir = args.binarized_output_dir or (tmp_path / "binarized_predictions")
            count = binarize_predictions(
                pred_dir=pred_dir,
                output_dir=package_dir,
                thresholds=thresholds,
                water_cc_min_size=args.water_cc_min_size,
            )
            print(
                f"Binarized {count} files at "
                f"({thresholds[0]:.3f}, {thresholds[1]:.3f}, {thresholds[2]:.3f}): {package_dir}"
            )
        elif args.binarized_output_dir is not None:
            print("WARN: --binarized-output-dir was provided but no thresholds were set; packaging continuous predictions.")

        if not args.skip_validation:
            shapes = validate_prediction_dir(package_dir, expected_count=args.expected_count, sample_only=True)
            print(f"Validated sample shapes: {shapes}")

        package_submission(package_dir, args.output)
        size_mb = args.output.stat().st_size / (1024 * 1024)
        print(f"Submission written: {args.output} ({size_mb:.1f} MB)")
        print("Internal layout: predictions/<id>.npy")

        manifest_path = args.output.with_suffix(".manifest.json")
        write_manifest(
            manifest_path,
            {
                "source_pred_dir": str(pred_dir),
                "packaged_pred_dir": str(package_dir),
                "output": str(args.output),
                "thresholds": list(thresholds) if thresholds is not None else None,
                "water_cc_min_size": args.water_cc_min_size,
                "expected_count": args.expected_count,
            },
        )
        print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
