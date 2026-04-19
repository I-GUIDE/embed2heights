"""
Package a directory of test-set .npy predictions into a leaderboard submission
zip. The zip has a single top-level folder `predictions/` containing every
.npy, matching the format verified in logs/METRIC_PROBE_REPORT.md.

Typical use:
    # 1. Run inference on the test embeddings
    python predict.py --experiment-name lightunet_alphaearth_v3head \\
        --model-type lightunet \\
        --test-embeddings-dir /u/dingqi2/workspace/esa/data/test/alphaearth_test_emb \\
        --predictions-dir runs/lightunet_alphaearth_v3head/test_predictions_alphaearth

    # 2a. Continuous submission (keeps class channels as-is — large ~800 MB zip)
    python tools/make_submission.py \\
        --pred-dir runs/lightunet_alphaearth_v3head/test_predictions_alphaearth \\
        --output   runs/lightunet_alphaearth_v3head/submission_base.zip

    # 2b. Binarized submission (class channels -> {0, 1} at given thresholds;
    #     height stays float32. Smaller zip, and identical IoU when thresholds
    #     are 0.5,0.5,0.5 because the server uses pred > 0.5.)
    python tools/make_submission.py \\
        --pred-dir runs/lightunet_alphaearth_v3head/test_predictions_alphaearth \\
        --binarize-thresholds 0.5 0.5 0.5 \\
        --output   runs/lightunet_alphaearth_v3head/submission_binary05.zip

The script sanity-checks:
  - expected file count (default 946 — override with --expected-count)
  - each .npy has shape [4, H, W] with float32 / float64 dtype
  - channels 0-2 values in [0, 1]; channel 3 non-negative
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", type=Path, required=True,
                   help="Directory of test-set .npy predictions (one per patch).")
    p.add_argument("--output", type=Path, required=True, help="Output submission .zip path.")
    p.add_argument("--expected-count", type=int, default=946,
                   help="Expected number of .npy files (default 946; pass 0 to skip check).")
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip per-file shape/range validation (faster).")
    p.add_argument("--binarize-thresholds", type=float, nargs=3, default=None,
                   metavar=("BLD", "VEG", "WAT"),
                   help="If set, threshold class channels (0-2) to {0.0, 1.0} using "
                        "these per-class thresholds (pred > threshold). Channel 3 "
                        "(height) is left untouched. Cuts zip size ~4x without "
                        "changing IoU when thresholds match the server's 0.5.")
    return p.parse_args()


def validate_one(path):
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[0] != 4:
        raise ValueError(f"{path.name}: expected shape [4, H, W], got {arr.shape}")
    if arr.dtype not in (np.float32, np.float64):
        raise ValueError(f"{path.name}: expected float dtype, got {arr.dtype}")
    # Only sample ranges — full check is too slow on 946 patches
    cls = arr[:3]
    if cls.min() < -1e-4 or cls.max() > 1 + 1e-4:
        print(f"  WARN  {path.name}: class channels outside [0,1] — min={cls.min():.4f} max={cls.max():.4f}",
              file=sys.stderr)
    if arr[3].min() < -1e-4:
        print(f"  WARN  {path.name}: height channel has negative values — min={arr[3].min():.4f}",
              file=sys.stderr)
    return arr.shape


def main():
    args = parse_args()

    if not args.pred_dir.is_dir():
        raise FileNotFoundError(f"--pred-dir does not exist: {args.pred_dir}")

    files = sorted(args.pred_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {args.pred_dir}")

    print(f"Found {len(files)} .npy files in {args.pred_dir}")
    if args.expected_count > 0 and len(files) != args.expected_count:
        print(f"ERROR: expected {args.expected_count} files for submission, got {len(files)}", file=sys.stderr)
        print(f"       (pass --expected-count 0 to skip this check)", file=sys.stderr)
        sys.exit(1)

    if not args.skip_validation:
        print("Validating a sample of files...")
        sample_idxs = [0, len(files) // 2, len(files) - 1]
        shapes = set()
        for i in sample_idxs:
            shape = validate_one(files[i])
            shapes.add(shape)
        print(f"  sample shapes: {shapes}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # We always materialize the (possibly transformed) files into a temp
    # `predictions/` subdirectory and then shell out to the system `zip`
    # command (Info-ZIP). Python's zipfile module produces archives whose
    # ZipInfo fields (no UT extra timestamps, placeholder 1980 date_time,
    # lower create_version) have been observed to hang strict server-side
    # parsers — the known-working submissions were all created by Info-ZIP
    # via the `zip` CLI. Shelling out gives us byte-pattern-identical
    # output to those working zips.
    import shutil
    import subprocess
    import tempfile

    zip_cli = shutil.which("zip")
    if zip_cli is None:
        raise RuntimeError(
            "/usr/bin/zip (Info-ZIP) not found on PATH. "
            "This tool requires it to produce a server-compatible archive."
        )

    with tempfile.TemporaryDirectory(prefix="make_submission_") as tmpd:
        tmp_path = Path(tmpd)
        pred_dir_in_tmp = tmp_path / "predictions"
        pred_dir_in_tmp.mkdir()

        if args.binarize_thresholds is None:
            print(f"Staging {len(files)} files (no transform) to {pred_dir_in_tmp}")
            for npy in files:
                shutil.copy2(npy, pred_dir_in_tmp / npy.name)
        else:
            thr = np.asarray(args.binarize_thresholds, dtype=np.float32)
            print(f"Binarizing class channels at thresholds {tuple(float(t) for t in thr)} (bld, veg, wat)")
            for npy in files:
                arr = np.load(npy).astype(np.float32)
                for c in range(3):
                    arr[c] = (arr[c] > thr[c]).astype(np.float32)
                np.save(pred_dir_in_tmp / npy.name, arr)

        # Remove any pre-existing output zip so `zip` creates a fresh archive.
        if args.output.exists():
            args.output.unlink()

        print(f"Zipping via {zip_cli} -r ...")
        proc = subprocess.run(
            [zip_cli, "-r", "-q", str(args.output.absolute()), "predictions"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"zip failed (rc={proc.returncode}):\n{proc.stderr}")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nSubmission written: {args.output}  ({size_mb:.1f} MB, {len(files)} entries + 1 dir)")
    print(f"Internal layout: predictions/<id>.npy")


if __name__ == "__main__":
    main()
