"""
Generate a dummy submission (all-zero or all-constant predictions) for probing
the official leaderboard metric. Matches the 946-patch test set layout and
produces a ready-to-upload .zip.

Examples:
    # Default: all zeros (recommended probe)
    python tools/make_dummy_submission.py

    # All-0.5 probe (tests threshold convention > vs >=)
    python tools/make_dummy_submission.py --class-value 0.5 --output-dir /tmp/dummy_05 \
        --zip-path /tmp/dummy_05.zip

    # Different test embedding source
    python tools/make_dummy_submission.py --test-embeddings-dir /path/to/test/embs
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.dataset import submission_id  # noqa: E402

DEFAULT_TEST_DIR = Path("/projects/bcrm/emb2height/data/test/alphaearth_test_emb")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-embeddings-dir", type=Path, default=DEFAULT_TEST_DIR,
                   help="Directory of test .tif embeddings — used to derive submission ids.")
    p.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs" / "dummy_zero" / "predictions",
                   help="Where to write the .npy files before zipping.")
    p.add_argument("--zip-path", type=Path, default=SCRIPT_DIR / "runs" / "dummy_zero" / "submission.zip",
                   help="Output submission zip.")
    p.add_argument("--class-value", type=float, default=0.0,
                   help="Value to fill channels 0-2 (building/veg/water). Default 0.")
    p.add_argument("--height-value", type=float, default=0.0,
                   help="Value to fill channel 3 (height, meters). Default 0.")
    p.add_argument("--patch-size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    test_files = sorted(args.test_embeddings_dir.glob("*.tif"))
    if not test_files:
        raise RuntimeError(f"No .tif files found in {args.test_embeddings_dir}")

    print(f"Found {len(test_files)} test embeddings in {args.test_embeddings_dir}")
    if len(test_files) != 946:
        print(f"WARNING: expected 946 test patches, got {len(test_files)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    arr = np.zeros((4, args.patch_size, args.patch_size), dtype=np.float32)
    arr[0:3] = args.class_value
    arr[3] = args.height_value

    for f in test_files:
        sub_id = submission_id(str(f))
        np.save(args.output_dir / f"{sub_id}.npy", arr)

    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for npy in sorted(args.output_dir.glob("*.npy")):
            zf.write(npy, arcname=f"predictions/{npy.name}")

    n_zipped = sum(1 for _ in zipfile.ZipFile(args.zip_path).namelist())
    print(f"Wrote {len(test_files)} .npy files to {args.output_dir}")
    print(f"Zipped {n_zipped} entries to {args.zip_path}")
    print(f"Probe: class_value={args.class_value}, height_value={args.height_value}")


if __name__ == "__main__":
    main()
