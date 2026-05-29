"""
Generate train.json and val.json split files from the labeled training data.

This ensures reproducible and independent train/validation splits across all
baselines. The split is based on normalized core IDs, so the same physical
patch is always in the same split regardless of which embedding backbone
is used.

Usage:
    python generate_split.py                          # default 80/20 split
    python generate_split.py --val-ratio 0.15         # custom split ratio
    python generate_split.py --output-dir ./splits    # custom output directory

The generated files (train.json, val.json) contain lists of normalized core IDs.
Pass them to train.py via --split-file to enforce a consistent split.
"""

import argparse
import json
import os
import glob
import sys

from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from core.data.discovery import normalize_core_id  # noqa: E402

DATA_ROOT = "/projects/bcrm/emb2height/data"
DEFAULT_LABELS_DIR = os.path.join(DATA_ROOT, "train", "labels")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "splits")
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR,
                   help=f"Directory containing label_*.tif files (default: {DEFAULT_LABELS_DIR})")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Directory to write train.json and val.json (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                   help=f"Fraction of data for validation (default: {DEFAULT_VAL_RATIO})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random seed for split (default: {DEFAULT_SEED})")
    return p.parse_args()


def main():
    args = parse_args()

    # Find all label files and extract core IDs
    label_files = sorted(glob.glob(os.path.join(args.labels_dir, "**", "label_*.tif"), recursive=True))
    if not label_files:
        raise FileNotFoundError(f"No label_*.tif files found in {args.labels_dir}")

    core_ids = sorted(set(normalize_core_id(f) for f in label_files))
    print(f"Found {len(core_ids)} unique patches in {args.labels_dir}")

    # Split
    train_ids, val_ids = train_test_split(
        core_ids, test_size=args.val_ratio, random_state=args.seed
    )
    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val (ratio={args.val_ratio}, seed={args.seed})")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Save as individual files
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")
    with open(train_path, "w") as f:
        json.dump(train_ids, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_ids, f, indent=2)

    # Also save as a combined split.json (compatible with train.py --split-file)
    combined_path = os.path.join(args.output_dir, "split.json")
    with open(combined_path, "w") as f:
        json.dump({"train": train_ids, "val": val_ids}, f, indent=2)

    print(f"\nSaved:")
    print(f"  {train_path}    ({len(train_ids)} IDs)")
    print(f"  {val_path}      ({len(val_ids)} IDs)")
    print(f"  {combined_path} (combined, compatible with train.py --split-file)")


if __name__ == "__main__":
    main()
