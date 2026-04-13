#!/usr/bin/env python3
"""Download the embed2heights dataset from EOTDL.

Requires authentication. Run `eotdl auth login` first if you haven't already.

Usage:
    python download_data.py
    python download_data.py --path ./data
"""

import argparse
import sys


def check_auth():
    """Verify that the user is authenticated with EOTDL."""
    try:
        from eotdl.auth import is_logged
    except ImportError:
        print("ERROR: eotdl package not installed.")
        print("Install it with:  pip install eotdl")
        print("Or create the conda environment:  conda env create -f environment.yml")
        sys.exit(1)

    try:
        if not is_logged():
            print("ERROR: Not authenticated with EOTDL.")
            print("Please run:  eotdl auth login")
            print("Then re-run this script.")
            sys.exit(1)
    except Exception:
        # Auth check may behave differently across versions;
        # let stage_dataset raise a clear error if auth is missing.
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Download the embed2heights dataset from EOTDL."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data",
        help="Output directory for the dataset (default: ./data)",
    )
    args = parser.parse_args()

    check_auth()

    from eotdl.datasets import stage_dataset

    print(f"Downloading embed2heights dataset to: {args.path}")
    print("This may take a while depending on your connection speed...")
    print()

    try:
        stage_dataset("embed2heights", path=args.path, assets=True)
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print()
        print("Common fixes:")
        print("  1. Check authentication:  eotdl auth login")
        print("  2. Check internet connection")
        print("  3. Check disk space (dataset is several GB)")
        sys.exit(1)

    print()
    print(f"Download complete! Dataset saved to: {args.path}")
    print()
    print("Example training command:")
    print(f"  python train.py \\")
    print(f"      --train-embeddings-dir {args.path}/embed2heights/<embedding_dir> \\")
    print(f"      --train-targets-dir {args.path}/embed2heights/<labels_dir>")


if __name__ == "__main__":
    main()
