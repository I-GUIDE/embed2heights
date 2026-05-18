"""Binarize channels 0-2 of an ensemble predictions directory at the
given per-class thresholds. Channel 3 (height) is left untouched.

Use after tools/sweep_thresholds.py reveals the optimal per-class
thresholds on val — apply them to the *test* ensemble dir to produce
a binarized submission.
"""
import argparse
import glob
import os
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True,
                   help="Continuous prediction directory (e.g. "
                        "runs/submission/ctaskattn_5fold_ensemble).")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the binarized .npy files.")
    p.add_argument("--thresholds", nargs=3, type=float, required=True,
                   metavar=("B", "V", "W"),
                   help="Per-class thresholds for {building, vegetation, water}.")
    p.add_argument("--expected-count", type=int, default=946)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not files:
        sys.exit(f"No .npy files in {args.input_dir}")
    t_b, t_v, t_w = args.thresholds
    print(f"Binarizing {len(files)} files at thresholds "
          f"(b={t_b:.3f}, v={t_v:.3f}, w={t_w:.3f})")
    print(f"  in:  {args.input_dir}")
    print(f"  out: {args.output_dir}")

    for i, pf in enumerate(files):
        a = np.load(pf).astype(np.float32)
        a[0] = (a[0] > t_b).astype(np.float32)
        a[1] = (a[1] > t_v).astype(np.float32)
        a[2] = (a[2] > t_w).astype(np.float32)
        # height (channel 3) untouched
        np.save(os.path.join(args.output_dir, os.path.basename(pf)), a)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")

    written = sum(1 for f in os.listdir(args.output_dir) if f.endswith(".npy"))
    print(f"Wrote {written} binarized files.")
    if written != args.expected_count:
        sys.exit(f"ERROR: expected {args.expected_count}, got {written}")
    print("OK — count matches.")


if __name__ == "__main__":
    main()
