"""Average per-fold .npy predictions into a single ensemble prediction set.

Reads N input directories (one per fold), each containing
<id>.npy files of shape (4, 256, 256) = [building%, veg%, water%, height_m].
Writes the per-pixel mean to <output-dir>/<id>.npy with the same shape and dtype.

Cross-fold IDs must match exactly. Any ID missing from one or more folds is
skipped with a warning. The submission requires 946 patches; the script
verifies the output count and exits non-zero if it doesn't match.
"""
import argparse
import os
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dirs", nargs="+", required=True,
                   help="One predictions directory per fold model.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--expected-count", type=int, default=946,
                   help="Expected number of output files (challenge requirement).")
    args = p.parse_args()

    if len(args.input_dirs) < 2:
        print(f"WARNING: ensembling only {len(args.input_dirs)} fold(s).", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)

    # Use the first dir as the reference set of IDs; require the rest match.
    id_sets = [set(f for f in os.listdir(d) if f.endswith(".npy"))
               for d in args.input_dirs]
    common = set.intersection(*id_sets)
    union = set.union(*id_sets)
    missing = union - common
    if missing:
        print(f"WARNING: {len(missing)} IDs not present in all folds; "
              f"averaging only over the {len(common)} common IDs.", file=sys.stderr)
        for m in sorted(missing)[:5]:
            print(f"  missing example: {m}", file=sys.stderr)

    common = sorted(common)
    print(f"Averaging {len(common)} predictions across {len(args.input_dirs)} folds...")

    n_folds = len(args.input_dirs)
    for i, fid in enumerate(common):
        acc = None
        for d in args.input_dirs:
            pred = np.load(os.path.join(d, fid)).astype(np.float32)
            if acc is None:
                acc = pred
            else:
                acc = acc + pred
        mean = acc / n_folds
        np.save(os.path.join(args.output_dir, fid), mean)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(common)}")

    written = sum(1 for f in os.listdir(args.output_dir) if f.endswith(".npy"))
    print(f"Wrote {written} files to {args.output_dir}")
    if written != args.expected_count:
        print(f"ERROR: expected {args.expected_count} files, got {written}",
              file=sys.stderr)
        sys.exit(1)
    print("OK — file count matches challenge requirement.")


if __name__ == "__main__":
    main()
