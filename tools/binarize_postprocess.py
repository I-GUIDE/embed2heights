"""Binarize predictions with optional per-class connected-component size
filtering. Same as tools/binarize_ensemble.py but adds --min-water-size K to
remove spurious water components smaller than K pixels (and analogous
--min-building-size, --min-veg-size).

Motivation: teammate observed +0.021 iou_wat from a K=12 water postprocess.
Tiny scattered water predictions are usually noise; removing them improves
IoU without retraining.
"""
import argparse
import glob
import os
import sys
import numpy as np
from scipy.ndimage import label, find_objects


def remove_small_components(mask, min_size):
    if min_size <= 0:
        return mask
    labels, n = label(mask.astype(np.uint8))
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background
    keep = counts >= min_size
    return keep[labels].astype(mask.dtype)


def morphological_closing(mask, kernel_size):
    """Binary closing: dilate then erode. Fills small gaps inside positive
    regions, increasing recall on under-segmented classes."""
    if kernel_size <= 0:
        return mask
    from scipy.ndimage import binary_closing
    struct = np.ones((kernel_size, kernel_size), dtype=bool)
    return binary_closing(mask.astype(bool), structure=struct).astype(mask.dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--thresholds", nargs=3, type=float, required=True,
                   metavar=("B", "V", "W"))
    p.add_argument("--min-building-size", type=int, default=0)
    p.add_argument("--min-veg-size", type=int, default=0)
    p.add_argument("--min-water-size", type=int, default=0)
    p.add_argument("--max-height", type=float, default=None,
                   help="Cap height predictions at this value (m).")
    p.add_argument("--expected-count", type=int, default=946)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not files:
        sys.exit(f"No .npy files in {args.input_dir}")
    t_b, t_v, t_w = args.thresholds
    print(f"Binarizing {len(files)} files at thresholds "
          f"(b={t_b:.3f}, v={t_v:.3f}, w={t_w:.3f}) | "
          f"min sizes b={args.min_building_size} v={args.min_veg_size} w={args.min_water_size}")
    print(f"  in:  {args.input_dir}")
    print(f"  out: {args.output_dir}")

    for i, pf in enumerate(files):
        a = np.load(pf).astype(np.float32)
        a[0] = (a[0] > t_b).astype(np.float32)
        a[1] = (a[1] > t_v).astype(np.float32)
        a[2] = (a[2] > t_w).astype(np.float32)
        # Apply min-component-size filter per class
        a[0] = remove_small_components(a[0], args.min_building_size)
        a[1] = remove_small_components(a[1], args.min_veg_size)
        a[2] = remove_small_components(a[2], args.min_water_size)
        # Cap height predictions if requested (kill outliers)
        if args.max_height is not None:
            a[3] = np.minimum(a[3], args.max_height)
        np.save(os.path.join(args.output_dir, os.path.basename(pf)), a)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")

    written = sum(1 for f in os.listdir(args.output_dir) if f.endswith(".npy"))
    print(f"Wrote {written} binarized files.")


if __name__ == "__main__":
    main()
