"""Binarize predictions with density-adaptive per-tile building threshold.

The building threshold for each tile is predicted from that tile's own
prediction-density features (mean, p90, p99, frac>0.5 of the building
probability channel) using a linear regression fit on val data.

Coefficients are loaded from a JSON spec. Veg and water use global thresholds
(per-tile density doesn't correlate with their optimal thresholds — see
[[density-adaptive-building-threshold-breakthrough]] memory).

Example spec JSON:
  {
    "regression": {
      "feature_names": ["mean_bld", "p90_bld", "p99_bld", "frac05_bld", "intercept"],
      "coefs":         [-2.2241,    -0.0918,    0.0860,    1.3180,       0.6477],
      "clip": [0.30, 0.95]
    },
    "global_thresholds": [0.640, 0.570, 0.820],
    "k_water": 14
  }
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))


def features_bld(prob_bld):
    return [
        float(prob_bld.mean()),
        float(np.percentile(prob_bld, 90)),
        float(np.percentile(prob_bld, 99)),
        float((prob_bld > 0.5).mean()),
        1.0,  # intercept
    ]


def remove_small_components(mask, min_size):
    if min_size <= 0:
        return mask
    from scipy.ndimage import label
    labels, n = label(mask.astype(np.uint8))
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = counts >= min_size
    return keep[labels].astype(mask.dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--spec", required=True, help="JSON spec with regression coefs + globals.")
    p.add_argument("--min-water-size", type=int, default=None,
                   help="Override K-water from spec.")
    p.add_argument("--max-height", type=float, default=None)
    args = p.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    coefs = np.array(spec["regression"]["coefs"], dtype=np.float64)
    clip_lo, clip_hi = spec["regression"]["clip"]
    global_t = tuple(spec["global_thresholds"])
    k_water = args.min_water_size if args.min_water_size is not None else int(spec.get("k_water", 0))

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not files:
        sys.exit(f"No .npy files in {args.input_dir}")
    print(f"Density-adaptive binarize on {len(files)} tiles")
    print(f"  Regression coefs: {coefs}")
    print(f"  Global v/w: t_v={global_t[1]}, t_w={global_t[2]}; K_water={k_water}")
    print(f"  Clip range for t_b: [{clip_lo}, {clip_hi}]")

    tb_dist = []
    for i, pf in enumerate(files):
        a = np.load(pf).astype(np.float32)
        feat = np.array(features_bld(a[0]))
        tb = float(np.clip(feat @ coefs, clip_lo, clip_hi))
        tb_dist.append(tb)

        a[0] = (a[0] > tb).astype(np.float32)
        a[1] = (a[1] > global_t[1]).astype(np.float32)
        a[2] = (a[2] > global_t[2]).astype(np.float32)
        a[2] = remove_small_components(a[2], k_water)

        if args.max_height is not None:
            a[3] = np.minimum(a[3], args.max_height)
        np.save(os.path.join(args.output_dir, os.path.basename(pf)), a)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")

    tb_arr = np.array(tb_dist)
    print(f"\nPredicted t_b distribution: mean={tb_arr.mean():.3f} std={tb_arr.std():.3f} "
          f"range [{tb_arr.min():.3f}, {tb_arr.max():.3f}]")
    print(f"Wrote {sum(1 for f in os.listdir(args.output_dir) if f.endswith('.npy'))} binarized files.")


if __name__ == "__main__":
    main()
