"""Binarize predictions with per-region (tile suffix) thresholds.

Reads a JSON spec (e.g. runs/per_region_oof_thresholds.json) and applies
region-specific (t_b, t_v, t_w) triples to each tile. Falls back to a global
default for regions not in the spec (e.g. test-set regions never seen in OOF).

Use this AFTER training, for binarizing test predictions. Pairs naturally
with tools/binarize_postprocess.py — accepts the same K-filter args.

Spec JSON format (matches the per-region OOF output):
  {
    "global_thresholds": [0.640, 0.570, 0.820],
    "per_region_thresholds": {"KE": [0.34, 0.56, 0.69], ...},
    "k_water": 12,
    ...
  }
"""
import argparse
import glob
import json
import os
import re
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.inference.calibration import apply_water_cc_filter  # noqa: E402


SUFFIX_RE = re.compile(r"_([A-Z]+)(?:_\d{4})?$")


def get_region(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    for pre in ("label_", "gee_emb_", "tessera_emb_", "tokens_", "emb_", "s2_", "s1_"):
        if base.startswith(pre):
            base = base[len(pre):]
    base = re.sub(r"_\d{4}$", "", base)
    m = SUFFIX_RE.search(base)
    return m.group(1) if m else "??"


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
    p.add_argument("--input-dir", required=True,
                   help="Directory of [4,H,W] .npy probability predictions.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--spec", required=True,
                   help="JSON spec file produced by sweep_thresholds_per_region.py or run_per_region_oof.bash.")
    p.add_argument("--global-fallback", nargs=3, type=float, default=None,
                   metavar=("B", "V", "W"),
                   help="Override the global thresholds in the spec (e.g. for testing).")
    p.add_argument("--min-water-size", type=int, default=None,
                   help="Override K-water from spec.")
    p.add_argument("--min-building-size", type=int, default=0)
    p.add_argument("--min-veg-size", type=int, default=0)
    p.add_argument("--max-height", type=float, default=None)
    p.add_argument("--expected-count", type=int, default=946)
    args = p.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    global_t = tuple(args.global_fallback or spec["global_thresholds"])
    per_region = {r: tuple(t) for r, t in spec["per_region_thresholds"].items()}
    k_water = args.min_water_size if args.min_water_size is not None else int(spec.get("k_water", 0))

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not files:
        sys.exit(f"No .npy files in {args.input_dir}")
    print(f"Binarizing {len(files)} files with per-region thresholds.")
    print(f"  Global fallback: t={global_t}, K_water={k_water}")
    print(f"  Regions with custom thresholds: {len(per_region)}")
    print(f"  in:  {args.input_dir}")
    print(f"  out: {args.output_dir}")

    region_counts = {}
    fallback_count = 0
    for i, pf in enumerate(files):
        a = np.load(pf).astype(np.float32)
        reg = get_region(pf)
        t = per_region.get(reg, global_t)
        if reg not in per_region:
            fallback_count += 1
        region_counts[reg] = region_counts.get(reg, 0) + 1

        a[0] = (a[0] > t[0]).astype(np.float32)
        a[1] = (a[1] > t[1]).astype(np.float32)
        a[2] = (a[2] > t[2]).astype(np.float32)

        a[0] = remove_small_components(a[0], args.min_building_size)
        a[1] = remove_small_components(a[1], args.min_veg_size)
        a[2] = remove_small_components(a[2], k_water if k_water > 0 else args.min_water_size or 0)

        if args.max_height is not None:
            a[3] = np.minimum(a[3], args.max_height)
        np.save(os.path.join(args.output_dir, os.path.basename(pf)), a)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")

    print(f"\nRegion coverage:")
    for reg, n in sorted(region_counts.items(), key=lambda x: -x[1])[:20]:
        custom = "custom" if reg in per_region else "FALLBACK"
        print(f"  {reg}: {n} tiles ({custom})")
    print(f"\nFallback (no region-specific threshold) tiles: {fallback_count}/{len(files)}")
    print(f"Wrote {sum(1 for f in os.listdir(args.output_dir) if f.endswith('.npy'))} binarized files.")


if __name__ == "__main__":
    main()
