"""Per-channel routing: assemble a final prediction by picking each output channel
from a DIFFERENT source model. The key insight: blending hurts when constituent
models have asymmetric strengths (e.g. model A has better heights, model B has
better building IoU). Routing keeps each channel's best source.

Output channels (4 total):
  0 = building presence
  1 = vegetation presence
  2 = water presence
  3 = height

Usage:
  python tools/per_channel_router.py \
    --bld-src DIR_for_building \
    --veg-src DIR_for_vegetation \
    --wat-src DIR_for_water \
    --hgt-src DIR_for_height \
    --output-dir OUT
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bld-src", required=True, help="Source dir for building presence (ch 0)")
    ap.add_argument("--veg-src", required=True, help="Source dir for veg presence (ch 1)")
    ap.add_argument("--wat-src", required=True, help="Source dir for water presence (ch 2)")
    ap.add_argument("--hgt-src", required=True, help="Source dir for height (ch 3)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    sources = {0: Path(args.bld_src), 1: Path(args.veg_src), 2: Path(args.wat_src), 3: Path(args.hgt_src)}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use building source as anchor for file list
    files = sorted(sources[0].glob("*.npy"))
    print(f"Routing {len(files)} tiles → {out_dir}")
    print(f"  bld←{sources[0]}, veg←{sources[1]}, wat←{sources[2]}, hgt←{sources[3]}")
    for i, f in enumerate(files):
        per_channel = {}
        for ch, src in sources.items():
            p = src / f.name
            if not p.exists():
                continue
            per_channel[ch] = np.load(p)
        if len(per_channel) != 4:
            continue
        # Align shapes
        h = min(arr.shape[1] for arr in per_channel.values())
        w = min(arr.shape[2] for arr in per_channel.values())
        out = np.zeros((4, h, w), dtype=np.float32)
        for ch in range(4):
            out[ch] = per_channel[ch][ch, :h, :w]
        np.save(out_dir / f.name, out)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"Wrote {len(files)} routed preds")


if __name__ == "__main__":
    main()
