"""Merge two-stage predictions: segmentation (ch0-2) from the Stage-1 model,
height (ch3) from the Stage-2 'purify' model.

Both prediction dirs hold per-tile .npy arrays shaped (4, H, W) =
[building%, veg%, water%, height_m]. Stage 1 supplies the presence channels
(its backbone was trained coupled); Stage 2 supplies height (its backbone was
re-tuned for height only). READ-ONLY on inputs; writes merged .npy to --out-dir.
"""
import argparse
import glob
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg-dir", required=True, help="Stage-1 predictions (supplies ch0-2)")
    p.add_argument("--height-dir", required=True, help="Stage-2 predictions (supplies ch3)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--build-height-offset", type=float, default=0.0,
                   help="Meters added to ch3 (height) at pixels where building is the "
                        "dominant predicted class (argmax of ch0-2). Corrects a "
                        "systematic building-height bias. 0.0 (default) = no-op. NOTE: "
                        "fit this per-model with tools/bias_lever_analysis.py on the "
                        "run's own val split -- the optimum is model-specific and a "
                        "too-large offset HURTS the per-tile RMSE metric.")
    p.add_argument("--veg-height-offset", type=float, default=0.0,
                   help="Meters added to ch3 at veg-dominant pixels (argmax of ch0-2). "
                        "Vegetation height tends to be over-predicted, so this is "
                        "usually negative. 0.0 (default) = no-op. Same per-model fit "
                        "caveat as --build-height-offset.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seg_files = sorted(glob.glob(os.path.join(args.seg_dir, "*.npy")))
    if not seg_files:
        raise SystemExit(f"No .npy predictions in {args.seg_dir}")

    n, missing = 0, 0
    for sp in seg_files:
        name = os.path.basename(sp)
        hp = os.path.join(args.height_dir, name)
        if not os.path.exists(hp):
            missing += 1
            continue
        seg = np.load(sp)
        hgt = np.load(hp)
        if seg.shape != hgt.shape:
            raise SystemExit(f"shape mismatch {name}: {seg.shape} vs {hgt.shape}")
        merged = seg.copy()
        merged[3] = hgt[3]  # take height channel from the Stage-2 purify model
        if args.build_height_offset != 0.0 or args.veg_height_offset != 0.0:
            # Per-pixel dominant presence class (argmax over ch0-2): 0=building,
            # 1=veg, 2=water. Apply the class-specific height offset only there.
            arg = np.argmax(merged[:3], axis=0)
            if args.build_height_offset != 0.0:
                merged[3] = np.where(arg == 0, merged[3] + args.build_height_offset, merged[3])
            if args.veg_height_offset != 0.0:
                merged[3] = np.where(arg == 1, merged[3] + args.veg_height_offset, merged[3])
        np.save(os.path.join(args.out_dir, name), merged)
        n += 1

    notes = []
    if args.build_height_offset != 0.0:
        notes.append(f"+{args.build_height_offset}m bld")
    if args.veg_height_offset != 0.0:
        notes.append(f"{args.veg_height_offset:+}m veg")
    off_note = f"  [ch3 offsets: {', '.join(notes)}]" if notes else ""
    print(f"Merged {n} tiles -> {args.out_dir}  (ch0-2 from seg, ch3 from height){off_note}")
    if missing:
        print(f"WARNING: {missing} seg tiles had no matching height prediction (skipped).")


if __name__ == "__main__":
    main()
