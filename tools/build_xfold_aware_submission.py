"""Build a new submission from existing continuous test predictions using
cross-fold-aware thresholds.

Strategy A (recommended baseline): apply the GLOBAL cross-fold optima
(bld/veg/wat) uniformly to all 946 test tiles.

Strategy B: apply PER-AREA-CODE thresholds (from per_area_code_thresholds.json)
to each test tile, looked up by its location code, falling back to global when
the location is absent or has <2 tiles total.
"""
import argparse, json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]


def loc_of(name):
    parts = Path(name).stem.split("_")
    return parts[1] if len(parts) >= 2 else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True,
                    help="Continuous .npy predictions directory (e.g. runs/submission/preds_submit_e100_5fold)")
    ap.add_argument("--out-dir", required=True,
                    help="Output dir for binarized .npy submission files")
    ap.add_argument("--strategy", choices=["global", "per_area"], default="global")
    ap.add_argument("--bld-t", type=float, default=0.525)
    ap.add_argument("--veg-t", type=float, default=0.600)
    ap.add_argument("--wat-t", type=float, default=0.475)
    ap.add_argument("--per-area-json", default=None,
                    help="Required when --strategy per_area")
    ap.add_argument("--min-tiles-for-loc", type=int, default=2,
                    help="If a location has fewer total tiles in the diagnostic, fall back to global thresholds")
    ap.add_argument("--water-k", type=int, default=0,
                    help="Optional water connected-component filter size (0 = off)")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_t = {"bld": args.bld_t, "veg": args.veg_t, "wat": args.wat_t}

    per_area = {}
    if args.strategy == "per_area":
        with open(args.per_area_json) as f:
            data = json.load(f)
        for loc, r in data["per_location"].items():
            if r["n_tiles"] >= args.min_tiles_for_loc:
                per_area[loc] = r["thresholds"]
        print(f"Loaded per-area thresholds for {len(per_area)} locations "
              f"(min_tiles_for_loc={args.min_tiles_for_loc}); global fallback applies to others.")

    files = sorted(pred_dir.glob("*.npy"))
    print(f"Binarizing {len(files)} files...")

    optional_K = args.water_k
    n_loc_specific = 0
    n_global = 0
    for i, f in enumerate(files):
        pred = np.load(f)  # [4, H, W]
        loc = loc_of(f.name)
        if args.strategy == "per_area" and loc in per_area:
            t = per_area[loc]
            n_loc_specific += 1
        else:
            t = global_t
            n_global += 1

        bld_bin = (pred[0] >= t["bld"]).astype(np.uint8)
        veg_bin = (pred[1] >= t["veg"]).astype(np.uint8)
        wat_bin = (pred[2] >= t["wat"]).astype(np.uint8)

        if optional_K > 0:
            # Filter water by min connected-component size
            from scipy.ndimage import label as cclabel
            lbl, n = cclabel(wat_bin)
            sizes = np.bincount(lbl.ravel())
            sizes[0] = 0
            keep = sizes >= optional_K
            wat_bin = keep[lbl].astype(np.uint8)

        height = pred[3].astype(np.float32)

        out = np.stack([bld_bin, veg_bin, wat_bin, height], axis=0).astype(np.float32)
        np.save(out_dir / f.name, out)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")

    print(f"Wrote {len(files)} binarized files to {out_dir}")
    if args.strategy == "per_area":
        print(f"  per-location thresholds: {n_loc_specific} tiles; global fallback: {n_global} tiles")


if __name__ == "__main__":
    main()
