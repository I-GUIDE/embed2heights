"""Per-area-code threshold sweep across all 5 folds.

For each location code that appears in ANY fold's val set, find the threshold
that maximizes iou_bld (or any class) AVERAGED across folds. The averaging
prevents fold0 (KE-heavy) from dominating threshold selection.

Output: per-location optimal thresholds for {bld, veg, wat}, plus the cross-
fold-averaged score using these thresholds.

The thresholds can then be applied to the submit_e100_5fold ensemble's test
predictions, splitting the 946 test tiles by location code and binarizing
each location with its own threshold.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from core.metrics import build_label_map  # noqa: E402

LABELS_DIR = Path("/projects/bcrm/emb2height/data/train/labels")


def loc_of(tid):
    parts = tid.split("_")
    return parts[1] if len(parts) >= 2 else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-template", default="exp_tess128_e100_fold{f}")
    ap.add_argument("--splits-template",
                    default=str(SCRIPT_DIR / "splits" / "group_code_5fold_seed42" / "fold_{f}" / "split.json"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--grid-start", type=float, default=0.30)
    ap.add_argument("--grid-stop", type=float, default=0.95)
    ap.add_argument("--grid-step", type=float, default=0.025)
    ap.add_argument("--output", default=str(SCRIPT_DIR / "runs" / "diagnostics" / "per_area_code_thresholds.json"))
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    thr_grid = np.arange(args.grid_start, args.grid_stop + 1e-9, args.grid_step)
    label_map = build_label_map(str(LABELS_DIR))

    # Per-location-and-fold inter/union accumulators per threshold
    # locfold_iou[loc][fold][class][t] = {inter, union}
    locfold_iou = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"inter": 0, "union": 0}))))
    loc_tile_count = defaultdict(int)
    loc_fold_tiles = defaultdict(lambda: defaultdict(int))

    for f in folds:
        exp = args.exp_template.format(f=f)
        pred_dir = SCRIPT_DIR / "runs" / exp / "predictions"
        split_path = Path(args.splits_template.format(f=f))
        with open(split_path) as sh:
            val_ids = json.load(sh)["val"]
        print(f"fold{f}: pred_dir={pred_dir}, n_val={len(val_ids)}")
        for tid in val_ids:
            pf = pred_dir / f"{tid}.npy"
            if not pf.exists() or tid not in label_map:
                continue
            pred = np.load(pf)
            with rasterio.open(label_map[tid]) as src:
                lab = src.read().astype(np.float32)
            h = min(pred.shape[1], lab.shape[1])
            w = min(pred.shape[2], lab.shape[2])
            pred = pred[:, :h, :w]
            lab = lab[:, :h, :w]
            loc = loc_of(tid)
            loc_tile_count[loc] += 1
            loc_fold_tiles[loc][f] += 1
            for ci, key in [(0, "bld"), (1, "veg"), (2, "wat")]:
                pp = pred[ci]
                lb = (lab[ci] > 0).astype(np.uint8)
                for t in thr_grid:
                    pb = (pp >= t).astype(np.uint8)
                    tk = float(round(t, 3))
                    locfold_iou[loc][f][key][tk]["inter"] += int((pb & lb).sum())
                    locfold_iou[loc][f][key][tk]["union"] += int((pb | lb).sum())

    print(f"\nTotal locations: {len(loc_tile_count)}")
    print(f"{'loc':>5} {'n':>5} {'fold_b':>15} {'best_bt':>9} {'best_b_iou':>11} {'best_vt':>9} {'best_v_iou':>11} {'best_wt':>9} {'best_w_iou':>11}")
    print("-" * 90)

    per_loc = {}
    for loc in sorted(loc_tile_count.keys(), key=lambda x: -loc_tile_count[x]):
        results_per_class = {}
        for key in ("bld", "veg", "wat"):
            # For each threshold t, compute cross-fold-averaged IoU for this location:
            # iou_t = sum_inter_over_folds / sum_union_over_folds (correct aggregation)
            sweep_iou = {}
            for t in thr_grid:
                tk = float(round(t, 3))
                total_inter = 0
                total_union = 0
                for f in folds:
                    if f in locfold_iou[loc]:
                        total_inter += locfold_iou[loc][f][key][tk]["inter"]
                        total_union += locfold_iou[loc][f][key][tk]["union"]
                sweep_iou[f"{tk:.3f}"] = total_inter / max(total_union, 1)
            best_t = max(sweep_iou, key=sweep_iou.get)
            results_per_class[key] = {
                "best_threshold": float(best_t),
                "best_iou": sweep_iou[best_t],
                "sweep": sweep_iou,
            }
        fold_counts = ",".join(f"{f}={loc_fold_tiles[loc][f]}" for f in folds)
        b = results_per_class["bld"]
        v = results_per_class["veg"]
        w = results_per_class["wat"]
        print(f"{loc:>5} {loc_tile_count[loc]:>5} {fold_counts:>15} {b['best_threshold']:>9.3f} {b['best_iou']:>11.4f} "
              f"{v['best_threshold']:>9.3f} {v['best_iou']:>11.4f} {w['best_threshold']:>9.3f} {w['best_iou']:>11.4f}")
        per_loc[loc] = {
            "n_tiles": loc_tile_count[loc],
            "fold_counts": dict(loc_fold_tiles[loc]),
            "thresholds": {k: results_per_class[k]["best_threshold"] for k in ("bld", "veg", "wat")},
            "ious": {k: results_per_class[k]["best_iou"] for k in ("bld", "veg", "wat")},
            "sweep": {k: results_per_class[k]["sweep"] for k in ("bld", "veg", "wat")},
        }

    # Also compute global per-class best threshold (no location split)
    global_iou = {key: {} for key in ("bld", "veg", "wat")}
    for key in ("bld", "veg", "wat"):
        for t in thr_grid:
            tk = float(round(t, 3))
            total_inter = sum(locfold_iou[loc][f][key][tk]["inter"]
                              for loc in locfold_iou
                              for f in folds if f in locfold_iou[loc])
            total_union = sum(locfold_iou[loc][f][key][tk]["union"]
                              for loc in locfold_iou
                              for f in folds if f in locfold_iou[loc])
            global_iou[key][f"{tk:.3f}"] = total_inter / max(total_union, 1)

    print("\n=== Global cross-fold best thresholds (no location split) ===")
    for key in ("bld", "veg", "wat"):
        best_t = max(global_iou[key], key=global_iou[key].get)
        print(f"  {key}: best_t={best_t} → iou={global_iou[key][best_t]:.4f}")

    out = {
        "per_location": per_loc,
        "global_thresholds": {key: max(global_iou[key], key=global_iou[key].get) for key in ("bld", "veg", "wat")},
        "global_ious": {key: max(global_iou[key].values()) for key in ("bld", "veg", "wat")},
        "global_sweep": global_iou,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
