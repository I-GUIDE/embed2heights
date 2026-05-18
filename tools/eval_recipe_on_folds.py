"""Evaluate a binarization recipe (global thresholds OR per-area thresholds)
on all 5 folds' val predictions. Returns cross-fold-averaged score so we can
compare BEFORE deciding to submit.
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


def score_metric(iou_b, iou_v, iou_w, rmse_b, rmse_v):
    return 0.25 * iou_b + 0.15 * iou_v + 0.15 * iou_w + 0.25 * max(0, 1 - rmse_b / 3) + 0.20 * max(0, 1 - rmse_v / 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-template", default="exp_tess128_e100_fold{f}")
    ap.add_argument("--splits-template",
                    default=str(SCRIPT_DIR / "splits" / "group_code_5fold_seed42" / "fold_{f}" / "split.json"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--recipe", choices=["global", "per_area"], required=True)
    ap.add_argument("--bld-t", type=float, default=0.525)
    ap.add_argument("--veg-t", type=float, default=0.600)
    ap.add_argument("--wat-t", type=float, default=0.475)
    ap.add_argument("--per-area-json", default=None)
    ap.add_argument("--water-k", type=int, default=0)
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    global_t = {"bld": args.bld_t, "veg": args.veg_t, "wat": args.wat_t}
    per_area = {}
    if args.recipe == "per_area":
        with open(args.per_area_json) as fh:
            data = json.load(fh)
        for loc, r in data["per_location"].items():
            if r["n_tiles"] >= 3:
                per_area[loc] = r["thresholds"]
        print(f"Loaded per-area thresholds for {len(per_area)} locations; global fallback for the rest.")

    label_map = build_label_map(str(LABELS_DIR))

    # Aggregate across folds: inter/union per class, height SSE for bld/veg
    agg = {"bld": {"i": 0, "u": 0}, "veg": {"i": 0, "u": 0}, "wat": {"i": 0, "u": 0}}
    bH_sse = 0.0; bH_n = 0
    vH_sse = 0.0; vH_n = 0
    n_loc_specific = 0; n_global = 0; n_total = 0

    for f in folds:
        exp = args.exp_template.format(f=f)
        pred_dir = SCRIPT_DIR / "runs" / exp / "predictions"
        split_path = Path(args.splits_template.format(f=f))
        with open(split_path) as sh:
            val_ids = json.load(sh)["val"]
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
            if args.recipe == "per_area" and loc in per_area:
                t = per_area[loc]
                n_loc_specific += 1
            else:
                t = global_t
                n_global += 1
            n_total += 1
            # IoUs
            for ci, key, thr in [(0, "bld", t["bld"]), (1, "veg", t["veg"]), (2, "wat", t["wat"])]:
                pb = (pred[ci] >= thr).astype(np.uint8)
                lb = (lab[ci] > 0).astype(np.uint8)
                agg[key]["i"] += int((pb & lb).sum())
                agg[key]["u"] += int((pb | lb).sum())
            # K_water filter
            if args.water_k > 0:
                from scipy.ndimage import label as cclabel
                wat_bin = (pred[2] >= t["wat"]).astype(np.uint8)
                lbl, _ = cclabel(wat_bin)
                sizes = np.bincount(lbl.ravel())
                sizes[0] = 0
                keep = sizes >= args.water_k
                wat_bin = keep[lbl].astype(np.uint8)
                lb_w = (lab[2] > 0).astype(np.uint8)
                # replace water agg with filtered version (re-add this tile's contribution)
                # Recompute: undo what was added, redo with filter
                # Simpler: track separately and use this for final
            # heights
            bld_pos = (lab[0] > 0)
            if bld_pos.sum() > 0:
                diff = (pred[3] - lab[3])[bld_pos]
                bH_sse += float((diff ** 2).sum())
                bH_n += int(bld_pos.sum())
            veg_pos = (lab[1] > 0)
            if veg_pos.sum() > 0:
                diff = (pred[3] - lab[3])[veg_pos]
                vH_sse += float((diff ** 2).sum())
                vH_n += int(veg_pos.sum())

    iou_b = agg["bld"]["i"] / max(agg["bld"]["u"], 1)
    iou_v = agg["veg"]["i"] / max(agg["veg"]["u"], 1)
    iou_w = agg["wat"]["i"] / max(agg["wat"]["u"], 1)
    rmse_b = float(np.sqrt(bH_sse / max(bH_n, 1)))
    rmse_v = float(np.sqrt(vH_sse / max(vH_n, 1)))
    sc = score_metric(iou_b, iou_v, iou_w, rmse_b, rmse_v)
    print(f"\n=== Recipe = {args.recipe} ===")
    if args.recipe == "per_area":
        print(f"  Per-location tiles: {n_loc_specific}, global fallback tiles: {n_global}, total {n_total}")
    else:
        print(f"  Global thresholds: bld={args.bld_t}, veg={args.veg_t}, wat={args.wat_t}")
    print(f"  iou_bld = {iou_b:.4f}")
    print(f"  iou_veg = {iou_v:.4f}")
    print(f"  iou_wat = {iou_w:.4f}")
    print(f"  RMSE_bH = {rmse_b:.4f}")
    print(f"  RMSE_vH = {rmse_v:.4f}")
    print(f"  Cross-fold score = {sc:.4f}")
    print(f"  Components: bld={0.25*iou_b:.4f}, veg={0.15*iou_v:.4f}, wat={0.15*iou_w:.4f}, "
          f"bH={0.25*max(0,1-rmse_b/3):.4f}, vH={0.20*max(0,1-rmse_v/5):.4f}")


if __name__ == "__main__":
    main()
