"""Cross-fold robust threshold sweep.

For each fold i (0-4):
  - Load that fold's model's predictions on its OWN val tiles (continuous probs).
  - Aggregate across folds to compute total cross-fold iou_bld/iou_veg/iou_wat
    AT EACH threshold from the sweep grid.

This gives a CROSS-FOLD-AVERAGED score curve, NOT a fold0-overfitting one.
Useful for picking conservative global thresholds for submissions built from
similar single-seed-per-fold ensembles.
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from core.metrics import build_label_map  # noqa: E402

LABELS_DIR = Path("/projects/bcrm/emb2height/data/train/labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-template", default="exp_tess128_e100_fold{f}",
                    help="Python format string with {f} = fold index.")
    ap.add_argument("--splits-template",
                    default=str(SCRIPT_DIR / "splits" / "group_code_5fold_seed42" / "fold_{f}" / "split.json"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--grid-start", type=float, default=0.30)
    ap.add_argument("--grid-stop", type=float, default=0.95)
    ap.add_argument("--grid-step", type=float, default=0.025)
    ap.add_argument("--output", default=str(SCRIPT_DIR / "runs" / "diagnostics" / "cross_fold_threshold_sweep.json"))
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    thr_grid = np.arange(args.grid_start, args.grid_stop + 1e-9, args.grid_step)

    # Aggregate inter/union per threshold per class across all fold val sets
    agg = {
        "bld": {round(float(t), 3): {"inter": 0, "union": 0} for t in thr_grid},
        "veg": {round(float(t), 3): {"inter": 0, "union": 0} for t in thr_grid},
        "wat": {round(float(t), 3): {"inter": 0, "union": 0} for t in thr_grid},
    }
    # Heights: just use ALL pixels where the GT class is present for that fold (single threshold for now)
    height_sse_bH = 0.0
    height_n_bH = 0
    height_sse_vH = 0.0
    height_n_vH = 0
    n_tiles_total = 0

    label_map = build_label_map(str(LABELS_DIR))
    for f in folds:
        exp = args.exp_template.format(f=f)
        pred_dir = SCRIPT_DIR / "runs" / exp / "predictions"
        split_path = Path(args.splits_template.format(f=f))
        with open(split_path) as sh:
            val_ids = json.load(sh)["val"]
        print(f"fold{f}: pred_dir={pred_dir}, n_val={len(val_ids)}")
        for tid in val_ids:
            pf = pred_dir / f"{tid}.npy"
            if not pf.exists():
                continue
            if tid not in label_map:
                continue
            pred = np.load(pf)  # [4, H, W]
            with rasterio.open(label_map[tid]) as src:
                lab = src.read().astype(np.float32)
            # Align shapes: crop both to the common (min) HxW
            h = min(pred.shape[1], lab.shape[1])
            w = min(pred.shape[2], lab.shape[2])
            pred = pred[:, :h, :w]
            lab = lab[:, :h, :w]
            n_tiles_total += 1
            for ci, key in [(0, "bld"), (1, "veg"), (2, "wat")]:
                pp = pred[ci]
                lb = (lab[ci] > 0).astype(np.uint8)
                for t in thr_grid:
                    pb = (pp >= t).astype(np.uint8)
                    tk = round(float(t), 3)
                    agg[key][tk]["inter"] += int((pb & lb).sum())
                    agg[key][tk]["union"] += int((pb | lb).sum())
            # heights (no threshold sweep here — building height on building-positive pixels)
            bld_pos = lab[0] > 0
            if bld_pos.sum() > 0:
                diff = (pred[3] - lab[3])[bld_pos]
                height_sse_bH += float((diff ** 2).sum())
                height_n_bH += int(bld_pos.sum())
            veg_pos = lab[1] > 0
            if veg_pos.sum() > 0:
                diff = (pred[3] - lab[3])[veg_pos]
                height_sse_vH += float((diff ** 2).sum())
                height_n_vH += int(veg_pos.sum())

    print(f"\nTotal tiles processed: {n_tiles_total}")
    rmse_bH = float(np.sqrt(height_sse_bH / max(height_n_bH, 1)))
    rmse_vH = float(np.sqrt(height_sse_vH / max(height_n_vH, 1)))
    print(f"Cross-fold RMSE_bH = {rmse_bH:.4f}, RMSE_vH = {rmse_vH:.4f}")

    # Build sweep curves and find best per class
    sweep_iou = {}
    for key in ("bld", "veg", "wat"):
        sweep_iou[key] = {
            f"{t:.3f}": agg[key][t]["inter"] / max(agg[key][t]["union"], 1)
            for t in sorted(agg[key].keys())
        }
        best_t = max(sweep_iou[key], key=sweep_iou[key].get)
        print(f"\nClass {key}: cross-fold best threshold = {best_t} → iou = {sweep_iou[key][best_t]:.4f}")
        print(f"  (compared to t=0.5: iou = {sweep_iou[key].get('0.500', '?'):.4f}, "
              f"t=0.725: iou = {sweep_iou[key].get('0.725', '?')})")

    # Score at canonical thresholds (submitted)
    def score(iou_b, iou_v, iou_w, rmse_b, rmse_v):
        return 0.25 * iou_b + 0.15 * iou_v + 0.15 * iou_w + 0.25 * (1 - rmse_b / 3) + 0.20 * (1 - rmse_v / 5)

    # Best per-class threshold combination
    best_b_t = max(sweep_iou["bld"], key=sweep_iou["bld"].get)
    best_v_t = max(sweep_iou["veg"], key=sweep_iou["veg"].get)
    best_w_t = max(sweep_iou["wat"], key=sweep_iou["wat"].get)
    best_score = score(sweep_iou["bld"][best_b_t], sweep_iou["veg"][best_v_t],
                       sweep_iou["wat"][best_w_t], rmse_bH, rmse_vH)
    print(f"\nBest per-class cross-fold thresholds: bld={best_b_t}, veg={best_v_t}, wat={best_w_t}")
    print(f"  → cross-fold score (no K_water filter): {best_score:.4f}")

    # Score at the SUBMITTED thresholds for submit_e100_5fold_bin
    sub_b, sub_v, sub_w = "0.725", "0.575", "0.875"
    sub_score = score(sweep_iou["bld"][sub_b], sweep_iou["veg"][sub_v],
                      sweep_iou["wat"][sub_w], rmse_bH, rmse_vH)
    print(f"At submitted thresholds (0.725/0.575/0.875): cross-fold score = {sub_score:.4f}")

    out = {
        "n_tiles": n_tiles_total,
        "rmse_bH": rmse_bH,
        "rmse_vH": rmse_vH,
        "sweep_iou_bld": sweep_iou["bld"],
        "sweep_iou_veg": sweep_iou["veg"],
        "sweep_iou_wat": sweep_iou["wat"],
        "best_thresholds": {"bld": best_b_t, "veg": best_v_t, "wat": best_w_t},
        "best_cross_fold_score": best_score,
        "submitted_thresholds": {"bld": sub_b, "veg": sub_v, "wat": sub_w},
        "submitted_cross_fold_score": sub_score,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
