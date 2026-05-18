"""Average N seed predictions and score on fold0 val.

Each seed has a `predictions/` dir with (4, H, W) npy files. We average per-tile
across seeds and run a threshold sweep + K_water filter to find the ensemble's
best fold0 val score.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import rasterio

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from core.metrics import build_label_map  # noqa: E402

LABELS_DIR = Path("/projects/bcrm/emb2height/data/train/labels")


def score(iou_b, iou_v, iou_w, rmse_b, rmse_v):
    return 0.25 * iou_b + 0.15 * iou_v + 0.15 * iou_w + 0.25 * max(0, 1 - rmse_b / 3) + 0.20 * max(0, 1 - rmse_v / 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dirs", nargs="+", required=True,
                    help="List of experiment dirs (each must have predictions/)")
    ap.add_argument("--split-file", default=str(SCRIPT_DIR / "splits" / "group_code_5fold_seed42" / "fold_0" / "split.json"))
    ap.add_argument("--thresholds-grid", default="0.30,0.95,0.025",
                    help="start,stop,step for threshold sweep")
    ap.add_argument("--water-k-list", default="0,8,12,14")
    args = ap.parse_args()

    with open(args.split_file) as f:
        val_ids = json.load(f)["val"]
    label_map = build_label_map(str(LABELS_DIR))
    pred_dirs = [Path(d) / "predictions" for d in args.exp_dirs]
    print(f"Ensembling {len(pred_dirs)} seeds on {len(val_ids)} val tiles")
    for d in pred_dirs:
        if not d.exists():
            raise FileNotFoundError(d)

    # Threshold grid
    a, b, s = (float(x) for x in args.thresholds_grid.split(","))
    thr_grid = np.arange(a, b + 1e-9, s)
    k_list = [int(x) for x in args.water_k_list.split(",")]

    # Aggregate inter/union per (threshold, channel) and heights
    # Average predictions per tile, then sweep
    from collections import defaultdict
    from scipy.ndimage import label as cclabel

    # Compute per-class iou at each threshold across all val tiles
    agg = {
        "bld": {round(float(t), 3): {"inter": 0, "union": 0} for t in thr_grid},
        "veg": {round(float(t), 3): {"inter": 0, "union": 0} for t in thr_grid},
        "wat": {(round(float(t), 3), k): {"inter": 0, "union": 0}
                for t in thr_grid for k in k_list},
    }
    bH_sse = 0.0; bH_n = 0
    vH_sse = 0.0; vH_n = 0
    n_processed = 0

    for tid in val_ids:
        files = [pd / f"{tid}.npy" for pd in pred_dirs]
        if not all(f.exists() for f in files):
            continue
        if tid not in label_map:
            continue
        # Average
        preds = [np.load(f) for f in files]
        mean_pred = np.mean(preds, axis=0)
        with rasterio.open(label_map[tid]) as src:
            lab = src.read().astype(np.float32)
        h = min(mean_pred.shape[1], lab.shape[1])
        w = min(mean_pred.shape[2], lab.shape[2])
        mean_pred = mean_pred[:, :h, :w]
        lab = lab[:, :h, :w]
        n_processed += 1
        # IoU for bld/veg (no K filter)
        for ci, key in [(0, "bld"), (1, "veg")]:
            pp = mean_pred[ci]
            lb = (lab[ci] > 0).astype(np.uint8)
            for t in thr_grid:
                pb = (pp >= t).astype(np.uint8)
                tk = round(float(t), 3)
                agg[key][tk]["inter"] += int((pb & lb).sum())
                agg[key][tk]["union"] += int((pb | lb).sum())
        # Water with K filters
        pp = mean_pred[2]
        lb = (lab[2] > 0).astype(np.uint8)
        for t in thr_grid:
            pb_raw = (pp >= t).astype(np.uint8)
            for k in k_list:
                if k > 0:
                    lbl, _ = cclabel(pb_raw)
                    sizes = np.bincount(lbl.ravel())
                    sizes[0] = 0
                    pb = (sizes >= k)[lbl].astype(np.uint8)
                else:
                    pb = pb_raw
                tk = round(float(t), 3)
                agg["wat"][(tk, k)]["inter"] += int((pb & lb).sum())
                agg["wat"][(tk, k)]["union"] += int((pb | lb).sum())
        # Heights
        bpos = lab[0] > 0
        if bpos.sum() > 0:
            diff = (mean_pred[3] - lab[3])[bpos]
            bH_sse += float((diff**2).sum()); bH_n += int(bpos.sum())
        vpos = lab[1] > 0
        if vpos.sum() > 0:
            diff = (mean_pred[3] - lab[3])[vpos]
            vH_sse += float((diff**2).sum()); vH_n += int(vpos.sum())

    print(f"Processed: {n_processed} tiles")
    rmse_b = float(np.sqrt(bH_sse / max(bH_n, 1)))
    rmse_v = float(np.sqrt(vH_sse / max(vH_n, 1)))
    print(f"Heights: RMSE_bH={rmse_b:.4f}, RMSE_vH={rmse_v:.4f}")

    # Find best per-class threshold
    def best_iou(key_dict):
        best_t = max(key_dict, key=lambda t: key_dict[t]["inter"] / max(key_dict[t]["union"], 1))
        best_iou = key_dict[best_t]["inter"] / max(key_dict[best_t]["union"], 1)
        return best_t, best_iou
    bld_t, bld_iou = best_iou(agg["bld"])
    veg_t, veg_iou = best_iou(agg["veg"])
    # Water has both thr and K
    wat_best = max(agg["wat"], key=lambda tk: agg["wat"][tk]["inter"] / max(agg["wat"][tk]["union"], 1))
    wat_iou = agg["wat"][wat_best]["inter"] / max(agg["wat"][wat_best]["union"], 1)

    final_score = score(bld_iou, veg_iou, wat_iou, rmse_b, rmse_v)
    print(f"\n=== ENSEMBLE BEST ===")
    print(f"  iou_bld @ t={bld_t}: {bld_iou:.4f}")
    print(f"  iou_veg @ t={veg_t}: {veg_iou:.4f}")
    print(f"  iou_wat @ (t,K)={wat_best}: {wat_iou:.4f}")
    print(f"  RMSE_bH = {rmse_b:.4f}")
    print(f"  RMSE_vH = {rmse_v:.4f}")
    print(f"  Final score = {final_score:.4f}")


if __name__ == "__main__":
    main()
