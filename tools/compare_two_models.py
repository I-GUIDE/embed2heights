"""Side-by-side per-tile comparison of two models on the same val set.

Pulls per-tile metrics for each, computes deltas, and shows:
  - Tiles where model B is much better than A (biggest improvements)
  - Tiles where model B is much worse than A (regressions)
  - Per-location delta summary
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


def score(iou_b, iou_v, iou_w, rmse_b, rmse_v):
    return (0.25 * iou_b + 0.15 * iou_v + 0.15 * iou_w
            + 0.25 * max(0, 1 - rmse_b / 3) + 0.20 * max(0, 1 - rmse_v / 5))


def loc_of(tid):
    parts = tid.split("_")
    return parts[1] if len(parts) >= 2 else "?"


def tile_metrics(pred, lab, t_bld, t_veg, t_wat, water_k):
    from scipy.ndimage import label as cclabel
    h = min(pred.shape[1], lab.shape[1]); w = min(pred.shape[2], lab.shape[2])
    pred = pred[:, :h, :w]; lab = lab[:, :h, :w]
    bld_p = (pred[0] >= t_bld).astype(np.uint8)
    veg_p = (pred[1] >= t_veg).astype(np.uint8)
    wat_pr = (pred[2] >= t_wat).astype(np.uint8)
    if water_k > 0:
        lbl, _ = cclabel(wat_pr); sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        wat_p = (sizes >= water_k)[lbl].astype(np.uint8)
    else:
        wat_p = wat_pr
    bld_l = (lab[0] > 0).astype(np.uint8); veg_l = (lab[1] > 0).astype(np.uint8); wat_l = (lab[2] > 0).astype(np.uint8)
    def iou(p, l):
        u = int((p | l).sum())
        return 1.0 if u == 0 and l.sum() == 0 else (int((p & l).sum()) / max(u, 1))
    iou_b = iou(bld_p, bld_l); iou_v = iou(veg_p, veg_l); iou_w = iou(wat_p, wat_l)
    bpos = bld_l.astype(bool); rmse_b = float(np.sqrt(((pred[3] - lab[3])[bpos]**2).mean())) if bpos.sum() else 0.0
    vpos = veg_l.astype(bool); rmse_v = float(np.sqrt(((pred[3] - lab[3])[vpos]**2).mean())) if vpos.sum() else 0.0
    return iou_b, iou_v, iou_w, rmse_b, rmse_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-a", required=True, help="Baseline model exp name")
    ap.add_argument("--exp-b", required=True, help="Comparison model exp name")
    ap.add_argument("--split-file", default=str(SCRIPT_DIR / "splits" / "group_code_5fold_seed42" / "fold_0" / "split.json"))
    ap.add_argument("--bld-t", type=float, default=0.725)
    ap.add_argument("--veg-t", type=float, default=0.575)
    ap.add_argument("--wat-t", type=float, default=0.875)
    ap.add_argument("--water-k", type=int, default=0)
    ap.add_argument("--n", type=int, default=15, help="Show top-N tiles improved AND regressed")
    args = ap.parse_args()

    pdA = SCRIPT_DIR / "runs" / args.exp_a / "predictions"
    pdB = SCRIPT_DIR / "runs" / args.exp_b / "predictions"
    with open(args.split_file) as f:
        val_ids = json.load(f)["val"]
    label_map = build_label_map(str(LABELS_DIR))
    rows = []
    for tid in val_ids:
        fA = pdA / f"{tid}.npy"
        fB = pdB / f"{tid}.npy"
        if not (fA.exists() and fB.exists() and tid in label_map):
            continue
        predA = np.load(fA); predB = np.load(fB)
        with rasterio.open(label_map[tid]) as src:
            lab = src.read().astype(np.float32)
        mA = tile_metrics(predA, lab, args.bld_t, args.veg_t, args.wat_t, args.water_k)
        mB = tile_metrics(predB, lab, args.bld_t, args.veg_t, args.wat_t, args.water_k)
        sA = score(*mA); sB = score(*mB)
        rows.append({
            "tid": tid, "loc": loc_of(tid),
            "sA": sA, "sB": sB, "delta": sB - sA,
            "iou_b_A": mA[0], "iou_b_B": mB[0], "d_iou_b": mB[0] - mA[0],
            "iou_v_A": mA[1], "iou_v_B": mB[1], "d_iou_v": mB[1] - mA[1],
            "iou_w_A": mA[2], "iou_w_B": mB[2], "d_iou_w": mB[2] - mA[2],
            "rmse_bH_A": mA[3], "rmse_bH_B": mB[3], "d_rmse_bH": mB[3] - mA[3],
            "rmse_vH_A": mA[4], "rmse_vH_B": mB[4], "d_rmse_vH": mB[4] - mA[4],
        })

    rows.sort(key=lambda x: x["delta"])
    print(f"\n=== Compare A='{args.exp_a}' vs B='{args.exp_b}' ===")
    print(f"Thresholds bld={args.bld_t}, veg={args.veg_t}, wat={args.wat_t}, K={args.water_k}")
    print(f"Tiles compared: {len(rows)}")
    sA_all = np.mean([r["sA"] for r in rows]); sB_all = np.mean([r["sB"] for r in rows])
    print(f"Mean score A: {sA_all:.4f}")
    print(f"Mean score B: {sB_all:.4f}")
    print(f"Delta (B-A): {sB_all - sA_all:+.4f}")
    n_win = sum(1 for r in rows if r["delta"] > 0)
    print(f"B beats A on {n_win}/{len(rows)} tiles ({n_win/len(rows)*100:.1f}%)")

    print(f"\n--- BIGGEST B-WORSE-THAN-A (B regressed): top {args.n} ---")
    print(f"{'tid':<14} {'loc':<3} {'sA':>7} {'sB':>7} {'delta':>7} {'d_iou_b':>9} {'d_iou_v':>9} {'d_iou_w':>9} {'d_rmse_bH':>10} {'d_rmse_vH':>10}")
    for r in rows[:args.n]:
        print(f"{r['tid']:<14} {r['loc']:<3} {r['sA']:>7.4f} {r['sB']:>7.4f} {r['delta']:>+7.4f} {r['d_iou_b']:>+9.4f} {r['d_iou_v']:>+9.4f} {r['d_iou_w']:>+9.4f} {r['d_rmse_bH']:>+10.3f} {r['d_rmse_vH']:>+10.3f}")

    print(f"\n--- BIGGEST B-BETTER-THAN-A: top {args.n} ---")
    print(f"{'tid':<14} {'loc':<3} {'sA':>7} {'sB':>7} {'delta':>7} {'d_iou_b':>9} {'d_iou_v':>9} {'d_iou_w':>9} {'d_rmse_bH':>10} {'d_rmse_vH':>10}")
    for r in rows[-args.n:][::-1]:
        print(f"{r['tid']:<14} {r['loc']:<3} {r['sA']:>7.4f} {r['sB']:>7.4f} {r['delta']:>+7.4f} {r['d_iou_b']:>+9.4f} {r['d_iou_v']:>+9.4f} {r['d_iou_w']:>+9.4f} {r['d_rmse_bH']:>+10.3f} {r['d_rmse_vH']:>+10.3f}")

    # Per-location delta
    per_loc = defaultdict(list)
    for r in rows: per_loc[r["loc"]].append(r)
    print(f"\n=== Per-location delta (sorted by delta) ===")
    print(f"{'loc':<5} {'n':<4} {'sA':>7} {'sB':>7} {'delta':>7} {'d_iou_b':>9}")
    locs = sorted(per_loc.items(), key=lambda x: np.mean([r["delta"] for r in x[1]]))
    for loc, rs in locs:
        sA_l = np.mean([r["sA"] for r in rs]); sB_l = np.mean([r["sB"] for r in rs])
        d_b = np.mean([r["d_iou_b"] for r in rs])
        print(f"{loc:<5} {len(rs):<4} {sA_l:>7.4f} {sB_l:>7.4f} {sB_l - sA_l:>+7.4f} {d_b:>+9.4f}")


if __name__ == "__main__":
    main()
