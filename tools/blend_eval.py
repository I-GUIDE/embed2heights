"""Evaluate heterogeneous ensemble blends on fold0 val.

For each (primary, secondary, mix_weight) triple, average preds:
    blended = w * primary + (1 - w) * secondary
Then score on fold0 val using per-class + K=8/12 sweep.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import rasterio

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from core.metrics import build_label_map  # noqa: E402

LABELS_DIR = Path("/projects/bcrm/emb2height/data/train/labels")


def score_metric(iou_b, iou_v, iou_w, rmse_b, rmse_v):
    return 0.25 * iou_b + 0.15 * iou_v + 0.15 * iou_w + 0.25 * max(0, 1 - rmse_b / 3) + 0.20 * max(0, 1 - rmse_v / 5)


def eval_predictions(blended_preds_dir_or_dict, val_ids, label_map, thresholds=(0.575, 0.575, 0.825), water_k=8):
    """Score a blended prediction set on val_ids."""
    agg = {"bld": {"i": 0, "u": 0}, "veg": {"i": 0, "u": 0}, "wat": {"i": 0, "u": 0}}
    bH_sse = 0.0; bH_n = 0; vH_sse = 0.0; vH_n = 0
    bt, vt, wt = thresholds
    from scipy.ndimage import label as cclabel

    for tid in val_ids:
        if tid not in blended_preds_dir_or_dict:
            continue
        if tid not in label_map:
            continue
        pred = blended_preds_dir_or_dict[tid]
        with rasterio.open(label_map[tid]) as src:
            lab = src.read().astype(np.float32)
        h = min(pred.shape[1], lab.shape[1]); w = min(pred.shape[2], lab.shape[2])
        pred = pred[:, :h, :w]; lab = lab[:, :h, :w]
        bld_b = (pred[0] >= bt).astype(np.uint8); veg_b = (pred[1] >= vt).astype(np.uint8); wat_b = (pred[2] >= wt).astype(np.uint8)
        if water_k > 0:
            lbl, _ = cclabel(wat_b); sizes = np.bincount(lbl.ravel()); sizes[0] = 0
            wat_b = (sizes >= water_k)[lbl].astype(np.uint8)
        for ci, key, pb in [(0, "bld", bld_b), (1, "veg", veg_b), (2, "wat", wat_b)]:
            lb = (lab[ci] > 0).astype(np.uint8)
            agg[key]["i"] += int((pb & lb).sum()); agg[key]["u"] += int((pb | lb).sum())
        bpos = lab[0] > 0
        if bpos.sum() > 0:
            diff = (pred[3] - lab[3])[bpos]; bH_sse += float((diff**2).sum()); bH_n += int(bpos.sum())
        vpos = lab[1] > 0
        if vpos.sum() > 0:
            diff = (pred[3] - lab[3])[vpos]; vH_sse += float((diff**2).sum()); vH_n += int(vpos.sum())

    iou_b = agg["bld"]["i"] / max(agg["bld"]["u"], 1)
    iou_v = agg["veg"]["i"] / max(agg["veg"]["u"], 1)
    iou_w = agg["wat"]["i"] / max(agg["wat"]["u"], 1)
    rmse_b = float(np.sqrt(bH_sse / max(bH_n, 1)))
    rmse_v = float(np.sqrt(vH_sse / max(vH_n, 1)))
    return iou_b, iou_v, iou_w, rmse_b, rmse_v, score_metric(iou_b, iou_v, iou_w, rmse_b, rmse_v)


def load_preds(pred_dir):
    """Returns dict tid -> np array [4, H, W]"""
    preds = {}
    for f in Path(pred_dir).glob("*.npy"):
        tid = f.stem
        preds[tid] = np.load(f)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="Primary predictions dir")
    ap.add_argument("--secondary-list", required=True, help="comma-separated list of pred dirs")
    ap.add_argument("--mix-weights", default="1.0,0.8,0.7,0.6,0.5", help="comma-separated primary weights")
    ap.add_argument("--split-file", default=str(SCRIPT_DIR / "splits" / "group_code_5fold_seed42" / "fold_0" / "split.json"))
    ap.add_argument("--thresholds", default="0.575,0.575,0.825")
    ap.add_argument("--water-k", type=int, default=8)
    args = ap.parse_args()

    with open(args.split_file) as f:
        val_ids = json.load(f)["val"]
    label_map = build_label_map(str(LABELS_DIR))
    thresholds = tuple(float(x) for x in args.thresholds.split(","))
    mix_weights = [float(x) for x in args.mix_weights.split(",")]

    print(f"Loading primary: {args.primary}")
    primary = load_preds(args.primary)
    print(f"  loaded {len(primary)} preds")
    print(f"thresholds={thresholds}, K_water={args.water_k}")

    # Baseline: primary alone (mix_weight=1.0)
    print(f"\n=== Primary alone (w=1.0) ===")
    iou_b, iou_v, iou_w, rmse_b, rmse_v, score = eval_predictions(primary, val_ids, label_map, thresholds, args.water_k)
    print(f"  iou_bld={iou_b:.4f}, iou_veg={iou_v:.4f}, iou_wat={iou_w:.4f}, RMSE_bH={rmse_b:.4f}, RMSE_vH={rmse_v:.4f}, score={score:.4f}")

    sec_dirs = args.secondary_list.split(",")
    for sec_dir in sec_dirs:
        sec_name = Path(sec_dir).name
        print(f"\n=== Secondary: {sec_name} ===")
        sec = load_preds(sec_dir)
        # First eval secondary alone
        iou_b, iou_v, iou_w, rmse_b, rmse_v, score = eval_predictions(sec, val_ids, label_map, thresholds, args.water_k)
        print(f"  alone (w_pri=0.0): iou_bld={iou_b:.4f}, iou_veg={iou_v:.4f}, iou_wat={iou_w:.4f}, RMSE_bH={rmse_b:.4f}, RMSE_vH={rmse_v:.4f}, score={score:.4f}")
        # Then various blends
        for w in mix_weights:
            if w == 1.0 or w == 0.0:
                continue
            blended = {}
            for tid in primary:
                if tid not in sec:
                    blended[tid] = primary[tid]
                    continue
                p = primary[tid]; s = sec[tid]
                h = min(p.shape[1], s.shape[1]); w_dim = min(p.shape[2], s.shape[2])
                blended[tid] = (w * p[:, :h, :w_dim] + (1.0 - w) * s[:, :h, :w_dim])
            iou_b, iou_v, iou_w, rmse_b, rmse_v, score = eval_predictions(blended, val_ids, label_map, thresholds, args.water_k)
            print(f"  w_pri={w:.2f}: iou_bld={iou_b:.4f}, iou_veg={iou_v:.4f}, iou_wat={iou_w:.4f}, RMSE_bH={rmse_b:.4f}, RMSE_vH={rmse_v:.4f}, score={score:.4f}")


if __name__ == "__main__":
    main()
