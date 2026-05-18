"""Per-fold per-location diagnostic for the 5-fold ensemble.

For each fold i, loads its model's predictions on its val tiles and computes:
  - per-location iou_bld/iou_veg/iou_wat, RMSE_bH, RMSE_vH at base 0.5 thresholds
  - per-location optimal building threshold via grid search
  - cross-fold robust building threshold

Outputs:
  - JSON report per fold with per-location stats
  - Aggregated cross-fold report
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


def loc_of(tile_id):
    parts = tile_id.split("_")
    return parts[1] if len(parts) >= 2 else "?"


def load_pair(pred_path, label_path):
    pred = np.load(pred_path)  # shape [4, H, W]: bld_prob, veg_prob, wat_prob, height_m
    with rasterio.open(label_path) as src:
        lab = src.read().astype(np.float32)  # shape [4, H, W]: bld_frac, veg_frac, wat_frac, height_m
    # Align shapes: crop both to the common (min) HxW
    h = min(pred.shape[1], lab.shape[1])
    w = min(pred.shape[2], lab.shape[2])
    pred = pred[:, :h, :w]
    lab = lab[:, :h, :w]
    return pred, lab


def iou_at_thresh(pred_prob, label_frac, thresh, label_thresh=0.0):
    pred_bin = (pred_prob >= thresh).astype(np.uint8)
    lab_bin = (label_frac > label_thresh).astype(np.uint8)
    inter = (pred_bin & lab_bin).sum()
    union = (pred_bin | lab_bin).sum()
    return inter / max(union, 1)


def rmse_on_pixels(pred_h, label_h, label_pres):
    mask = label_pres > 0
    if mask.sum() == 0:
        return None
    diff = (pred_h - label_h)[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def aggregate_fold(pred_dir, labels_dir, val_ids):
    """Returns dict: loc -> {iou_b, iou_v, iou_w, rmse_bh, rmse_vh, n_tiles, n_pix, bld_dens, veg_dens, wat_dens}.
    iou values are computed CUMULATIVELY across pixels per location (correct IoU aggregation, not mean-of-means)."""
    accum = defaultdict(lambda: {
        "bld_inter05": 0, "bld_union05": 0,
        "veg_inter05": 0, "veg_union05": 0,
        "wat_inter05": 0, "wat_union05": 0,
        "bH_sse": 0.0, "bH_n": 0,
        "vH_sse": 0.0, "vH_n": 0,
        "n_tiles": 0, "n_pix": 0,
        "bld_pix": 0, "veg_pix": 0, "wat_pix": 0,
        # For sweep
        "bld_probs": [], "bld_labs": [],
    })
    # We also need raw probs/labels for building threshold sweep — but storing all is expensive.
    # Instead, build per-tile probability histogram and label-positives count for sweep.
    # Simpler: bucket bld_prob into 1024 bins and remember inter/union of (pred>=t) and (lab>0) per tile.
    # That's complex. For initial diagnostic, just do iou at thresh 0.5 and a sweep at the end with 5-tile sample per loc.
    # Better: directly accumulate per-location pred and label arrays. Memory:
    #   2024 tiles * 256*256 * 2 channels * 4 bytes = ~1 GB. Manageable.
    # We'll keep flat arrays per location for buildings only (the bottleneck metric).
    bld_flat = defaultdict(lambda: {"prob": [], "lab": []})

    label_map = build_label_map(str(labels_dir))
    for tid in val_ids:
        pf = Path(pred_dir) / f"{tid}.npy"
        if not pf.exists():
            continue
        if tid not in label_map:
            continue
        lf = label_map[tid]
        loc = loc_of(pf.stem)
        pred, lab = load_pair(pf, lf)
        # channels: 0 bld, 1 veg, 2 wat, 3 height
        for c, key, lab_thresh in [(0, "bld", 0.0), (1, "veg", 0.0), (2, "wat", 0.0)]:
            pb = (pred[c] >= 0.5).astype(np.uint8)
            lb = (lab[c] > lab_thresh).astype(np.uint8)
            accum[loc][f"{key}_inter05"] += int((pb & lb).sum())
            accum[loc][f"{key}_union05"] += int((pb | lb).sum())
            accum[loc][f"{key}_pix"] += int(lb.sum())
        # heights
        for c, pres_c, key in [(3, 0, "bH"), (3, 1, "vH")]:
            mask = lab[pres_c] > 0
            if mask.sum() > 0:
                diff = (pred[3] - lab[3])[mask]
                accum[loc][f"{key}_sse"] += float((diff ** 2).sum())
                accum[loc][f"{key}_n"] += int(mask.sum())
        accum[loc]["n_tiles"] += 1
        accum[loc]["n_pix"] += int(lab.shape[1] * lab.shape[2])
        # building flat for sweep — subsample 16x to keep memory ok
        bld_flat[loc]["prob"].append(pred[0, ::4, ::4].astype(np.float32).flatten())
        bld_flat[loc]["lab"].append((lab[0, ::4, ::4] > 0).astype(np.uint8).flatten())

    # Compute per-location IoU at 0.5 and RMSE
    results = {}
    for loc, d in accum.items():
        iou_b = d["bld_inter05"] / max(d["bld_union05"], 1)
        iou_v = d["veg_inter05"] / max(d["veg_union05"], 1)
        iou_w = d["wat_inter05"] / max(d["wat_union05"], 1)
        rmse_bh = float(np.sqrt(d["bH_sse"] / max(d["bH_n"], 1))) if d["bH_n"] else None
        rmse_vh = float(np.sqrt(d["vH_sse"] / max(d["vH_n"], 1))) if d["vH_n"] else None
        # Sweep building threshold on this location
        all_probs = np.concatenate(bld_flat[loc]["prob"]) if bld_flat[loc]["prob"] else np.array([])
        all_labs = np.concatenate(bld_flat[loc]["lab"]) if bld_flat[loc]["lab"] else np.array([])
        sweep = {}
        for t in np.arange(0.30, 0.81, 0.025):
            pb = (all_probs >= t).astype(np.uint8)
            inter = int((pb & all_labs).sum())
            union = int((pb | all_labs).sum())
            sweep[float(round(t, 3))] = inter / max(union, 1)
        best_t = max(sweep, key=sweep.get) if sweep else None
        results[loc] = {
            "iou_b@0.5": iou_b,
            "iou_v@0.5": iou_v,
            "iou_w@0.5": iou_w,
            "rmse_bh": rmse_bh,
            "rmse_vh": rmse_vh,
            "n_tiles": d["n_tiles"],
            "n_pix": d["n_pix"],
            "bld_density": d["bld_pix"] / max(d["n_pix"], 1),
            "veg_density": d["veg_pix"] / max(d["n_pix"], 1),
            "wat_density": d["wat_pix"] / max(d["n_pix"], 1),
            "sweep_iou_b_by_thresh": sweep,
            "best_bld_thresh": best_t,
            "best_bld_iou": sweep.get(best_t, 0),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--exp-name", default=None,
                    help="experiment dir name (default exp_tess128_e100_fold{N})")
    ap.add_argument("--output-dir", default=str(SCRIPT_DIR / "runs" / "diagnostics"))
    args = ap.parse_args()

    exp = args.exp_name or f"exp_tess128_e100_fold{args.fold}"
    pred_dir = SCRIPT_DIR / "runs" / exp / "predictions"
    split_path = SCRIPT_DIR / "splits" / "group_code_5fold_seed42" / f"fold_{args.fold}" / "split.json"

    with open(split_path) as f:
        split = json.load(f)
    val_ids = split["val"]
    print(f"Fold {args.fold}: {len(val_ids)} val tiles; pred dir: {pred_dir}")

    results = aggregate_fold(pred_dir, LABELS_DIR, val_ids)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Default name: per_location_{exp_name}.json; legacy: per_location_fold{N}.json
    if args.exp_name and args.exp_name != f"exp_tess128_e100_fold{args.fold}":
        out_path = out_dir / f"per_location_{args.exp_name}.json"
    else:
        out_path = out_dir / f"per_location_fold{args.fold}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote: {out_path}")

    # Print summary table sorted by n_tiles
    print(f"\n{'loc':>5} {'n_tiles':>7} {'bld_dens':>9} {'iou_b@.5':>9} {'best_t':>7} {'iou_b@best':>10}")
    print("-" * 60)
    for loc, r in sorted(results.items(), key=lambda x: -x[1]["n_tiles"]):
        print(f"{loc:>5} {r['n_tiles']:>7} {r['bld_density']:>9.4f} {r['iou_b@0.5']:>9.4f} {r['best_bld_thresh']:>7} {r['best_bld_iou']:>10.4f}")


if __name__ == "__main__":
    main()
