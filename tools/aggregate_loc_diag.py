"""Aggregate per-fold per-location diagnostics into a cross-fold view.

Reads runs/diagnostics/per_location_fold{0..4}.json and produces:
 - cross-fold per-location aggregates (sum tiles, weighted iou_b, best thresh)
 - Top locations sorted by total tiles
 - Per-threshold-sweep curve at the cross-fold level
"""
import json
from pathlib import Path
from collections import defaultdict

DIAG_DIR = Path(__file__).resolve().parents[1] / "runs" / "diagnostics"


def main():
    folds = {}
    for f in range(5):
        p = DIAG_DIR / f"per_location_fold{f}.json"
        folds[f] = json.load(open(p))

    # Cross-fold aggregate per location
    agg = defaultdict(lambda: {"n_tiles": 0, "iou_b_weighted_sum": 0.0,
                                "best_t_sum": 0.0, "best_t_n": 0,
                                "iou_b_at_best_sum": 0.0, "iou_b_at_best_n": 0,
                                "sweep": defaultdict(lambda: {"iou_sum": 0.0, "n": 0})})
    for fid, fdata in folds.items():
        for loc, r in fdata.items():
            agg[loc]["n_tiles"] += r["n_tiles"]
            # weighted by tiles
            agg[loc]["iou_b_weighted_sum"] += r["iou_b@0.5"] * r["n_tiles"]
            if r["best_bld_thresh"] is not None:
                agg[loc]["best_t_sum"] += float(r["best_bld_thresh"]) * r["n_tiles"]
                agg[loc]["best_t_n"] += r["n_tiles"]
                agg[loc]["iou_b_at_best_sum"] += r["best_bld_iou"] * r["n_tiles"]
                agg[loc]["iou_b_at_best_n"] += r["n_tiles"]
            for t_str, iou in r["sweep_iou_b_by_thresh"].items():
                t = float(t_str)
                agg[loc]["sweep"][t]["iou_sum"] += iou * r["n_tiles"]
                agg[loc]["sweep"][t]["n"] += r["n_tiles"]

    print(f"{'loc':>5} {'n_tiles':>8} {'mean_iou@0.5':>13} {'mean_best_t':>12} {'mean_iou@best':>14}")
    print("-" * 70)
    for loc, r in sorted(agg.items(), key=lambda x: -x[1]["n_tiles"]):
        mean_iou_05 = r["iou_b_weighted_sum"] / max(r["n_tiles"], 1)
        mean_t = r["best_t_sum"] / max(r["best_t_n"], 1)
        mean_iou_best = r["iou_b_at_best_sum"] / max(r["iou_b_at_best_n"], 1)
        print(f"{loc:>5} {r['n_tiles']:>8} {mean_iou_05:>13.4f} {mean_t:>12.3f} {mean_iou_best:>14.4f}")

    print()
    print("=== Global cross-fold threshold sweep (weighted by tile count) ===")
    global_sweep = defaultdict(lambda: {"iou_sum": 0.0, "n": 0})
    for loc, r in agg.items():
        for t, d in r["sweep"].items():
            global_sweep[t]["iou_sum"] += d["iou_sum"]
            global_sweep[t]["n"] += d["n"]
    print(f"{'thresh':>8} {'global_iou_b':>14}")
    print("-" * 26)
    sorted_t = sorted(global_sweep.keys())
    for t in sorted_t:
        d = global_sweep[t]
        mean_iou = d["iou_sum"] / max(d["n"], 1)
        print(f"{t:>8.3f} {mean_iou:>14.4f}")
    best_t = max(global_sweep, key=lambda t: global_sweep[t]["iou_sum"] / max(global_sweep[t]["n"], 1))
    best_iou = global_sweep[best_t]["iou_sum"] / max(global_sweep[best_t]["n"], 1)
    print(f"\nGlobal cross-fold best building threshold: {best_t:.3f} → iou_b = {best_iou:.4f}")

    # Save aggregated JSON
    out = {
        "per_location": {
            loc: {
                "n_tiles": r["n_tiles"],
                "mean_iou_b_at_0.5": r["iou_b_weighted_sum"] / max(r["n_tiles"], 1),
                "mean_best_threshold": r["best_t_sum"] / max(r["best_t_n"], 1) if r["best_t_n"] else None,
                "mean_iou_b_at_best": r["iou_b_at_best_sum"] / max(r["iou_b_at_best_n"], 1) if r["iou_b_at_best_n"] else None,
            }
            for loc, r in agg.items()
        },
        "global_sweep": {
            f"{t:.3f}": global_sweep[t]["iou_sum"] / max(global_sweep[t]["n"], 1)
            for t in sorted_t
        },
        "global_best_threshold": best_t,
        "global_best_iou_b": best_iou,
    }
    with open(DIAG_DIR / "cross_fold_aggregate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {DIAG_DIR / 'cross_fold_aggregate.json'}")


if __name__ == "__main__":
    main()
