#!/usr/bin/env python
"""Compare val leaderboard metrics across training runs.

For each run directory it reads loss_history.jsonl and selects the epoch with
the highest val_weighted_score (the same criterion as model_best_ws.pth, i.e.
what you would submit). It prints a table of that epoch's leaderboard metrics
and, when a baseline run is given, the delta vs the baseline.

Usage
-----
  # compare specific runs, delta vs the bilinear baseline
  python tools/compare_runs.py \
      --baseline uw_gated_F_bilinear_fold0 \
      uw_gated_F_carafe_fold0 uw_gated_F_dysample_fold0 \
      arch_carafe_softbin_pinball_fold0 arch_dysample_softbin_pinball_fold0

  # or auto-discover every run under runs/ that has a history file
  python tools/compare_runs.py --baseline uw_gated_F_bilinear_fold0

Metrics (leaderboard column -> history key):
  iou_build    <- iou_buildings        (higher better, weight 0.25)
  iou_veg      <- iou_trees            (higher better, weight 0.15)
  iou_water    <- iou_water            (higher better, weight 0.15)
  rmse_h_build <- RMSE_building_height (lower  better, weight 0.25, norm 3.0)
  rmse_h_veg   <- RMSE_vegetation_height(lower better, weight 0.20, norm 5.0)
  final_score  <- val_weighted_score
"""

import argparse
import glob
import json
import math
import os

# (label, history key, higher_is_better)
METRICS = [
    ("iou_build", "iou_buildings", True),
    ("iou_veg", "iou_trees", True),
    ("iou_water", "iou_water", True),
    ("rmse_h_build", "RMSE_building_height", False),
    ("rmse_h_veg", "RMSE_vegetation_height", False),
]


def read_history(run_dir):
    """Return list of per-epoch records (dicts) from loss_history.jsonl."""
    path = os.path.join(run_dir, "loss_history.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a half-written trailing line on a live run
    return records


def best_record(records):
    """Pick the epoch with the highest finite val_weighted_score."""
    finite = [r for r in records
              if isinstance(r.get("val_weighted_score"), (int, float))
              and math.isfinite(r["val_weighted_score"])]
    if not finite:
        return None
    return max(finite, key=lambda r: r["val_weighted_score"])


def summarize(run_dir):
    records = read_history(run_dir)
    best = best_record(records)
    if best is None:
        return {"epochs": len(records), "best": None}
    lb = best.get("val_leaderboard_metrics", {})
    row = {
        "epochs": len(records),
        "best_epoch": best.get("epoch"),
        "final_score": best.get("val_weighted_score"),
    }
    for label, key, _ in METRICS:
        row[label] = lb.get(key)
    return {"epochs": len(records), "best": row}


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and math.isfinite(v) else "  -  "


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="Run names (under --runs-dir) or paths.")
    ap.add_argument("--runs-dir", default="runs", help="Base directory of runs (default: runs).")
    ap.add_argument("--baseline", default=None,
                    help="Run name/path to diff against (shown as Δ rows).")
    args = ap.parse_args()

    def resolve(name):
        if os.path.isdir(name):
            return name
        return os.path.join(args.runs_dir, name)

    names = list(args.runs)
    if not names:
        # auto-discover runs that have a history file
        for p in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
            if os.path.exists(os.path.join(p, "loss_history.jsonl")):
                names.append(os.path.basename(p))
    if args.baseline and args.baseline not in names:
        names = [args.baseline] + names

    cols = ["final_score"] + [m[0] for m in METRICS]
    header = f"{'run':<42} {'ep':>5} " + " ".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))

    base_row = None
    rows = {}
    for name in names:
        s = summarize(resolve(name))
        rows[name] = s
        if s["best"] is None:
            print(f"{name:<42} {s['epochs']:>5}   (no finite val_weighted_score yet)")
            continue
        b = s["best"]
        line = f"{name:<42} {b['best_epoch']:>5} " + \
               " ".join(f"{fmt(b[c]):>12}" for c in cols)
        # mark runs still in progress
        suffix = "" if s["epochs"] else ""
        print(line + suffix)
        if args.baseline and name == args.baseline:
            base_row = b

    # Delta block vs baseline
    if base_row is not None:
        print()
        print(f"Δ vs baseline ({args.baseline}) — IoU: + is better; RMSE: - is better")
        print("-" * len(header))
        for name in names:
            if name == args.baseline or rows[name]["best"] is None:
                continue
            b = rows[name]["best"]
            parts = []
            for c in cols:
                bv, rv = base_row.get(c), b.get(c)
                if isinstance(bv, (int, float)) and isinstance(rv, (int, float)):
                    d = rv - bv
                    parts.append(f"{d:>+12.4f}")
                else:
                    parts.append(f"{'-':>12}")
            print(f"{name:<42} {'':>5} " + " ".join(parts))

    print()
    print("Note: 'ep' = epoch of best val_weighted_score (= what model_best_ws.pth holds).")
    print("Runs still training show their best-so-far; re-run this after they finish.")


if __name__ == "__main__":
    main()
