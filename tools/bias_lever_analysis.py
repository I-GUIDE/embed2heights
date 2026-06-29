"""Pre-submission bias / lever analysis on OOF (val) predictions.

Mirrors the statistical digging that surfaced the labmate's levers:
  (1) seg-purify  -- building IoU kept climbing inside the height-purify stage;
  (2) height bias -- per-class height is systematically off.

IMPORTANT: it scores the *real* competition metric -- per-tile RMSE then mean
over tiles (NOT global-pixel RMSE), and applies candidate height offsets on the
*predicted* dominant class (argmax ch0-2), exactly as tools/merge_twostage_preds
does at submission time. Global-pixel RMSE badly overstates the value of large
offsets; under the true metric the building optimum is small and +0.25 can be
net-negative. Re-run this on the run you actually intend to submit -- the optimal
offset is model-specific.

Read-only. Prints a report; writes nothing.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.calibration import load_labeled_predictions  # noqa: E402
from core.metrics import LABEL_THRESHOLD  # noqa: E402


def region_of(cid):
    parts = cid.split("_")
    return parts[1] if len(parts) > 1 else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--labels-dir",
                    default="/projects/bcrm/emb2height/data/train/labels")
    ap.add_argument("--split-file", required=True,
                    help="the fold's split.json so only val (held-out) tiles score")
    args = ap.parse_args()

    preds, labels = load_labeled_predictions(
        args.pred_dir, args.labels_dir, split_file=args.split_file)
    cids = sorted(preds)
    print(f"# matched {len(cids)} val tiles from {args.pred_dir}\n")

    # Pre-crop and cache per tile: pred height, predicted argmax class, GT masks.
    tiles = []
    region_res = defaultdict(list)   # GT-building residuals by region
    for cid in cids:
        p, l = preds[cid], labels[cid]
        h = min(p.shape[1], l.shape[1]); w = min(p.shape[2], l.shape[2])
        p, l = p[:, :h, :w], l[:, :h, :w]
        arg = np.argmax(p[:3], axis=0)
        mb = l[0] > LABEL_THRESHOLD
        mv = l[1] > LABEL_THRESHOLD
        tiles.append((p[3], arg, mb, mv, l[3]))
        if mb.any():
            region_res[region_of(cid)].append(p[3][mb] - l[3][mb])

    def metric(bo, vo):
        """Per-tile RMSE (then mean) with offsets applied on PREDICTED class."""
        rb, rv = [], []
        for ph, arg, mb, mv, gh in tiles:
            h = ph
            if bo or vo:
                h = ph.copy()
                if bo:
                    h[arg == 0] += bo
                if vo:
                    h[arg == 1] += vo
            if mb.any():
                rb.append(np.sqrt(np.mean((h[mb] - gh[mb]) ** 2)))
            if mv.any():
                rv.append(np.sqrt(np.mean((h[mv] - gh[mv]) ** 2)))
        return float(np.mean(rb)), float(np.mean(rv))

    b0, v0 = metric(0.0, 0.0)
    print(f"baseline    RMSE_bld {b0:.4f}   RMSE_veg {v0:.4f}\n")

    # Sweep building offset (veg held 0) and veg offset (building held 0).
    grid_b = np.round(np.arange(-0.10, 0.61, 0.05), 3)
    grid_v = np.round(np.arange(-0.60, 0.11, 0.05), 3)
    bbest = min(((metric(o, 0.0)[0], o) for o in grid_b))
    vbest = min(((metric(0.0, o)[1], o) for o in grid_v))
    print("== building height offset sweep (real metric, pred-class mask) ==")
    for o in [0.0, 0.10, 0.25, 0.50, bbest[1]]:
        r = metric(o, 0.0)[0]
        tag = "  <== optimum" if abs(o - bbest[1]) < 1e-9 else ""
        print(f"  {o:+.2f} -> RMSE_bld {r:.4f}  ({r - b0:+.4f}){tag}")
    print("== vegetation height offset sweep ==")
    for o in [0.0, -0.10, -0.25, vbest[1]]:
        r = metric(0.0, o)[1]
        tag = "  <== optimum" if abs(o - vbest[1]) < 1e-9 else ""
        print(f"  {o:+.2f} -> RMSE_veg {r:.4f}  ({r - v0:+.4f}){tag}")

    bb, vv = bbest[1], vbest[1]
    cb, cv = metric(bb, vv)
    print(f"\n== RECOMMENDED (this model/fold) ==")
    print(f"  --build-height-offset {bb:+.2f}  --veg-height-offset {vv:+.2f}")
    print(f"  RMSE_bld {b0:.4f}->{cb:.4f} ({cb-b0:+.4f})   "
          f"RMSE_veg {v0:.4f}->{cv:.4f} ({cv-v0:+.4f})")
    print("  (only apply if the gain is clearly > val noise; re-fit per model.)")

    print("\n== per-region GT-building bias (diagnostic; pixel-weighted) ==")
    rows = [(r, (d := np.concatenate(ds)).size, float(d.mean()))
            for r, ds in region_res.items()]
    for r, n, m in sorted(rows, key=lambda x: -abs(x[2]))[:10]:
        print(f"  {r:<4} n={n:>9}  mean_res={m:+.3f}")


if __name__ == "__main__":
    main()
