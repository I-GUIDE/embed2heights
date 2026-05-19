"""Average multiple prediction directories per-tile, then run a sweep.

Usage:
  python tools/avg_and_sweep.py \
    --pred-dirs runs/exp_a/predictions,runs/exp_b/predictions,... \
    --output-dir runs/ens_my_ensemble/predictions \
    --split-file splits/.../fold_0/split.json
"""
import argparse, sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dirs", required=True, help="Comma-separated list of pred dirs to average")
    ap.add_argument("--output-dir", required=True, help="Where to write the averaged preds")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.pred_dirs.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(dirs[0].glob("*.npy"))
    print(f"Averaging {len(dirs)} dirs over {len(files)} tiles → {out_dir}")
    for i, f in enumerate(files):
        preds = []
        for d in dirs:
            p = d / f.name
            if p.exists():
                preds.append(np.load(p))
        if not preds:
            continue
        # Align shapes
        h = min(p.shape[1] for p in preds)
        w = min(p.shape[2] for p in preds)
        preds = [p[:, :h, :w] for p in preds]
        avg = np.mean(np.stack(preds), axis=0).astype(np.float32)
        np.save(out_dir / f.name, avg)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"Wrote {len(files)} averaged preds")


if __name__ == "__main__":
    main()
