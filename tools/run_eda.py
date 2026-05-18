#!/usr/bin/env python3
"""CLI orchestrator for the OOP EDA library in tools/eda.py.

Usage:
    python tools/run_eda.py                           # run full pipeline
    python tools/run_eda.py --only labels embeddings_alphaearth
    python tools/run_eda.py --sample 100              # smoke test on 100 patches
    python tools/run_eda.py --overwrite               # force re-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda  # noqa: E402  (local import from tools/)


def _has_outputs(name: str, out_dir: Path) -> bool:
    """Quick idempotency check: does this analyzer already have its marker CSV?"""
    markers = {
        "labels": ["label_per_patch.csv", "label_aggregate.csv"],
        "catalog": ["catalog_geometry.csv"],
        "split": ["split_parity.csv"],
        "probe": ["probe_r2.csv"],
        "shift": ["train_test_shift.csv"],
        "difficulty": ["patch_difficulty.csv"],
        "report": ["REPORT.md"],
    }
    if name.startswith("embeddings_"):
        fam = name.split("_", 1)[1]
        return (out_dir / f"emb_{fam}_summary.csv").exists()
    return all((out_dir / m).exists() for m in markers.get(name, []))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None,
                    help="Subset of analyzer names to run (e.g. labels embeddings_alphaearth report).")
    ap.add_argument("--sample", type=int, default=None,
                    help="Cap number of patches processed (for smoke tests).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run analyzers even if their output CSVs already exist.")
    ap.add_argument("--out-dir", type=Path, default=eda.DEFAULT_EDA_OUT)
    ap.add_argument("--figures-dir", type=Path, default=eda.DEFAULT_FIGURES)
    args = ap.parse_args()

    print(f"[run_eda] data_root={eda.DEFAULT_DATA_ROOT}")
    print(f"[run_eda] out_dir={args.out_dir}")
    print(f"[run_eda] figures_dir={args.figures_dir}")
    print(f"[run_eda] sample={args.sample}  overwrite={args.overwrite}")

    t0 = time.time()
    print("[run_eda] building PatchIndex...")
    index = eda.PatchIndex()
    print(f"[run_eda] train patches: {len(index.ids('train'))}  "
          f"labeled: {len(index.labeled_ids())}  "
          f"test patches: {len(index.ids('test'))}  "
          f"({time.time() - t0:.1f}s)")

    selected = set(args.only) if args.only else None
    for name, factory in eda.ANALYZER_ORDER:
        if selected is not None and name not in selected:
            continue
        if not args.overwrite and _has_outputs(name, args.out_dir):
            print(f"[{name}] outputs already exist — skipping (use --overwrite to force)")
            continue
        print(f"[{name}] starting...")
        t = time.time()
        analyzer = factory(
            index,
            out_dir=args.out_dir,
            figures_dir=args.figures_dir,
            sample=args.sample,
        )
        try:
            analyzer.run()
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            raise
        print(f"[{name}] done in {time.time() - t:.1f}s")

    print(f"[run_eda] total elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
