"""
Generate the 15 ringw ep70 ensemble configs (seeds 0/1/42 x folds 0-4).

Each config is the xfusion_095_p3_ringw_fold0 recipe with epochs=70 and the
(seed, fold) swapped in. Experiment names follow
    xfusion_095_p3_ringw_ep70_s{seed}_f{fold}

Usage:
    python tools/gen_ringw_ep70_configs.py            # write configs
    python tools/gen_ringw_ep70_configs.py --check    # list without writing
"""

import argparse
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1]

SEEDS = [0, 1, 42]
FOLDS = [0, 1, 2, 3, 4]
BASE_CONFIG = SCRIPT_DIR / "configs" / "active" / "xfusion_095_p3_ringw_fold0.yml"
OUTPUT_DIR = SCRIPT_DIR / "configs" / "active" / "ringw_ep70"
EPOCHS = 70


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="List configs without writing.")
    args = parser.parse_args()

    with BASE_CONFIG.open() as handle:
        base = yaml.safe_load(handle)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for seed in SEEDS:
        for fold in FOLDS:
            name = f"xfusion_095_p3_ringw_ep70_s{seed}_f{fold}"
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg["name"] = name
            cfg["description"] = (
                f"ringw ensemble member: seed={seed} fold={fold} epochs={EPOCHS}. "
                "Same recipe as xfusion_095_p3_ringw_fold0 (boundary-ring-weighted "
                "building presence BCE on the P3 baseline)."
            )
            cfg["reference"] = "xfusion_095_p3_ringw_fold0"
            cfg["data"]["split_file"] = (
                "${REPO_DIR}/splits/group_code_5fold_seed42/fold_%d/split.json" % fold
            )
            cfg["training"]["seed"] = seed
            cfg["training"]["epochs"] = EPOCHS
            cfg["runtime"]["experiment_name"] = name

            path = OUTPUT_DIR / f"{name}.yml"
            names.append(name)
            if args.check:
                print(f"would write {path}")
                continue
            with path.open("w") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
            print(f"wrote {path}")

    if len(names) != len(SEEDS) * len(FOLDS):
        sys.exit("config count mismatch")


if __name__ == "__main__":
    main()
