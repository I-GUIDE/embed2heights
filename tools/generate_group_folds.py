"""
Generate grouped cross-validation splits for train.py.

The default grouping is the region/code token in label filenames such as
`label_0000_BE_2023.tif`: every patch with code `BE` is assigned to the same
validation fold. Split files are written in the existing train.py-compatible
format:

    {"train": ["0000_BE", ...], "val": ["0123_KE", ...]}

Usage:
    python tools/generate_group_folds.py
    python tools/generate_group_folds.py --group-mode code_year --n-splits 5
"""

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "train", "labels"))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "splits", "group_code_5fold_seed42")

_YEAR_RE = re.compile(r"_\d{4}$")


def normalize_core_id(path):
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("label_"):
        base = base[len("label_"):]
    if base.startswith("pred_"):
        base = base[len("pred_"):]
    return _YEAR_RE.sub("", base)


def raw_label_id(path):
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("label_"):
        base = base[len("label_"):]
    return base


def group_id_from_label(path, mode):
    raw = raw_label_id(path)
    parts = raw.split("_")
    has_year = len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 4

    if mode == "full":
        return raw
    if mode == "year":
        return parts[-1] if has_year else "no_year"
    if mode == "code_year":
        if has_year:
            return "{}_{}".format(parts[-2], parts[-1])
        return parts[-1] if parts else raw
    if mode == "code":
        if has_year:
            return parts[-2]
        if len(parts) >= 2:
            return parts[-1]
        return raw
    raise ValueError("unknown group mode: {}".format(mode))


def assign_groups_to_folds(group_to_ids, n_splits, seed):
    if n_splits < 2:
        raise ValueError("--n-splits must be >= 2")
    if len(group_to_ids) < n_splits:
        raise ValueError(
            "Need at least as many groups as folds: groups={} folds={}".format(
                len(group_to_ids), n_splits
            )
        )

    rng = random.Random(seed)
    groups = list(group_to_ids.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    folds = [{"groups": [], "ids": []} for _ in range(n_splits)]
    for group, ids in groups:
        target = min(range(n_splits), key=lambda i: len(folds[i]["ids"]))
        folds[target]["groups"].append(group)
        folds[target]["ids"].extend(ids)

    for fold in folds:
        fold["groups"].sort()
        fold["ids"].sort()
    return folds


def write_folds(core_ids, folds, output_dir, group_mode, seed):
    os.makedirs(output_dir, exist_ok=True)
    all_ids = set(core_ids)
    summary = {
        "group_mode": group_mode,
        "seed": seed,
        "n_splits": len(folds),
        "n_samples": len(core_ids),
        "folds": [],
    }

    for i, fold in enumerate(folds):
        val_ids = set(fold["ids"])
        train_ids = sorted(all_ids - val_ids)
        val_ids = sorted(val_ids)

        fold_dir = os.path.join(output_dir, "fold_{}".format(i))
        os.makedirs(fold_dir, exist_ok=True)
        split_path = os.path.join(fold_dir, "split.json")
        with open(split_path, "w") as f:
            json.dump({"train": train_ids, "val": val_ids}, f, indent=2)

        summary["folds"].append({
            "fold": i,
            "split_file": split_path,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_val_groups": len(fold["groups"]),
            "val_groups": fold["groups"],
        })

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path, summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--group-mode",
        choices=["code", "code_year", "year", "full"],
        default="code",
        help=(
            "Grouping key from label filename. code keeps all patches with the "
            "same region/code token together; code_year separates years; year "
            "is a coarse year holdout; full is one group per raw label id."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    label_files = sorted(glob.glob(os.path.join(args.labels_dir, "**", "label_*.tif"), recursive=True))
    if not label_files:
        raise FileNotFoundError("No label_*.tif files found in {}".format(args.labels_dir))

    group_to_ids = defaultdict(list)
    for path in label_files:
        core_id = normalize_core_id(path)
        group = group_id_from_label(path, args.group_mode)
        group_to_ids[group].append(core_id)

    for group in group_to_ids:
        group_to_ids[group] = sorted(set(group_to_ids[group]))

    core_ids = sorted({normalize_core_id(path) for path in label_files})
    folds = assign_groups_to_folds(group_to_ids, args.n_splits, args.seed)
    summary_path, summary = write_folds(core_ids, folds, args.output_dir, args.group_mode, args.seed)

    print("Found {} samples in {} groups (mode={})".format(
        len(core_ids), len(group_to_ids), args.group_mode
    ))
    print("Wrote grouped folds to {}".format(args.output_dir))
    for fold in summary["folds"]:
        print(
            "  fold {fold}: train={n_train} val={n_val} val_groups={n_val_groups} split={split_file}".format(
                **fold
            )
        )
    print("Summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
