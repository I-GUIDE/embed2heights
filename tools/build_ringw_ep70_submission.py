"""
End-to-end ensemble + calibration + packaging for the ringw ep70 fleet
(3 seeds x 5 folds, see train_predict_ringw_ep70.sbatch).

Steps:
  1. OOF build: per fold, mean the 3 seed runs' val predictions over that
     fold's val ids -> runs/<ens>/oof/fold_F/predictions (+ split.json copy).
     Union over folds covers every train tile exactly once, out-of-fold.
  2. Test ensemble: 15-way mean of all members' test_predictions.
  3. Global calibration on OOF records: per-class thresholds + water CC K +
     height affine via fit_params, scored with leave-one-fold-out nested CV.
     The affine on/off choice is made by nested score.
  4. Per-area-code calibration via tools/sweep_per_code.py (split-half
     validated, falls back to global per code).
  5. Binarize the test ensemble under both calibrations and package two
     submission zips (+ manifests + SUMMARY.md).

Run from the worktree root:
    python tools/build_ringw_ep70_submission.py
Idempotent: finished OOF folds / test ensemble are skipped on rerun.
"""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.inference.calibration import (  # noqa: E402
    aggregate_metrics,
    collect_oof_records,
    eval_records,
    fit_params,
    fold_records,
    not_fold_records,
)
from core.inference.ensemble import ensemble_mean  # noqa: E402
from core.inference.submission import (  # noqa: E402
    package_submission,
    validate_prediction_dir,
)
from tools.sweep_per_code import (  # noqa: E402
    apply_per_code,
    eval_per_code,
    fit_per_code,
    print_fit_report,
)

SEEDS = [0, 1, 42]
N_FOLDS = 5
EXP_PATTERN = "xfusion_095_p3_ringw_ep70_s{seed}_f{fold}"
ENS_NAME = "ens_ringw_ep70"
EXPECTED_TEST_COUNT = 946


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", type=Path, default=SCRIPT_DIR / "runs")
    parser.add_argument("--labels-dir", type=Path,
                        default=Path("/u/dingqi2/workspace/esa/data/train/labels"))
    parser.add_argument("--splits-root", type=Path,
                        default=SCRIPT_DIR / "splits" / "group_code_5fold_seed42")
    parser.add_argument("--submission-dir", type=Path, default=SCRIPT_DIR / "submission")
    parser.add_argument("--water-k-grid", default="0,4,8,12,16,24,32")
    parser.add_argument("--per-code-min-samples", type=int, default=6)
    parser.add_argument("--per-code-min-gain", type=float, default=0.0)
    return parser.parse_args()


def member_dirs(runs_dir):
    members = []
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            members.append((seed, fold, runs_dir / EXP_PATTERN.format(seed=seed, fold=fold)))
    return members


def check_members(members):
    missing = []
    for seed, fold, exp_dir in members:
        for sub in ("predictions", "test_predictions"):
            if not (exp_dir / sub).is_dir() or not any((exp_dir / sub).glob("*.npy")):
                missing.append(f"{exp_dir.name}/{sub}")
    if missing:
        sys.exit("Missing member predictions:\n  " + "\n  ".join(missing))


def build_oof(members, splits_root, ens_root):
    """Per fold, mean the seed runs' predictions over the fold's val ids."""
    for fold in range(N_FOLDS):
        split_src = splits_root / f"fold_{fold}" / "split.json"
        fold_dir = ens_root / "oof" / f"fold_{fold}"
        pred_out = fold_dir / "predictions"
        val_ids = sorted(json.loads(split_src.read_text())["val"])

        existing = list(pred_out.glob("*.npy")) if pred_out.is_dir() else []
        if len(existing) == len(val_ids) and (fold_dir / "split.json").exists():
            print(f"fold {fold}: OOF ensemble already built ({len(existing)} files), skipping")
            continue

        seed_dirs = [exp_dir / "predictions" for seed, f, exp_dir in members if f == fold]
        from core.inference.ensemble import index_prediction_dir
        indexes = [index_prediction_dir(d) for d in seed_dirs]
        pred_out.mkdir(parents=True, exist_ok=True)
        for cid in val_ids:
            arrs = []
            for index in indexes:
                if cid not in index:
                    sys.exit(f"fold {fold}: missing val prediction {cid} in one of {seed_dirs}")
                arrs.append(np.load(index[cid]).astype(np.float32))
            out = np.mean(arrs, axis=0).astype(np.float32)
            out[:3] = np.clip(out[:3], 0.0, 1.0)
            out[3] = np.maximum(out[3], 0.0)
            np.save(pred_out / f"{cid}.npy", out)
        shutil.copy(split_src, fold_dir / "split.json")
        print(f"fold {fold}: wrote {len(val_ids)} seed-mean OOF predictions")


def build_test_ensemble(members, ens_root):
    out_dir = ens_root / "test_predictions"
    existing = list(out_dir.glob("*.npy")) if out_dir.is_dir() else []
    if len(existing) == EXPECTED_TEST_COUNT:
        print(f"test ensemble already built ({len(existing)} files), skipping")
        return out_dir
    inputs = [exp_dir / "test_predictions" for _, _, exp_dir in members]
    count = ensemble_mean(inputs, out_dir)
    print(f"test ensemble: wrote {count} files (mean of {len(inputs)} members)")
    return out_dir


def without_affine(params):
    out = copy.deepcopy(params)
    out["height_affine"] = False
    out.pop("height_affine_params", None)
    return out


def global_calibration(records, water_k_grid):
    """Nested leave-one-fold-out CV; height affine on/off chosen by nested score."""
    base = eval_records(records, {"thresholds": [0.5, 0.5, 0.5],
                                  "water_min_component": 0, "height_affine": False})
    rows_affine, rows_plain, fold_fits = [], [], []
    for fold in range(N_FOLDS):
        train = not_fold_records(records, fold)
        valid = fold_records(records, fold)
        params = fit_params(train, water_k_grid=water_k_grid, height_affine=True)
        rows_affine.append(eval_records(valid, params))
        rows_plain.append(eval_records(valid, without_affine(params)))
        fold_fits.append({"fold": fold, "params": params})
        print(f"  nested fold {fold}: affine {rows_affine[-1]['weighted_score']:.4f} "
              f"| plain {rows_plain[-1]['weighted_score']:.4f}")

    nested_affine = aggregate_metrics(rows_affine)
    nested_plain = aggregate_metrics(rows_plain)
    use_affine = nested_affine["weighted_score"] >= nested_plain["weighted_score"]

    full = fit_params(records, water_k_grid=water_k_grid, height_affine=True)
    chosen = full if use_affine else without_affine(full)
    report = {
        "base_metrics_at_0_5": base,
        "nested_cv_affine": nested_affine,
        "nested_cv_plain": nested_plain,
        "height_affine_chosen": use_affine,
        "nested_cv_folds": fold_fits,
        "full_fit_params": full,
        "chosen_params": chosen,
        "chosen_oof_metrics_optimistic": eval_records(records, chosen),
    }
    return chosen, report


def package(binary_dir, zip_path, manifest_path, manifest):
    validate_prediction_dir(binary_dir, expected_count=EXPECTED_TEST_COUNT)
    package_submission(binary_dir, zip_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packaged {zip_path}")


def main():
    args = parse_args()
    ens_root = args.runs_dir / ENS_NAME
    ens_root.mkdir(parents=True, exist_ok=True)
    members = member_dirs(args.runs_dir)
    check_members(members)

    print("=== 1/5 OOF seed-mean ensembles ===")
    build_oof(members, args.splits_root, ens_root)

    print("=== 2/5 test ensemble (15-way mean) ===")
    test_pred = build_test_ensemble(members, ens_root)

    print("=== 3/5 global calibration (nested 5-fold CV on OOF) ===")
    records, _ = collect_oof_records(str(ens_root / "oof" / "fold_{fold}"), args.labels_dir,
                                     n_folds=N_FOLDS)
    print(f"collected {len(records)} OOF records")
    chosen_global, global_report = global_calibration(records, args.water_k_grid)
    (ens_root / "global_calibration.json").write_text(json.dumps(global_report, indent=2) + "\n")
    print(f"base@0.5 OOF score:   {global_report['base_metrics_at_0_5']['weighted_score']:.4f}")
    print(f"nested CV (honest):   affine={global_report['nested_cv_affine']['weighted_score']:.4f} "
          f"plain={global_report['nested_cv_plain']['weighted_score']:.4f} "
          f"-> affine_chosen={global_report['height_affine_chosen']}")
    print(f"chosen thresholds:    {chosen_global['thresholds']} "
          f"K={chosen_global['water_min_component']}")

    print("=== 4/5 per-area-code calibration ===")
    per_code = fit_per_code(
        records, chosen_global,
        water_k_grid=args.water_k_grid,
        min_samples=args.per_code_min_samples,
        min_gain=args.per_code_min_gain,
    )
    print_fit_report(per_code)
    per_code["oof_global_metrics"] = global_report["chosen_oof_metrics_optimistic"]
    per_code["oof_per_code_metrics_optimistic"] = eval_per_code(records, per_code)
    (ens_root / "per_code_params.json").write_text(json.dumps(per_code, indent=2) + "\n")
    accepted = [c for c, e in per_code["codes"].items() if e["accepted"]]
    print(f"per-code OOF score (optimistic): "
          f"{per_code['oof_per_code_metrics_optimistic']['weighted_score']:.4f} "
          f"({len(accepted)} codes accepted)")

    print("=== 5/5 binarize + package submissions ===")
    args.submission_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [round(t, 4) for t in chosen_global["thresholds"]]

    global_binary = ens_root / "test_predictions_binary_global"
    apply_per_code(test_pred, {"global": chosen_global, "codes": {}}, global_binary)
    package(
        global_binary,
        args.submission_dir / f"{ENS_NAME}_binary_global.zip",
        args.submission_dir / f"{ENS_NAME}_binary_global.manifest.json",
        {
            "source_pred_dir": str(test_pred),
            "members": [d.name for _, _, d in members],
            "calibration": "global per-class thresholds + water CC, nested-CV validated",
            "thresholds": thresholds,
            "water_cc_min_size": chosen_global["water_min_component"],
            "height_affine": chosen_global["height_affine"],
            "oof_nested_cv_score": global_report[
                "nested_cv_affine" if global_report["height_affine_chosen"] else "nested_cv_plain"
            ]["weighted_score"],
            "expected_count": EXPECTED_TEST_COUNT,
        },
    )

    percode_binary = ens_root / "test_predictions_binary_percode"
    apply_per_code(test_pred, per_code, percode_binary)
    package(
        percode_binary,
        args.submission_dir / f"{ENS_NAME}_binary_percode.zip",
        args.submission_dir / f"{ENS_NAME}_binary_percode.manifest.json",
        {
            "source_pred_dir": str(test_pred),
            "members": [d.name for _, _, d in members],
            "calibration": "per-area-code thresholds (split-half validated) over global fallback",
            "global_thresholds": thresholds,
            "water_cc_min_size": chosen_global["water_min_component"],
            "height_affine": chosen_global["height_affine"],
            "accepted_codes": accepted,
            "oof_per_code_score_optimistic":
                per_code["oof_per_code_metrics_optimistic"]["weighted_score"],
            "expected_count": EXPECTED_TEST_COUNT,
        },
    )

    summary = [
        f"# {ENS_NAME} submission summary",
        "",
        f"- members: {len(members)} (seeds {SEEDS} x folds 0-{N_FOLDS - 1})",
        f"- OOF records: {len(records)}",
        f"- OOF base@0.5 score: {global_report['base_metrics_at_0_5']['weighted_score']:.4f}",
        f"- OOF nested-CV score (honest, global cal): "
        f"{global_report['nested_cv_affine']['weighted_score']:.4f} (affine) / "
        f"{global_report['nested_cv_plain']['weighted_score']:.4f} (plain)",
        f"- global thresholds: {thresholds}, water K={chosen_global['water_min_component']}, "
        f"height_affine={chosen_global['height_affine']}",
        f"- per-code accepted: {len(accepted)} codes -> {', '.join(accepted) if accepted else '(none)'}",
        f"- per-code OOF score (optimistic): "
        f"{per_code['oof_per_code_metrics_optimistic']['weighted_score']:.4f}",
        "",
        "Submissions:",
        f"- {args.submission_dir / (ENS_NAME + '_binary_global.zip')}",
        f"- {args.submission_dir / (ENS_NAME + '_binary_percode.zip')}",
        "",
        "Submit the global zip first (honest nested-CV backing); submit per-code "
        "only if its split-half gains look consistently positive.",
    ]
    (ens_root / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
