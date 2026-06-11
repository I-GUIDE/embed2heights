"""
Per-area-code threshold calibration on OOF predictions.

Area code = the two-letter group in the core id (e.g. 0042_KE -> KE). The
5-fold split is grouped by code, so every code's OOF predictions come from a
model that never saw that code in training, and test tiles reuse the same
codes -- per-code thresholds fitted on OOF transfer to test by code lookup.

`fit` groups OOF records by code and, for codes with enough samples, tunes
per-class presence thresholds (building / vegetation via greedy IoU, water
jointly with the connected-component min-size K). Each code is only accepted
if an interleaved split-half check (fit on half, score on the other half,
both directions) beats the global parameters; everything else falls back to
the global fit. Height handling (affine on/off + params) is inherited from
the global parameters.

`apply` binarizes a test prediction dir using the per-code parameter file:
channels 0-2 thresholded per code, water CC filter per code, channel 3 left
raw or globally affine-corrected.

Usage:
    python tools/sweep_per_code.py fit \\
        --oof-root runs/ens_ringw_ep70/oof \\
        --labels-dir /path/to/train/labels \\
        --global-params-json runs/ens_ringw_ep70/global_params.json \\
        --output-json runs/ens_ringw_ep70/per_code_params.json

    python tools/sweep_per_code.py apply \\
        --pred-dir runs/ens_ringw_ep70/test_predictions \\
        --params-json runs/ens_ringw_ep70/per_code_params.json \\
        --output-dir runs/ens_ringw_ep70/test_predictions_binary_percode
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from core.data.discovery import normalize_core_id  # noqa: E402
from core.inference.calibration import (  # noqa: E402
    aggregate_metrics,
    apply_height_channel,
    apply_water_cc_filter,
    collect_oof_records,
    eval_records,
    fit_params,
    tune_class_threshold,
    tune_water,
)


def code_of(core_id):
    parts = str(core_id).split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse area code from core id: {core_id}")
    return parts[1]


def with_height_from(params, reference):
    """Copy the height-affine decision from the global fit into per-code params."""
    out = dict(params)
    out["height_affine"] = bool(reference.get("height_affine", False))
    if out["height_affine"]:
        out["height_affine_params"] = reference["height_affine_params"]
    return out


def fit_code_thresholds(records, threshold_grid, water_grid, k_grid, reference):
    b_t, b_iou = tune_class_threshold(records, 0, threshold_grid)
    v_t, v_iou = tune_class_threshold(records, 1, threshold_grid)
    water = tune_water(records, water_grid, k_grid)
    params = {
        "thresholds": [b_t, v_t, water["threshold"]],
        "water_min_component": water["k"],
        "fit_details": {
            "building_iou": b_iou,
            "vegetation_iou": v_iou,
            "water_iou": water["iou"],
        },
    }
    return with_height_from(params, reference)


def split_half(records):
    ordered = sorted(records, key=lambda r: r["core_id"])
    return ordered[0::2], ordered[1::2]


def honest_gain(records, global_params, threshold_grid, water_grid, k_grid, min_half):
    """Fit on one interleaved half, score on the other, both directions.

    Returns (gain, details) where gain is the mean weighted-score delta of
    per-code params over global params on the held-out halves, or None when
    either half is too small.
    """
    half_a, half_b = split_half(records)
    if len(half_a) < min_half or len(half_b) < min_half:
        return None, None

    deltas = []
    details = []
    for fit_half, eval_half in ((half_a, half_b), (half_b, half_a)):
        local = fit_code_thresholds(fit_half, threshold_grid, water_grid, k_grid, global_params)
        local_score = eval_records(eval_half, local)["weighted_score"]
        global_score = eval_records(eval_half, global_params)["weighted_score"]
        deltas.append(local_score - global_score)
        details.append({
            "n_fit": len(fit_half),
            "n_eval": len(eval_half),
            "per_code_score": local_score,
            "global_score": global_score,
        })
    return float(np.mean(deltas)), details


def fit_per_code(
    records,
    global_params,
    threshold_start=0.40,
    threshold_stop=0.95,
    threshold_step=0.03,
    water_k_grid="0,4,8,12,16,24,32",
    min_samples=6,
    min_half=3,
    min_gain=0.0,
):
    threshold_grid = np.arange(threshold_start, threshold_stop + 1e-9, threshold_step)
    water_grid = threshold_grid
    k_grid = [int(v) for v in str(water_k_grid).split(",") if v.strip()]

    by_code = {}
    for record in records:
        by_code.setdefault(code_of(record["core_id"]), []).append(record)

    codes = {}
    for code in sorted(by_code):
        code_records = by_code[code]
        entry = {"n": len(code_records), "accepted": False, "reason": None}
        if len(code_records) < min_samples:
            entry["reason"] = f"n < {min_samples}"
            codes[code] = entry
            continue

        gain, halves = honest_gain(
            code_records, global_params, threshold_grid, water_grid, k_grid, min_half
        )
        if gain is None:
            entry["reason"] = f"halves < {min_half}"
            codes[code] = entry
            continue

        entry["split_half_gain"] = gain
        entry["split_half_details"] = halves
        if gain <= min_gain:
            entry["reason"] = f"split-half gain {gain:+.4f} <= {min_gain:+.4f}"
            codes[code] = entry
            continue

        params = fit_code_thresholds(
            code_records, threshold_grid, water_grid, k_grid, global_params
        )
        entry["accepted"] = True
        entry["params"] = params
        codes[code] = entry

    return {
        "global": global_params,
        "codes": codes,
        "fit_settings": {
            "threshold_grid": [threshold_start, threshold_stop, threshold_step],
            "water_k_grid": water_k_grid,
            "min_samples": min_samples,
            "min_half": min_half,
            "min_gain": min_gain,
        },
    }


def params_for_code(per_code, code):
    entry = per_code["codes"].get(code)
    if entry and entry.get("accepted"):
        return entry["params"]
    return per_code["global"]


def eval_per_code(records, per_code):
    """OOF metrics with each record scored under its own code's parameters.

    In-sample for accepted codes (the split-half gains are the honest
    estimate); use this as the optimistic bound alongside the global nested
    CV number.
    """
    groups = {}
    for record in records:
        code = code_of(record["core_id"])
        key = code if (per_code["codes"].get(code) or {}).get("accepted") else "__global__"
        groups.setdefault(key, []).append(record)

    rows = []
    for key, group in groups.items():
        params = per_code["global"] if key == "__global__" else per_code["codes"][key]["params"]
        rows.append(eval_records(group, params))
    return aggregate_metrics(rows)


def apply_per_code(pred_dir, per_code, output_dir):
    files = sorted(Path(pred_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    used_codes = set()
    for path in files:
        code = code_of(normalize_core_id(str(path)))
        params = params_for_code(per_code, code)
        if params is not per_code["global"]:
            used_codes.add(code)
        arr = np.load(path).astype(np.float32)
        height = apply_height_channel(arr, params) if params.get("height_affine") else arr[3]
        thresholds = params["thresholds"]
        for channel in range(3):
            arr[channel] = (arr[channel] > thresholds[channel]).astype(np.float32)
        water_mask = apply_water_cc_filter(
            arr[2].astype(bool), int(params.get("water_min_component", 0))
        )
        arr[2] = water_mask.astype(np.float32)
        arr[3] = np.maximum(0.0, height).astype(np.float32)
        np.save(output_dir / path.name, arr)
    return len(files), sorted(used_codes)


def print_fit_report(per_code):
    codes = per_code["codes"]
    accepted = [c for c, e in codes.items() if e["accepted"]]
    print(f"\nPer-code fit: {len(codes)} codes, {len(accepted)} accepted")
    print(f"{'code':<6} {'n':>4} {'gain':>9} {'accepted':>9}  thresholds / K")
    for code in sorted(codes):
        entry = codes[code]
        gain = entry.get("split_half_gain")
        gain_s = f"{gain:+.4f}" if gain is not None else "--"
        if entry["accepted"]:
            t = entry["params"]["thresholds"]
            extra = f"({t[0]:.2f},{t[1]:.2f},{t[2]:.2f}) K={entry['params']['water_min_component']}"
        else:
            extra = entry["reason"]
        print(f"{code:<6} {entry['n']:>4} {gain_s:>9} {str(entry['accepted']):>9}  {extra}")


def cmd_fit(args):
    fold_pattern = str(Path(args.oof_root) / "fold_{fold}")
    records, _ = collect_oof_records(fold_pattern, args.labels_dir, n_folds=args.n_folds)
    print(f"Collected {len(records)} OOF records from {args.oof_root}")

    if args.global_params_json:
        global_params = json.loads(Path(args.global_params_json).read_text())
        if "thresholds" not in global_params:  # allow nested-CV report files
            global_params = global_params["full_fit_params"]
    else:
        print("No --global-params-json given; fitting global params on all OOF records ...")
        global_params = fit_params(records, water_k_grid=args.water_k_grid)

    per_code = fit_per_code(
        records,
        global_params,
        water_k_grid=args.water_k_grid,
        min_samples=args.min_samples,
        min_half=args.min_half,
        min_gain=args.min_gain,
    )
    print_fit_report(per_code)

    print("\nEvaluating OOF under global vs per-code parameters ...")
    global_metrics = eval_records(records, global_params)
    per_code_metrics = eval_per_code(records, per_code)
    per_code["oof_global_metrics"] = global_metrics
    per_code["oof_per_code_metrics_optimistic"] = per_code_metrics
    print(f"global    OOF score: {global_metrics['weighted_score']:.4f}")
    print(f"per-code  OOF score: {per_code_metrics['weighted_score']:.4f} "
          "(optimistic; accepted codes are in-sample)")

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(per_code, indent=2) + "\n")
    print(f"Wrote {output}")


def cmd_apply(args):
    per_code = json.loads(Path(args.params_json).read_text())
    count, used = apply_per_code(args.pred_dir, per_code, args.output_dir)
    print(f"Binarized {count} files to {args.output_dir}")
    print(f"Per-code params used for {len(used)} codes: {', '.join(used) if used else '(none)'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fit", help="Fit per-code thresholds on OOF predictions.")
    pf.add_argument("--oof-root", type=Path, required=True,
                    help="Dir containing fold_{0..N}/split.json + fold_{0..N}/predictions.")
    pf.add_argument("--labels-dir", type=Path, required=True)
    pf.add_argument("--n-folds", type=int, default=5)
    pf.add_argument("--global-params-json", type=Path, default=None,
                    help="Global params JSON (fit_params output or nested-CV report).")
    pf.add_argument("--water-k-grid", default="0,4,8,12,16,24,32")
    pf.add_argument("--min-samples", type=int, default=6,
                    help="Minimum OOF tiles for a code to be considered.")
    pf.add_argument("--min-half", type=int, default=3,
                    help="Minimum tiles per split-half for the honest check.")
    pf.add_argument("--min-gain", type=float, default=0.0,
                    help="Required split-half score gain over global params.")
    pf.add_argument("--output-json", type=Path, required=True)
    pf.set_defaults(func=cmd_fit)

    pa = sub.add_parser("apply", help="Binarize a prediction dir with per-code params.")
    pa.add_argument("--pred-dir", type=Path, required=True)
    pa.add_argument("--params-json", type=Path, required=True)
    pa.add_argument("--output-dir", type=Path, required=True)
    pa.set_defaults(func=cmd_apply)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
