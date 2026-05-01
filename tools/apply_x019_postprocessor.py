"""Apply the learned xfusion_019 OOF post-processing parameters to predictions.

This materializes continuous predictions with the learned height affine applied.
Presence thresholds and water connected-component filtering are intentionally
left to tools/make_submission.py so the same output directory can be packaged
with different threshold policies.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def load_params(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report["full_fit_params"]


def apply_height_affine(arr, params):
    if not params.get("height_affine", False):
        return arr
    hp = params["height_affine_params"]
    b = hp["building"]
    v = hp["vegetation"]
    height = arr[3]
    h_b = np.maximum(0.0, b["a"] * height + b["b"])
    h_v = np.maximum(0.0, v["a"] * height + v["b"])
    p_b = np.clip(arr[0], 0.0, 1.0)
    p_v = np.clip(arr[1], 0.0, 1.0)
    fg = 1.0 - (1.0 - p_b) * (1.0 - p_v)
    denom = p_b + p_v + 1e-6
    h_fg = (p_b * h_b + p_v * h_v) / denom
    out = arr.astype(np.float32, copy=True)
    out[3] = np.maximum(0.0, fg * h_fg + (1.0 - fg) * height).astype(np.float32)
    out[:3] = np.clip(out[:3], 0.0, 1.0)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--report-json",
        type=Path,
        default=Path("runs/xfusion_019_oof_postprocess/x019_oof_postprocess_report.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    params = load_params(args.report_json)
    files = sorted(args.pred_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {args.pred_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(files, desc="Applying x019 postprocess"):
        arr = np.load(path).astype(np.float32)
        out = apply_height_affine(arr, params)
        np.save(args.output_dir / path.name, out)
    print(f"Wrote {len(files)} files to {args.output_dir}")
    print(
        "Use thresholds "
        f"{params['thresholds'][0]:.3f} {params['thresholds'][1]:.3f} {params['thresholds'][2]:.3f} "
        f"and water K={params['water_min_component']} when packaging."
    )


if __name__ == "__main__":
    main()
