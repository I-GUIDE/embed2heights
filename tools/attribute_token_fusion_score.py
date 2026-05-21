#!/usr/bin/env python
"""Score-based attribution probe for the hybrid token-fusion module.

Same idea as `attribute_token_fusion.py`, but every ablation is scored on the
5-metric leaderboard Score (iou_bld/tree/water + RMSE_bH/vH) instead of training
val_loss. Score = sum(iou_i * w_i) + max(0, 1-RMSE/X)*w (higher is better).

Probes:
  A. LOSO per token source (4) + ALL-tokens-off
  C. Pathway ablation (kill self-attn / kill additive A / kill spatial gate)

(Cheap σ(g)/weight-norm/attention-map data already collected by the val-loss
probe; this tool focuses on the Score reweighting.)

Run:
  python tools/attribute_token_fusion_score.py \\
      --run-dir runs/<exp> \\
      --output  runs/<exp>/attribution_score.json
"""

import argparse
import copy
import json
import os
import sys
from argparse import Namespace

import numpy as np
import rasterio
import torch
import yaml
from tqdm.auto import tqdm

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from core.data.datasets import HEIGHT_NORM_CONSTANT  # noqa: E402
from core.data.training import (  # noqa: E402
    build_train_val_datasets, discover_training_pairs, split_training_pairs,
)
from core.data.discovery import normalize_core_id  # noqa: E402
from core.engine import select_device, seed_everything  # noqa: E402
from core.metrics import (  # noqa: E402
    CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT, WEIGHTS,
    binary_iou, compute_weighted_score, label_gt_mask,
)
from core.models import build_model  # noqa: E402

SOURCE_NAMES = ["terramind_s1", "terramind_s2", "thor_s1", "thor_s2"]
TOKEN_SOURCE_CH = 768


# ----------------- Reuse helpers from val-loss probe -----------------------

from tools.attribute_token_fusion import (  # noqa: E402
    args_from_resolved, build_loaded_model, load_resolved_config,
    TokenMaskedModel, ablate_attention, ablate_additive, ablate_spatial_gate,
    clone_model,
)


# ----------------- Score evaluation -----------------------------------------

def _to_pred_numpy(out_tensor):
    """Match predict.py: copy to CPU float32, un-normalize height channel 3."""
    pred = out_tensor.detach().cpu().numpy().astype(np.float32)
    pred[3] = pred[3] * HEIGHT_NORM_CONSTANT
    return pred


def evaluate_score(model, val_ds, device, *, pred_threshold=0.5, desc=""):
    """Run model on every val sample, compute the 5 leaderboard metrics + Score.

    val_ds is the validation Dataset (PixelMultiTokenEmbeddingDataset). We pull
    the label path from val_ds.file_pairs[i][-1] and load it fresh from disk so
    we hit exactly the same numbers evaluate.py would.
    """
    if np.isscalar(pred_threshold):
        thr_b = thr_v = thr_w = float(pred_threshold)
    else:
        thr_b, thr_v, thr_w = map(float, pred_threshold)

    iou_b, iou_v, iou_w, rmse_b, rmse_v = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(val_ds)), desc=desc, leave=False):
            img, _, _ = val_ds[i]
            # img can be either a tensor or (pixel, token) tuple
            if isinstance(img, (tuple, list)):
                img = tuple(x.unsqueeze(0).to(device, non_blocking=True) for x in img)
            else:
                img = img.unsqueeze(0).to(device, non_blocking=True)
            out = model(img)
            if isinstance(out, (tuple, list)):
                out = out[0]
            pred = _to_pred_numpy(out.squeeze(0))

            # Load matching label
            pair = val_ds.file_pairs[i]
            label_path = pair[-1]
            with rasterio.open(label_path) as src:
                label = src.read().astype(np.float32)
            h = min(pred.shape[1], label.shape[1])
            w = min(pred.shape[2], label.shape[2])
            pred = pred[:, :h, :w]
            label = label[:, :h, :w]

            iou_b.append(binary_iou(pred[CH_BUILDING] > thr_b, label_gt_mask(label, CH_BUILDING)))
            iou_v.append(binary_iou(pred[CH_VEGETATION] > thr_v, label_gt_mask(label, CH_VEGETATION)))
            iou_w.append(binary_iou(pred[CH_WATER] > thr_w, label_gt_mask(label, CH_WATER)))

            bld_mask = label_gt_mask(label, CH_BUILDING)
            veg_mask = label_gt_mask(label, CH_VEGETATION)
            if bld_mask.any():
                diff = pred[CH_HEIGHT][bld_mask] - label[CH_HEIGHT][bld_mask]
                rmse_b.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))
            if veg_mask.any():
                diff = pred[CH_HEIGHT][veg_mask] - label[CH_HEIGHT][veg_mask]
                rmse_v.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))

    def m(arr):
        vals = [v for v in arr if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    metrics = {
        "iou_buildings": m(iou_b),
        "iou_trees": m(iou_v),
        "iou_water": m(iou_w),
        "RMSE_building_height": m(rmse_b),
        "RMSE_vegetation_height": m(rmse_v),
        "n_samples": len(iou_b),
    }
    score, parts = compute_weighted_score(metrics)
    return {"score": score, "metrics": metrics, "score_parts": parts}


# ----------------- Wrap our datasets to expose file_pairs --------------------

def build_val_dataset(args):
    """Construct val_ds with the same token_normalization wiring as training."""
    all_pairs = discover_training_pairs(args)
    train_pairs, val_pairs = split_training_pairs(all_pairs, args)
    _train_ds, val_ds, n_channels = build_train_val_datasets(train_pairs, val_pairs, args)
    return val_ds, n_channels


# ----------------- Main ------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="model_best.pth")
    p.add_argument("--output", default=None)
    p.add_argument("--pred-threshold", type=float, default=0.5)
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    ckpt_path = os.path.join(run_dir, args.ckpt)
    output_path = args.output or os.path.join(run_dir, "attribution_score.json")

    print(f">>> Loading resolved_config from {run_dir}")
    cfg = load_resolved_config(run_dir)
    run_args = args_from_resolved(cfg, run_dir)

    device = select_device()
    seed_everything(run_args.seed)
    print(f">>> Device: {device}")

    print(">>> Building val dataset (with the same normalization as training)")
    val_ds, n_channels = build_val_dataset(run_args)
    print(f"    val samples: {len(val_ds)}  n_channels: {n_channels}")

    print(">>> Loading model + checkpoint")
    model, name = build_loaded_model(run_args, n_channels, ckpt_path, device)
    print(f"    model: {name}  ckpt: {ckpt_path}")

    results = {
        "run_dir": run_dir,
        "ckpt": ckpt_path,
        "experiment_name": run_args.experiment_name,
        "model_type": run_args.model_type,
        "pred_threshold": args.pred_threshold,
    }

    print("\n=== Baseline Score ===")
    baseline = evaluate_score(model, val_ds, device, pred_threshold=args.pred_threshold, desc="baseline")
    print(f"    score = {baseline['score']:.4f}")
    print(f"    iou_bld={baseline['metrics']['iou_buildings']:.4f}  "
          f"iou_tree={baseline['metrics']['iou_trees']:.4f}  "
          f"iou_wat={baseline['metrics']['iou_water']:.4f}  "
          f"RMSE_bH={baseline['metrics']['RMSE_building_height']:.4f}  "
          f"RMSE_vH={baseline['metrics']['RMSE_vegetation_height']:.4f}")
    results["baseline"] = baseline

    n_sources = 4

    print("\n=== Probe A: Leave-one-source-out (LOSO) by Score ===")
    loso = []
    for i in range(n_sources):
        masked = TokenMaskedModel(model, [i]).to(device)
        r = evaluate_score(masked, val_ds, device, pred_threshold=args.pred_threshold,
                          desc=f"mask_{SOURCE_NAMES[i]}")
        d = r["score"] - baseline["score"]
        print(f"    mask {SOURCE_NAMES[i]:>13s}: score={r['score']:.4f} ({d:+.4f})  "
              f"iou_wat={r['metrics']['iou_water']:.4f}  "
              f"RMSE_bH={r['metrics']['RMSE_building_height']:.4f}")
        loso.append({
            "masked_source": SOURCE_NAMES[i],
            "score": r["score"],
            "delta_score": d,
            "metrics": r["metrics"],
            "score_parts": r["score_parts"],
        })
    all_masked = TokenMaskedModel(model, list(range(n_sources))).to(device)
    r_all = evaluate_score(all_masked, val_ds, device, pred_threshold=args.pred_threshold, desc="mask_all")
    d_all = r_all["score"] - baseline["score"]
    print(f"    mask ALL tokens:        score={r_all['score']:.4f} ({d_all:+.4f})")
    results["loso"] = {
        "per_source": loso,
        "all_tokens_zero": {
            "score": r_all["score"], "delta_score": d_all,
            "metrics": r_all["metrics"], "score_parts": r_all["score_parts"],
        },
    }

    print("\n=== Probe C: pathway ablation by Score ===")
    pathways = []
    for label, mut in (
        ("kill_self_attn", ablate_attention),
        ("kill_additive_A", ablate_additive),
        ("kill_spatial_gate", ablate_spatial_gate),
    ):
        m_copy = clone_model(model).to(device)
        mut(m_copy)
        r = evaluate_score(m_copy, val_ds, device, pred_threshold=args.pred_threshold, desc=label)
        d = r["score"] - baseline["score"]
        print(f"    {label:>20s}: score={r['score']:.4f} ({d:+.4f})  "
              f"iou_wat={r['metrics']['iou_water']:.4f}  "
              f"RMSE_bH={r['metrics']['RMSE_building_height']:.4f}")
        pathways.append({
            "ablation": label,
            "score": r["score"],
            "delta_score": d,
            "metrics": r["metrics"],
            "score_parts": r["score_parts"],
        })
    results["pathway_ablation"] = pathways

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n>>> Wrote {output_path}")


if __name__ == "__main__":
    main()
