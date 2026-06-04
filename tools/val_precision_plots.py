#!/usr/bin/env python3
"""
Run the validation loader with a trained checkpoint and save:

- confusion_landcover.png — 2×2 confusion matrices (building / vegetation / water)
- height_density_building_veg.png — hexbin of (GT, pred) height in meters

Reuses the same train/val split and height normalization as ``train.py`` when
``training_params.json`` + optional ``--split-file`` match the training run.

After ``train.py``, best checkpoints by leaderboard weighted score are listed in
``<exp>/topk_manifest.json``; pass ``--checkpoint`` to one of those ``path`` values.

Example::

  python tools/val_precision_plots.py \\
    --experiment-name my_run \\
    --base-dir runs \\
    --split-file runs/my_run/split.json

  python tools/val_precision_plots.py \\
    --experiment-name my_run \\
    --base-dir runs \\
    --checkpoint runs/my_run/topk_pool/cand_e0012_ws0.XXXXXX.pth
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import train as train_mod  # noqa: E402
from core.data.height_stats import denormalize_height_numpy  # noqa: E402
from core.data.datasets import HEIGHT_NORM_CONSTANT  # noqa: E402
from core.engine import select_device  # noqa: E402
from core.metrics import (  # noqa: E402
    CH_BUILDING,
    CH_VEGETATION,
    LABEL_THRESHOLD,
    pred_threshold_to_triplet,
)
from core.models import build_model  # noqa: E402
from predict import load_training_config, model_kwargs_from_run_config  # noqa: E402


def outputs_targets_to_numpy_meters(
    outputs,
    targets: torch.Tensor,
    *,
    height_norm_stats: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(outputs, dict):
        t = outputs["out"]
    else:
        t = outputs
    pred = t.detach().float().cpu().numpy()
    lab = targets.detach().float().cpu().numpy()
    b = pred.shape[0]
    pred = pred.copy()
    lab = lab.copy()
    for i in range(b):
        if height_norm_stats is not None:
            pred[i, 3] = denormalize_height_numpy(pred[i, 3], height_norm_stats).astype(np.float32)
            lab[i, 3] = denormalize_height_numpy(lab[i, 3], height_norm_stats).astype(np.float32)
        else:
            pred[i, 3] = (pred[i, 3] * HEIGHT_NORM_CONSTANT).astype(np.float32)
            lab[i, 3] = (lab[i, 3] * HEIGHT_NORM_CONSTANT).astype(np.float32)
    return pred, lab


def collect_val_pixel_tables(
    model,
    val_loader,
    device,
    *,
    height_norm_stats,
    pred_thr,
    label_thr: float,
    max_height_pairs: int,
    use_amp: bool,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    thr_b, thr_v, thr_w = pred_threshold_to_triplet(pred_thr)
    thrs = [thr_b, thr_v, thr_w]
    cms = [np.zeros((2, 2), dtype=np.int64) for _ in range(3)]
    h_true_list: List[np.ndarray] = []
    h_pred_list: List[np.ndarray] = []
    n_stored = 0

    model.eval()
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for imgs, targets, masks in tqdm(val_loader, desc="val plots", leave=False):
            imgs = train_mod.move_to_device(imgs, device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            masks = masks.to(device, non_blocking=non_blocking)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = train_mod.forward_for_training(model, imgs)
            pred_b, lab_b = outputs_targets_to_numpy_meters(
                outputs, targets, height_norm_stats=height_norm_stats
            )
            mb = pred_b.shape[0]
            m_glob = masks[:, 0].detach().cpu().numpy() > 0.5
            m_hgt = masks[:, 1].detach().cpu().numpy() > 0.5
            for i in range(mb):
                gb = m_glob[i]
                hh = m_hgt[i]
                if not gb.any():
                    continue
                for c in range(3):
                    pred_pos = pred_b[i, c] > thrs[c]
                    lab_pos = lab_b[i, c] > label_thr
                    p_flat = pred_pos[gb].ravel().astype(np.int64)
                    l_flat = lab_pos[gb].ravel().astype(np.int64)
                    if p_flat.size:
                        cm = confusion_matrix(l_flat, p_flat, labels=[0, 1])
                        cms[c] += cm.astype(np.int64)
                land = (
                    (lab_b[i, CH_BUILDING] > label_thr)
                    | (lab_b[i, CH_VEGETATION] > label_thr)
                ) & hh
                if land.any() and n_stored < max_height_pairs:
                    yt = lab_b[i, 3][land].astype(np.float64)
                    yp = pred_b[i, 3][land].astype(np.float64)
                    take = min(yt.size, max_height_pairs - n_stored)
                    if take > 0:
                        h_true_list.append(yt[:take])
                        h_pred_list.append(yp[:take])
                        n_stored += take

    if h_true_list:
        height_y_true = np.concatenate(h_true_list, axis=0)
        height_y_pred = np.concatenate(h_pred_list, axis=0)
    else:
        height_y_true = np.array([], dtype=np.float64)
        height_y_pred = np.array([], dtype=np.float64)
    return cms, height_y_true, height_y_pred


def plot_confusion_figure(
    cms: List[np.ndarray],
    out_path: str,
    *,
    class_names: Tuple[str, ...] = ("Building", "Vegetation", "Water"),
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    ims = []
    for ax, cm, name in zip(axes, cms, class_names):
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        norm = cm.astype(np.float64) / row_sums
        im = ax.imshow(norm, vmin=0.0, vmax=1.0, cmap="Blues")
        ims.append(im)
        ax.set_title(f"{name}\n(0=absent, 1=present)")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True (GT)")
        for (j, i), v in np.ndenumerate(cm):
            ax.text(i, j, f"{int(v)}\n({norm[j, i]:.2f})", ha="center", va="center", fontsize=8)
    fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.72, label="Recall-normalized row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_height_density_hexbin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    *,
    max_m: float = 80.0,
) -> None:
    import matplotlib.pyplot as plt

    if y_true.size == 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "No height pixels\n(building ∪ veg)", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6, 5.5))
    hb = ax.hexbin(
        y_true,
        y_pred,
        gridsize=60,
        cmap="viridis",
        mincnt=1,
        extent=(0, max_m, 0, max_m),
    )
    ax.plot([0, max_m], [0, max_m], "r--", lw=1.0, label="y = x")
    ax.set_xlabel("Ground-truth height (m)")
    ax.set_ylabel("Predicted height (m)")
    ax.set_title("Height density (building ∪ vegetation, height-valid pixels)")
    ax.set_xlim(0, max_m)
    ax.set_ylim(0, max_m)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def apply_training_config_to_args(args, cfg: Dict[str, Any]) -> None:
    for k, v in cfg.items():
        if k.startswith("_") or v is None:
            continue
        if hasattr(args, k):
            setattr(args, k, v)


def parse_tool_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--base-dir", default=train_mod.DEFAULTS["output_dir"])
    p.add_argument("--checkpoint", default=None, help="Defaults to model_best.pth in the experiment dir.")
    p.add_argument("--output-subdir", default="val_precision_plots", help="Subfolder under the experiment dir.")
    p.add_argument("--split-file", default=None, help="Override split JSON (should match training).")
    p.add_argument("--batch-size", type=int, default=None, help="Override DataLoader batch size.")
    p.add_argument("--val-pred-threshold", type=float, default=0.5)
    p.add_argument("--val-thresholds", type=float, nargs=3, default=None, metavar=("BLD", "VEG", "WAT"))
    p.add_argument("--analysis-height-max-m", type=float, default=80.0)
    p.add_argument("--analysis-max-pixels", type=int, default=2_000_000)
    p.add_argument("--amp", action="store_true", help="Use CUDA autocast during forward.")
    return p.parse_args()


def main() -> None:
    targs = parse_tool_args()
    exp_dir = os.path.join(targs.base_dir, targs.experiment_name)
    out_dir = os.path.join(exp_dir, targs.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    cfg, _ = load_training_config(exp_dir)
    args = train_mod.parse_args([])
    apply_training_config_to_args(args, cfg)
    if targs.split_file is not None:
        args.split_file = targs.split_file
    if targs.batch_size is not None:
        args.batch_size = targs.batch_size

    device = select_device()
    use_amp = targs.amp and device.type == "cuda"

    print("--- Building val loader (same logic as train.make_dataloaders) ---")
    _, val_loader, _, _, n_channels, height_stats = train_mod.make_dataloaders(args, device)

    pred_thr = (
        tuple(targs.val_thresholds)
        if targs.val_thresholds is not None
        else float(targs.val_pred_threshold)
    )

    ckpt = targs.checkpoint or os.path.join(exp_dir, "model_best.pth")
    model, selected_model = build_model(
        args.model_type,
        n_channels,
        n_classes=4,
        **{**model_kwargs_from_run_config(cfg), "height_norm_stats": height_stats},
    )
    model = model.to(device)
    state = torch.load(ckpt, map_location=device)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    state = {k.replace(".norm.", ".bn."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model {selected_model} | checkpoint {ckpt} | channels {n_channels}")

    cms, h_true, h_pred = collect_val_pixel_tables(
        model,
        val_loader,
        device,
        height_norm_stats=height_stats,
        pred_thr=pred_thr,
        label_thr=LABEL_THRESHOLD,
        max_height_pairs=targs.analysis_max_pixels,
        use_amp=use_amp,
    )

    plot_confusion_figure(cms, os.path.join(out_dir, "confusion_landcover.png"))
    plot_height_density_hexbin(
        h_true,
        h_pred,
        os.path.join(out_dir, "height_density_building_veg.png"),
        max_m=targs.analysis_height_max_m,
    )
    summary = {
        "experiment_name": targs.experiment_name,
        "checkpoint": ckpt,
        "pred_threshold": pred_thr,
        "height_norm_transform": cfg.get("height_norm_transform"),
        "confusion_counts": [c.tolist() for c in cms],
        "height_n_points": int(h_true.size),
    }
    with open(os.path.join(out_dir, "plot_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote plots and plot_summary.json under {out_dir}")


if __name__ == "__main__":
    main()
