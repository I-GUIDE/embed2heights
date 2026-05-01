"""Inspect the Tessera trunk-fusion gate of a TesseraIoUFusionGatedLightUNet.

Tests three predictions about Y's gate behavior:
  1. Gate is mostly closed to Tessera on building-positive pixels.
  2. Gate is more open to Tessera on water- and vegetation-positive pixels.
  3. Gate output meaningfully depends on the Tessera half of its input
     (zeroing it should change the gate distribution).

For the tied gate: fused = G * AE + (1 - G) * TES, where G = sigmoid(gate_conv).
Tessera weight = 1 - G. We report mean Tessera weight per class label.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from core.dataset import (
    MultiPixelEmbeddingDataset,
    find_multisource_file_pairs,
    load_split,
)
from core.model import build_model


def _load_training_cfg(exp_dir):
    with open(os.path.join(exp_dir, "training_params.json")) as f:
        return json.load(f)


def _build_y_model(exp_dir, n_channels, device):
    cfg = _load_training_cfg(exp_dir)
    model, _ = build_model(
        cfg["model_type"],
        n_channels=n_channels,
        n_classes=4,
        tessera_presence_ch=cfg.get("tessera_presence_ch", 0),
        tessera_hidden_ch=cfg.get("tessera_hidden_ch"),
        tessera_hidden_depth=cfg.get("tessera_hidden_depth", 0),
        height_specialist_depth=cfg.get("height_specialist_depth", 0),
        lightunet_base_ch=cfg.get("lightunet_base_ch", 32),
        lightunet_norm_kind=cfg.get("lightunet_norm_kind", "bn"),
        height_gate_source=cfg.get("height_gate_source", "alpha"),
        height_hidden_ch=cfg.get("height_hidden_ch"),
        height_trunk_depth=cfg.get("height_trunk_depth", 2),
        height_independent_branches=cfg.get("height_independent_branches", False),
        height_head_kind=cfg.get("height_head_kind", "linear"),
        height_n_bins=cfg.get("height_n_bins", 64),
        height_bin_max_m=cfg.get("height_bin_max_m", 80.0),
        gate_mode=cfg.get("gate_mode", "simple"),
        gate_untied=cfg.get("gate_untied", False),
        gate_init_bias=cfg.get("gate_init_bias", 4.0),
        modality_dropout=cfg.get("modality_dropout", 0.0),
    )
    state = torch.load(os.path.join(exp_dir, "model_best.pth"), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def _percentiles(arr, qs=(5, 25, 50, 75, 95)):
    return {q: float(np.percentile(arr, q)) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--alpha-dir", default="/u/dingqi2/workspace/esa/data/train/alphaearth_emb")
    ap.add_argument("--tessera-dir", default="/u/dingqi2/workspace/esa/data/train/tessera_emb")
    ap.add_argument("--labels-dir", default="/u/dingqi2/workspace/esa/data/train/labels")
    ap.add_argument("--split-file", default=os.path.join(PROJECT_DIR, "splits/split.json"))
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=80,
                    help="How many val patches to summarize over.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_pairs = find_multisource_file_pairs(args.alpha_dir, args.tessera_dir, args.labels_dir)
    _, val_pairs = load_split(args.split_file, all_pairs)
    val_pairs = val_pairs[: args.n_samples]
    ds = MultiPixelEmbeddingDataset(val_pairs, patch_size=args.patch_size, is_train=False)

    sample_img, _, _ = ds[0]
    n_channels = sample_img.shape[0]
    model, cfg = _build_y_model(args.exp_dir, n_channels, device)
    print(f"Loaded {cfg['model_type']} from {args.exp_dir}")
    print(f"  gate_mode={cfg.get('gate_mode')!r} untied={cfg.get('gate_untied')} "
          f"init_bias={cfg.get('gate_init_bias')} base_ch={cfg.get('lightunet_base_ch')}")
    if cfg.get("gate_untied"):
        print("WARN: untied gate — this script reports tied-gate stats only.")
        return

    captured = {}

    def _hook(_module, inp, out):
        captured["raw"] = out.detach()

    handle = model.gate_conv.register_forward_hook(_hook)

    sums = {"build": 0.0, "tree": 0.0, "water": 0.0, "bg": 0.0, "all": 0.0}
    counts = {k: 0 for k in sums}
    g_samples = []
    p_samples = []

    print(f"\nRunning {len(ds)} val patches...")
    with torch.no_grad():
        for i in range(len(ds)):
            img, target, _ = ds[i]
            img = img.unsqueeze(0).to(device)
            label = target[:3]  # (3, H, W) presence channels
            _ = model(img)
            raw = captured["raw"][0]                 # (C, H, W) gate logits
            g = torch.sigmoid(raw).mean(dim=0)       # mean over channels -> (H, W)
            tess_weight = (1.0 - g).cpu().numpy()    # how much Tessera flows in

            lb = label[0].numpy() > 0
            lt = label[1].numpy() > 0
            lw = label[2].numpy() > 0
            lbg = ~(lb | lt | lw)

            for key, mask in (("build", lb), ("tree", lt), ("water", lw), ("bg", lbg)):
                if mask.any():
                    sums[key] += float(tess_weight[mask].sum())
                    counts[key] += int(mask.sum())
            sums["all"] += float(tess_weight.sum())
            counts["all"] += int(tess_weight.size)

            if i < 16:
                # Save a sub-sample for percentile distributions and for the
                # ablation test below (cheap, capped to keep memory low).
                g_samples.append(g.cpu().numpy().reshape(-1))
                # AE-only presence prob (for AE-uncertainty correlation)
                # The model's `aux` would give us alpha_presence_prob, but
                # capturing the gate output is cheaper; correlate gate openness
                # with AE-prob distance from 0.5 instead.
                with torch.no_grad():
                    out = model(img)
                    p_samples.append(out[0, :3].cpu().numpy())

    handle.remove()

    print("\n=== Tessera weight (1 - sigmoid(gate)) per pixel-class ===")
    print(f"  All val pixels:         mean = {sums['all']/counts['all']:.4f}")
    for key, label in (("build", "Building-positive"),
                       ("tree",  "Tree-positive    "),
                       ("water", "Water-positive   "),
                       ("bg",    "Background       ")):
        if counts[key]:
            mean = sums[key] / counts[key]
            print(f"  {label} pixels: mean = {mean:.4f}   (n={counts[key]:,})")
        else:
            print(f"  {label} pixels: NONE FOUND")

    g_all = np.concatenate(g_samples)
    print("\n=== Gate sigmoid distribution (G; G=1 means AE-only) ===")
    print(f"  mean={g_all.mean():.4f}  std={g_all.std():.4f}")
    print(f"  percentiles: " + ", ".join(
        f"p{q}={v:.3f}" for q, v in _percentiles(g_all).items()))
    print(f"  fraction G > 0.95 (Tessera basically off): {(g_all > 0.95).mean():.3f}")
    print(f"  fraction G < 0.50 (Tessera-dominant):       {(g_all < 0.50).mean():.3f}")

    # Ablation: zero the Tessera half of the gate input. If the gate truly
    # depends on Tessera content, the mean output should change visibly.
    print("\n=== Ablation: zero the Tessera half of the gate input ===")
    img, _, _ = ds[0]
    img = img.unsqueeze(0).to(device)
    alpha = img[:, : model.alpha_channels]
    tessera = img[:, model.alpha_channels:]
    with torch.no_grad():
        ae_feat = model.alpha_unet.forward_features(alpha)
        tes_feat_real = model.tessera_feature_stem(tessera)
        tes_feat_zero = torch.zeros_like(tes_feat_real)
        g_real = torch.sigmoid(
            model.gate_conv(torch.cat([ae_feat, tes_feat_real], dim=1))
        )
        g_zero = torch.sigmoid(
            model.gate_conv(torch.cat([ae_feat, tes_feat_zero], dim=1))
        )
    diff = (g_real - g_zero).abs().mean().item()
    print(f"  mean |G_real - G_with_zeroed_tessera| = {diff:.4f}")
    print(f"  G_real  mean={g_real.mean().item():.4f}")
    print(f"  G_zero  mean={g_zero.mean().item():.4f}")
    if diff < 0.005:
        print("  --> gate is essentially independent of Tessera content (FAIL prediction 3)")
    else:
        print("  --> gate output depends on Tessera content (prediction 3 SUPPORTED)")


if __name__ == "__main__":
    main()
