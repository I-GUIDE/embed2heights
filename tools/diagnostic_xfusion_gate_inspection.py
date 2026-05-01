"""Inspect the TerraMind decoder64 gate and the Tessera trunk gate of an
xfusion_025 / xfusion_026 model.

For each gate (tied):
    fused = G * primary + (1 - G) * secondary,   G = sigmoid(gate_conv(...))
- TerraMind gate at decoder64: primary = AE trunk, secondary = TerraMind-S2 proj.
  Reported "TerraMind weight" = mean(1 - G).
- Tessera trunk gate at end-of-decoder (xfusion_026 only): primary = AE-trunk
  after TerraMind fusion, secondary = Tessera trunk-stem feature.
  Reported "Tessera weight" = mean(1 - G).

Reports:
  - global mean / per-channel / spatial percentile distribution
  - per-label-class breakdown (build / tree / water / background)
  - content-zeroing ablation: zero the secondary half of the gate input and
    check whether the gate's mean changes (tests gate-on-content dependency)

Both gates are warm-started so the secondary weight ≈ 0.018 at step 0; how
far they drift from that tells us how much the model actually relies on the
secondary modality.
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
    PixelTokenEmbeddingDataset,
    find_trisource_file_pairs,
    load_split,
)
from core.model import build_model


def _load_cfg(exp_dir):
    with open(os.path.join(exp_dir, "training_params.json")) as f:
        return json.load(f)


def _build(exp_dir, n_channels, device):
    cfg = _load_cfg(exp_dir)
    model, _ = build_model(
        cfg["model_type"],
        n_channels=n_channels,
        n_classes=4,
        tessera_presence_ch=cfg.get("tessera_presence_ch", 16),
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


def _percentiles(arr, qs=(1, 5, 25, 50, 75, 95, 99)):
    return ", ".join(f"p{q}={float(np.percentile(arr, q)):.3f}" for q in qs)


def _summarize_gate(name, sec_label, gate_logits_chw, label_3hw):
    """Print a per-class summary for one gate.

    gate_logits_chw : torch tensor (C, H, W) of pre-sigmoid gate logits
                      stacked over batches (concat along H or batch flatten).
    label_3hw       : numpy array (3, H, W) of presence labels matching the
                      same H, W (we only use the binary presence channels).
    """
    print(f"\n--- {name} ---")
    g_chw = torch.sigmoid(gate_logits_chw)            # (C, H, W)
    g_hw = g_chw.mean(dim=0)                          # mean over channels
    sec_w = (1.0 - g_hw).cpu().numpy()                # secondary weight per pixel
    print(f"  global mean SECONDARY weight ({sec_label}) = {sec_w.mean():.4f}")
    print(f"  global mean PRIMARY  weight (AE trunk)     = {g_hw.mean().item():.4f}")
    print(f"  per-pixel SECONDARY weight percentiles: {_percentiles(sec_w.reshape(-1))}")
    # per-channel mean (averaged over H, W). Tells us whether SOME channels
    # carry most of the secondary contribution while others stay AE-only.
    per_ch = (1.0 - g_chw).mean(dim=(1, 2)).cpu().numpy()
    print(f"  per-channel SECONDARY weight (mean over space):"
          f" mean={per_ch.mean():.4f} min={per_ch.min():.4f} max={per_ch.max():.4f}"
          f" std={per_ch.std():.4f}")

    # per-class breakdown
    lb = label_3hw[0] > 0
    lt = label_3hw[1] > 0
    lw = label_3hw[2] > 0
    lbg = ~(lb | lt | lw)
    for key, label, mask in (
        ("build", "Building", lb),
        ("tree",  "Tree    ", lt),
        ("water", "Water   ", lw),
        ("bg",    "BG      ", lbg),
    ):
        if mask.any():
            mean_sec = float(sec_w[mask].mean())
            print(f"  {label} pixels: secondary weight mean = {mean_sec:.4f}   (n={int(mask.sum()):,})")
        else:
            print(f"  {label} pixels: NONE FOUND")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--alpha-dir", default="/u/dingqi2/workspace/esa/data/train/alphaearth_emb")
    ap.add_argument("--tessera-dir", default="/u/dingqi2/workspace/esa/data/train/tessera_emb")
    ap.add_argument("--token-dir", default="/u/dingqi2/workspace/esa/data/train/terramind_s2_emb")
    ap.add_argument("--labels-dir", default="/u/dingqi2/workspace/esa/data/train/labels")
    ap.add_argument("--split-file", default=os.path.join(PROJECT_DIR, "splits/split.json"))
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=80,
                    help="How many val patches to summarize over.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_pairs = find_trisource_file_pairs(
        args.alpha_dir, args.tessera_dir, args.token_dir, args.labels_dir
    )
    _, val_pairs = load_split(args.split_file, all_pairs)
    val_pairs = val_pairs[: args.n_samples]
    ds = PixelTokenEmbeddingDataset(val_pairs, patch_size=args.patch_size,
                                    scale_factor=16, is_train=False)

    sample_img, _, _ = ds[0]
    pixel_ch = sample_img[0].shape[0]
    token_ch = sample_img[1].shape[0]
    n_channels = (pixel_ch, token_ch)

    model, cfg = _build(args.exp_dir, n_channels, device)
    print(f"Loaded model_type={cfg['model_type']}")
    print(f"  gate_mode={cfg.get('gate_mode')!r} untied={cfg.get('gate_untied')}"
          f" init_bias={cfg.get('gate_init_bias')} base_ch={cfg.get('lightunet_base_ch')}")
    print(f"  tessera_presence_ch={cfg.get('tessera_presence_ch')}")

    if cfg.get("gate_untied"):
        print("WARN: untied gate not supported by this script.")
        return

    # Identify which gates exist on this model
    has_tm_gate = (
        getattr(model, "decoder64_adapter", None) is not None
        and hasattr(model.decoder64_adapter, "gate_conv")
    )
    has_tessera_gate = (
        getattr(model, "trunk_gate_conv", None) is not None
        and getattr(model, "tessera_trunk_stem", None) is not None
    )
    print(f"  TerraMind decoder64 sigmoid gate: {'YES' if has_tm_gate else 'no'}")
    print(f"  Tessera trunk sigmoid gate:       {'YES' if has_tessera_gate else 'no'}")
    if not (has_tm_gate or has_tessera_gate):
        print("Nothing to inspect. Exiting.")
        return

    captured = {}

    def make_hook(key):
        def _hook(_module, _inp, out):
            captured.setdefault(key, []).append(out.detach())
        return _hook

    handles = []
    if has_tm_gate:
        handles.append(model.decoder64_adapter.gate_conv.register_forward_hook(
            make_hook("tm")))
    if has_tessera_gate:
        handles.append(model.trunk_gate_conv.register_forward_hook(
            make_hook("tes")))

    # Accumulate per-class sums + spatial samples for percentile reporting
    sec_sums = {"tm": {}, "tes": {}}
    sec_counts = {"tm": {}, "tes": {}}
    per_ch_accum = {"tm": [], "tes": []}
    sample_caps = {"tm": [], "tes": []}
    sample_label_caps = []

    print(f"\nRunning {len(ds)} val patches...")
    with torch.no_grad():
        for i in range(len(ds)):
            img, target, _ = ds[i]
            pixel = img[0].unsqueeze(0).to(device)
            token = img[1].unsqueeze(0).to(device)
            label = target[:3].numpy()                # (3, H, W)
            captured.clear()
            _ = model((pixel, token))

            for key in ("tm", "tes"):
                if key not in captured:
                    continue
                logits = captured[key][0][0]          # (C, h, w) for this patch
                g = torch.sigmoid(logits)             # (C, h, w)
                sec_chw = 1.0 - g                     # secondary weight
                # Spatial-mean per channel (track distribution across patches/channels)
                per_ch_accum[key].append(sec_chw.mean(dim=(1, 2)).cpu().numpy())
                # Channel-mean per pixel for per-class breakdown
                sec_hw = sec_chw.mean(dim=0).cpu().numpy()    # (h, w)
                # Resize label to gate spatial size if needed
                if sec_hw.shape != label.shape[1:]:
                    # Both gates here output at AE trunk's full 256x256, so
                    # this should rarely fire; if it does we just resize labels
                    # nearest-neighbor for class accounting.
                    import torch.nn.functional as F
                    lbl_t = torch.from_numpy(label).unsqueeze(0).float()
                    lbl_t = F.interpolate(lbl_t, size=sec_hw.shape, mode="nearest")
                    lab_use = lbl_t[0].numpy()
                else:
                    lab_use = label

                lb = lab_use[0] > 0
                lt = lab_use[1] > 0
                lw = lab_use[2] > 0
                lbg = ~(lb | lt | lw)
                for cls, mask in (("build", lb), ("tree", lt),
                                  ("water", lw), ("bg", lbg), ("all", np.ones_like(lb))):
                    if mask.any():
                        sec_sums[key][cls] = sec_sums[key].get(cls, 0.0) + float(sec_hw[mask].sum())
                        sec_counts[key][cls] = sec_counts[key].get(cls, 0) + int(mask.sum())

                if i < 8:
                    sample_caps[key].append(sec_hw)

            if i < 8:
                sample_label_caps.append(label)

    for h in handles:
        h.remove()

    # Report
    for key, name, sec_label in (
        ("tm",  "TerraMind decoder64 gate", "TerraMind weight"),
        ("tes", "Tessera trunk gate",       "Tessera weight"),
    ):
        if key not in sec_sums or not sec_sums[key]:
            continue
        print(f"\n========== {name} ==========")
        print(f"  Across all val pixels (over {sec_counts[key].get('all', 0):,} pixels):")
        for cls, label_str in (
            ("all",   "ALL pixels      "),
            ("build", "Building        "),
            ("tree",  "Tree            "),
            ("water", "Water           "),
            ("bg",    "Background      "),
        ):
            n = sec_counts[key].get(cls, 0)
            if n:
                mean = sec_sums[key][cls] / n
                print(f"    {label_str}: mean {sec_label} = {mean:.4f}   (n={n:,})")

        per_ch = np.concatenate(per_ch_accum[key])
        print(f"  Per-channel SECONDARY weight stats (over channels x patches):")
        print(f"    mean={per_ch.mean():.4f}  std={per_ch.std():.4f}")
        print(f"    {_percentiles(per_ch)}")
        print(f"    fraction channels with secondary weight > 0.10 : {(per_ch > 0.10).mean():.3f}")
        print(f"    fraction channels with secondary weight > 0.20 : {(per_ch > 0.20).mean():.3f}")
        print(f"    fraction channels with secondary weight < 0.02 (essentially off) : {(per_ch < 0.02).mean():.3f}")

    # Content-zero ablation per gate
    print("\n========== Content-zeroing ablation ==========")
    img, _, _ = ds[0]
    pixel = img[0].unsqueeze(0).to(device)
    token = img[1].unsqueeze(0).to(device)
    alpha = pixel[:, : model.alpha_channels]
    tessera = pixel[:, model.alpha_channels:]
    with torch.no_grad():
        token_pyr = model.token_pyramid(token)

        x1 = model.alpha_unet.inc(alpha)
        x2 = model.alpha_unet.down1(x1)
        x3 = model.alpha_unet.down2(x2)
        x4 = model.alpha_unet.down3(x3)
        feat = model.alpha_unet.up1(x4)
        feat = torch.cat([x3, feat], dim=1)
        feat = model.alpha_unet.conv1(feat)

        if has_tm_gate:
            tok_proj = model.decoder64_adapter.proj(
                torch.nn.functional.interpolate(
                    token_pyr[64], size=feat.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            )
            g_real = torch.sigmoid(
                model.decoder64_adapter.gate_conv(
                    torch.cat([feat, tok_proj], dim=1))
            )
            g_zero = torch.sigmoid(
                model.decoder64_adapter.gate_conv(
                    torch.cat([feat, torch.zeros_like(tok_proj)], dim=1))
            )
            tm_real = (1.0 - g_real).mean().item()
            tm_zero = (1.0 - g_zero).mean().item()
            diff = abs(tm_real - tm_zero)
            print(f"  TerraMind gate, mean TerraMind weight:")
            print(f"    real       = {tm_real:.4f}")
            print(f"    zero-token = {tm_zero:.4f}   |diff|={diff:.4f}")
            verdict = ("gate is essentially independent of TerraMind content"
                       if diff < 0.005 else "gate output depends on TerraMind content")
            print(f"    --> {verdict}")

            # apply to feat for downstream Tessera ablation
            from core.model import _apply_fusion_gate
            feat = _apply_fusion_gate(
                model.decoder64_adapter.gate_conv, feat, tok_proj,
                untied=model.gate_untied,
            )

        # decode the rest of the U-Net so we hit the trunk gate input
        feat = model.alpha_unet.up2(feat)
        feat = torch.cat([x2, feat], dim=1)
        feat = model.alpha_unet.conv2(feat)
        feat = model.alpha_unet.up3(feat)
        feat = torch.cat([x1, feat], dim=1)
        feat = model.alpha_unet.conv3(feat)

        if has_tessera_gate:
            tes_feat = model.tessera_trunk_stem(tessera)
            g_real = torch.sigmoid(
                model.trunk_gate_conv(torch.cat([feat, tes_feat], dim=1))
            )
            g_zero = torch.sigmoid(
                model.trunk_gate_conv(torch.cat([feat, torch.zeros_like(tes_feat)], dim=1))
            )
            tes_real = (1.0 - g_real).mean().item()
            tes_zero = (1.0 - g_zero).mean().item()
            diff = abs(tes_real - tes_zero)
            print(f"  Tessera trunk gate, mean Tessera weight:")
            print(f"    real         = {tes_real:.4f}")
            print(f"    zero-tessera = {tes_zero:.4f}   |diff|={diff:.4f}")
            verdict = ("gate is essentially independent of Tessera content"
                       if diff < 0.005 else "gate output depends on Tessera content")
            print(f"    --> {verdict}")


if __name__ == "__main__":
    main()
