"""Test whether AE trunk feature and TerraMind-token-projected feature are
spatially aligned at the decoder64 fusion point.

Why: TransFuse's BiFusion uses a Hadamard product (W1*t) ⊙ (W2*g) between
transformer and CNN streams at matching (h,w). That trick only helps if AE
trunk feature at (h,w) and TM projection at (h,w) actually share location-
specific information. If TM tokens are globally-mixed (memory:
project_terramind_fusion.md), the per-position Hadamard collapses to noise
and channel-only modulation is the only viable form.

Test: hook the decoder64 adapter, capture
    target     = AE trunk feature (B, c3, 64, 64)        # adapter input
    token_proj = adapter.proj(interp(token_pyr[64]))     # adapter output of proj

Stack across N val patches into (N*64*64, c3) row matrices A (AE) and T (TM).

Three metrics, all comparing aligned vs spatially-shuffled pairings:
  1. Per-position cosine similarity (mean ± std)
  2. Linear probe R²: fit T = W A + b on train rows, eval on holdout
       aligned: T's row stays paired with A's row
       shuffled: T's rows shuffled across spatial positions (within each batch
                 sample, to avoid sample-level domain shift confounds)
  3. Per-channel Pearson correlation (matched vs shuffled, mean over channels)

Decision rule:
  aligned_R2 - shuffled_R2 > 0.05  -> spatial alignment exists; Hadamard
                                       has signal beyond channel-only.
  aligned_R2 - shuffled_R2 < 0.02  -> no spatial alignment; channel-only
                                       (FiLM-class) is the ceiling here.
  in between                       -> ambiguous; consider channel-only first.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

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
    kwargs = dict(
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
    )
    for k in ("gate_mode", "gate_untied", "gate_init_bias",
              "modality_dropout", "token_gate_kind", "fusion_points"):
        if k in cfg and cfg[k] is not None:
            kwargs[k] = cfg[k]
    model, _ = build_model(cfg["model_type"], **kwargs)
    state = torch.load(os.path.join(exp_dir, "model_best.pth"), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def _linear_probe_r2(A_tr, T_tr, A_te, T_te, ridge=1.0):
    """Closed-form ridge regression. Returns mean per-channel R^2 on test."""
    # A_tr: (N, Da), T_tr: (N, Dt). Center, fit, eval.
    mu_a, mu_t = A_tr.mean(0, keepdim=True), T_tr.mean(0, keepdim=True)
    Ac, Tc = A_tr - mu_a, T_tr - mu_t
    # W = (Ac^T Ac + ridge*I)^-1 Ac^T Tc
    Da = Ac.shape[1]
    G = Ac.T @ Ac + ridge * torch.eye(Da, device=Ac.device, dtype=Ac.dtype)
    W = torch.linalg.solve(G, Ac.T @ Tc)
    pred_te = (A_te - mu_a) @ W + mu_t
    ss_res = ((T_te - pred_te) ** 2).sum(0)
    ss_tot = ((T_te - T_te.mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-8)
    r2_per_ch = 1.0 - ss_res / ss_tot
    return float(r2_per_ch.mean()), float(r2_per_ch.median())


def _per_channel_corr(A, T):
    """Mean over channels of |Pearson(A[:,i], T[:,i])| (only square pairs).
    If channel counts differ, pairs the first min(Da,Dt) channels.
    Returns (mean abs corr, mean signed corr)."""
    K = min(A.shape[1], T.shape[1])
    A, T = A[:, :K], T[:, :K]
    A = A - A.mean(0, keepdim=True)
    T = T - T.mean(0, keepdim=True)
    num = (A * T).sum(0)
    den = (A.pow(2).sum(0).sqrt() * T.pow(2).sum(0).sqrt()).clamp_min(1e-8)
    r = num / den
    return float(r.abs().mean()), float(r.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True,
                    help="Run dir with decoder64_adapter (e.g. xfusion_007).")
    ap.add_argument("--alpha-dir", default="/u/dingqi2/workspace/esa/data/train/alphaearth_emb")
    ap.add_argument("--tessera-dir", default="/u/dingqi2/workspace/esa/data/train/tessera_emb")
    ap.add_argument("--token-dir", default="/u/dingqi2/workspace/esa/data/train/terramind_s2_emb")
    ap.add_argument("--labels-dir", default="/u/dingqi2/workspace/esa/data/train/labels")
    ap.add_argument("--split-file", default=os.path.join(PROJECT_DIR, "splits/split.json"))
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    print(f"  base_ch={cfg.get('lightunet_base_ch')}  fusion_points={cfg.get('fusion_points')}"
          f"  token_gate_kind={cfg.get('token_gate_kind')}")

    if getattr(model, "decoder64_adapter", None) is None:
        print("ERROR: model has no decoder64_adapter; pick a run that fuses TM at decoder64.")
        return

    adapter = model.decoder64_adapter
    has_proj = hasattr(adapter, "proj") and not isinstance(adapter.proj, torch.nn.Identity)
    has_net = hasattr(adapter, "net")  # GatedTokenScaleResidual uses .net
    if not (has_proj or has_net):
        print("ERROR: decoder64_adapter exposes no proj/net submodule for token features.")
        return
    proj_module = adapter.proj if has_proj else adapter.net
    proj_name = "proj" if has_proj else "net"
    print(f"  using adapter.{proj_name} as TM projection")

    # Pre-hook adapter to capture target (AE trunk feature input).
    # Forward-hook proj_module to capture token_proj.
    captured = {"target": [], "tm_proj": []}

    def pre_hook(_module, inputs):
        # adapter.forward(target, token_feat) -> inputs = (target, token_feat)
        captured["target"].append(inputs[0].detach().cpu())

    def proj_hook(_module, _inp, out):
        captured["tm_proj"].append(out.detach().cpu())

    h1 = adapter.register_forward_pre_hook(pre_hook)
    h2 = proj_module.register_forward_hook(proj_hook)

    print(f"\nForwarding {len(ds)} val patches...")
    with torch.no_grad():
        for i in range(len(ds)):
            img, _, _ = ds[i]
            pixel = img[0].unsqueeze(0).to(device)
            token = img[1].unsqueeze(0).to(device)
            _ = model((pixel, token))

    h1.remove()
    h2.remove()

    A = torch.cat(captured["target"], dim=0)   # (N, c3, h, w)
    T = torch.cat(captured["tm_proj"], dim=0)  # (N, c3, h, w)
    print(f"Captured shapes:  AE={tuple(A.shape)}  TM={tuple(T.shape)}")
    if A.shape[-2:] != T.shape[-2:]:
        T = F.interpolate(T, size=A.shape[-2:], mode="bilinear", align_corners=False)
        print(f"  resized TM to {tuple(T.shape)}")

    N, Ca, H, W = A.shape
    Ct = T.shape[1]
    A_flat = A.permute(0, 2, 3, 1).reshape(-1, Ca).float()  # (N*H*W, Ca)
    T_flat = T.permute(0, 2, 3, 1).reshape(-1, Ct).float()  # (N*H*W, Ct)
    print(f"Row matrices: AE={tuple(A_flat.shape)}  TM={tuple(T_flat.shape)}")

    # ---- 1. Per-position cosine similarity ----
    K = min(Ca, Ct)
    a_n = A_flat[:, :K] / A_flat[:, :K].norm(dim=1, keepdim=True).clamp_min(1e-8)
    t_n = T_flat[:, :K] / T_flat[:, :K].norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos_aligned = (a_n * t_n).sum(1)
    # shuffled: permute T rows WITHIN each sample (so each (h,w) of A is paired
    # with a random (h',w') of T in the same patch). This isolates the spatial
    # info question without confounding by sample identity.
    perm = torch.empty_like(T_flat[:, 0], dtype=torch.long)
    HW = H * W
    for s in range(N):
        perm[s * HW : (s + 1) * HW] = torch.randperm(HW) + s * HW
    T_shuf = T_flat[perm]
    t_n_shuf = T_shuf[:, :K] / T_shuf[:, :K].norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos_shuf = (a_n * t_n_shuf).sum(1)

    print("\n========== 1. Per-position cosine similarity ==========")
    print(f"  aligned : mean={cos_aligned.mean():+.4f}  std={cos_aligned.std():.4f}"
          f"  median={cos_aligned.median():+.4f}")
    print(f"  shuffled: mean={cos_shuf.mean():+.4f}  std={cos_shuf.std():.4f}"
          f"  median={cos_shuf.median():+.4f}")
    print(f"  delta(aligned - shuffled) = {(cos_aligned.mean() - cos_shuf.mean()).item():+.4f}")

    # ---- 2. Linear probe R^2: T = W A + b ----
    # Fair comparison: train+test aligned vs train+test shuffled. The shuffled
    # case asks "if AE(h,w) is paired with a RANDOM TM position, how well can
    # any W do?" If TM is truly globally-mixed, shuffled should be nearly as
    # good as aligned (same global mean target everywhere). If aligned >>
    # shuffled with BOTH fairly fit, position-specific info exists.
    n_total = A_flat.shape[0]
    idx = torch.randperm(n_total)
    n_tr = int(0.7 * n_total)
    tr, te = idx[:n_tr], idx[n_tr:]
    A_tr, A_te = A_flat[tr].to(device), A_flat[te].to(device)
    T_tr, T_te = T_flat[tr].to(device), T_flat[te].to(device)
    # Independent shuffles for train and test rows of T (within-sample).
    T_shuf_tr = T_shuf[tr].to(device)
    # Re-shuffle test independently of train (different random pairing).
    perm_te = torch.empty_like(T_flat[:, 0], dtype=torch.long)
    for s in range(N):
        perm_te[s * HW : (s + 1) * HW] = torch.randperm(HW) + s * HW
    T_shuf2 = T_flat[perm_te]
    T_shuf_te = T_shuf2[te].to(device)

    r2_aligned_mean, r2_aligned_med = _linear_probe_r2(A_tr, T_tr, A_te, T_te, ridge=args.ridge)
    r2_shuf_mean, r2_shuf_med = _linear_probe_r2(
        A_tr, T_shuf_tr, A_te, T_shuf_te, ridge=args.ridge
    )

    print("\n========== 2. Linear probe (T = W A + b), R^2 on holdout ==========")
    print("  (each row trains and tests on its own pairing scheme)")
    print(f"  aligned : mean R2={r2_aligned_mean:+.4f}   median R2={r2_aligned_med:+.4f}")
    print(f"  shuffled: mean R2={r2_shuf_mean:+.4f}   median R2={r2_shuf_med:+.4f}")
    spatial_bonus = r2_aligned_mean - r2_shuf_mean
    print(f"  spatial information bonus (aligned - shuffled) = {spatial_bonus:+.4f}")

    # ---- 3. Per-channel correlation ----
    abs_aligned, sign_aligned = _per_channel_corr(A_flat, T_flat)
    abs_shuf, sign_shuf = _per_channel_corr(A_flat, T_shuf)
    print("\n========== 3. Per-channel Pearson |r| (channel pair i,i) ==========")
    print(f"  aligned : mean |r|={abs_aligned:.4f}   mean signed r={sign_aligned:+.4f}")
    print(f"  shuffled: mean |r|={abs_shuf:.4f}   mean signed r={sign_shuf:+.4f}")

    # ---- Verdict ----
    print("\n========== VERDICT ==========")
    if spatial_bonus > 0.05:
        verdict = ("STRONG spatial alignment. Hadamard (W1*t) ⊙ (W2*g) at "
                   "matching (h,w) has real signal to exploit beyond channel-only.")
    elif spatial_bonus > 0.02:
        verdict = ("WEAK spatial alignment. Hadamard MIGHT help marginally. "
                   "Channel-only (FiLM-class) likely captures most of it.")
    else:
        verdict = ("NO meaningful spatial alignment. Per-position Hadamard "
                   "would mostly multiply unrelated channel vectors. "
                   "Stay with channel-only modulation; FiLM failure suggests "
                   "even channel-only signal is weak here.")
    print(f"  spatial_bonus = {spatial_bonus:+.4f}")
    print(f"  --> {verdict}")


if __name__ == "__main__":
    main()
