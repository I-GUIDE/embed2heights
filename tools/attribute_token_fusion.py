#!/usr/bin/env python
"""Attribute the xfusion_083/085 hybrid-fusion gain to specific token sources and pathways.

Reads a trained run's resolved_config.yml + model_best.pth and runs four probes
on the validation split:

  A. LOSO (leave-one-source-out) — zero each of the 4 token sources at inference,
     measure val_loss and key components. Drop magnitude = source utility.

  B. Per-source weight-norm dashboard — ||film_conv_i||, ||add_conv_i||, gate
     bias for each of the N sources. Cheap proxy for "did the model learn to
     use this source."

  C. Pathway ablation — disable each hybrid pathway in turn (self-attn
     refinement / additive A_i / spatial gate sigma(g_i)). Drop magnitude =
     mechanism's contribution to the trained model.

  D. Cross-source attention map — extract mean attention weights between sources
     for one validation batch, reduce to N x N block-mean matrix.

Run:
    python tools/attribute_token_fusion.py \
        --run-dir runs/xfusion_085_hybrid_tied_no_residual_ep60 \
        --output  runs/xfusion_085_hybrid_tied_no_residual_ep60/attribution.json
"""

import argparse
import copy
import json
import os
import sys
from argparse import Namespace

import torch
import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from core.data.training import make_dataloaders  # noqa: E402
from core.engine import run_epoch, select_device, seed_everything  # noqa: E402
from core.losses import ImprovedCompositeLoss  # noqa: E402
from core.models import build_model  # noqa: E402

SOURCE_NAMES = ["terramind_s1", "terramind_s2", "thor_s1", "thor_s2"]
TOKEN_SOURCE_CH = 768


# ----------------- Setup helpers --------------------------------------------

def load_resolved_config(run_dir):
    path = os.path.join(run_dir, "resolved_config.yml")
    with open(path) as f:
        return yaml.safe_load(f)


def args_from_resolved(cfg, run_dir):
    """Flatten a resolved_config.yml back into an argparse-like Namespace.

    Only the fields used by make_dataloaders / build_model / ImprovedCompositeLoss
    are populated.
    """
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["training"]
    runtime = cfg["runtime"]
    ns = Namespace(
        # data
        train_embeddings_dir=data["train_embeddings_dir"],
        secondary_train_embeddings_dir=data.get("secondary_train_embeddings_dir"),
        token_train_embeddings_dir=data.get("token_train_embeddings_dir"),
        secondary_token_train_embeddings_dir=data.get("secondary_token_train_embeddings_dir"),
        third_token_train_embeddings_dir=data.get("third_token_train_embeddings_dir"),
        fourth_token_train_embeddings_dir=data.get("fourth_token_train_embeddings_dir"),
        token_normalization=data.get("token_normalization"),
        token_normalization_source_indices=data.get("token_normalization_source_indices"),
        token_normalization_stats_path=data.get("token_normalization_stats_path"),
        train_targets_dir=data["train_targets_dir"],
        split_file=data["split_file"],
        patch_size=data["patch_size"],
        val_split=0.2,
        # training-runtime
        batch_size=train["batch_size"],
        num_workers=runtime["num_workers"],
        prefetch_factor=runtime["prefetch_factor"],
        seed=train["seed"],
        # model (for build_model)
        model_type=model["model_type"],
        tessera_presence_ch=model["tessera_presence_ch"],
        tessera_hidden_ch=model["tessera_hidden_ch"],
        tessera_hidden_depth=model["tessera_hidden_depth"],
        height_specialist_depth=model["height_specialist_depth"],
        height_gate_source=model["height_gate_source"],
        height_hidden_ch=model["height_hidden_ch"],
        height_trunk_depth=model["height_trunk_depth"],
        height_independent_branches=model["height_independent_branches"],
        height_head_kind=model["height_head_kind"],
        height_n_bins=model["height_n_bins"],
        height_bin_max_m=model["height_bin_max_m"],
        lightunet_base_ch=model["lightunet_base_ch"],
        lightunet_norm_kind=model["lightunet_norm_kind"],
        gate_mode=model["gate_mode"],
        gate_untied=model["gate_untied"],
        gate_init_bias=model["gate_init_bias"],
        modality_dropout=model["modality_dropout"],
        presence_head_kind=model["presence_head_kind"],
        presence_head_depth=model["presence_head_depth"],
        presence_branch_ch=model["presence_branch_ch"],
        use_fraction_film=model["use_fraction_film"],
        use_fraction_aux=model["use_fraction_aux"],
        token_calibration=model.get("token_calibration", False),
        attn_heads=model.get("attn_heads", 4),
        use_additive=model.get("use_additive", True),
        use_spatial_gate=model.get("use_spatial_gate", True),
        # loss
        weight_mae=train["weight_mae"],
        weight_presence_tversky=train["weight_presence_tversky"],
        weight_fraction_mae=train["weight_fraction_mae"],
        weight_height_boost=train["weight_height_boost"],
        aux_weight=train["aux_weight"],
        height_loss_kind=train["height_loss_kind"],
        huber_delta=train["huber_delta"],
        build_height_boost=train["build_height_boost"],
        veg_height_boost=train["veg_height_boost"],
        aux_veg_weight=train["aux_veg_weight"],
        height_bin_aux_weight=train["height_bin_aux_weight"],
        height_bin_sigma_bins=train["height_bin_sigma_bins"],
        tversky_water_alpha=train["tversky_water_alpha"],
        water_empty_topk=train["water_empty_topk"],
        weight_water_empty_topk=train["weight_water_empty_topk"],
        # misc
        config=cfg.get("source_config"),
        experiment_name=runtime["experiment_name"],
        output_dir=runtime["output_dir"],
        amp=runtime["amp"],
        data_parallel=runtime["data_parallel"],
    )
    return ns


def build_criterion(args, device):
    return ImprovedCompositeLoss(
        weight_mae=args.weight_mae,
        weight_presence_tversky=args.weight_presence_tversky,
        weight_fraction_mae=args.weight_fraction_mae,
        weight_height_boost=args.weight_height_boost,
        aux_weight=args.aux_weight,
        loss_preset="presence_centered",
        height_loss_kind=args.height_loss_kind,
        huber_delta=args.huber_delta,
        build_height_boost=args.build_height_boost,
        veg_height_boost=args.veg_height_boost,
        aux_veg_weight=args.aux_veg_weight,
        height_bin_aux_weight=args.height_bin_aux_weight,
        height_bin_sigma_bins=args.height_bin_sigma_bins,
        tversky_water_alpha=args.tversky_water_alpha,
        water_empty_topk=args.water_empty_topk,
        weight_water_empty_topk=args.weight_water_empty_topk,
    ).to(device)


def build_loaded_model(args, n_channels, ckpt_path, device):
    model, name = build_model(
        args.model_type,
        n_channels,
        n_classes=4,
        tessera_presence_ch=args.tessera_presence_ch,
        tessera_hidden_ch=args.tessera_hidden_ch,
        tessera_hidden_depth=args.tessera_hidden_depth,
        height_specialist_depth=args.height_specialist_depth,
        lightunet_base_ch=args.lightunet_base_ch,
        height_gate_source=args.height_gate_source,
        height_hidden_ch=args.height_hidden_ch,
        height_trunk_depth=args.height_trunk_depth,
        height_independent_branches=args.height_independent_branches,
        height_head_kind=args.height_head_kind,
        height_n_bins=args.height_n_bins,
        height_bin_max_m=args.height_bin_max_m,
        lightunet_norm_kind=args.lightunet_norm_kind,
        gate_mode=args.gate_mode,
        gate_untied=args.gate_untied,
        gate_init_bias=args.gate_init_bias,
        modality_dropout=args.modality_dropout,
        presence_head_kind=args.presence_head_kind,
        presence_head_depth=args.presence_head_depth,
        presence_branch_ch=args.presence_branch_ch,
        use_fraction_film=args.use_fraction_film,
        use_fraction_aux=args.use_fraction_aux,
        attn_heads=args.attn_heads,
        use_additive=args.use_additive,
        use_spatial_gate=args.use_spatial_gate,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, name


# ----------------- Probe A: LOSO token source -------------------------------

class TokenMaskedModel(torch.nn.Module):
    """Zero out one or more token sources before delegating to the base model."""

    supports_aux_outputs = True

    def __init__(self, base, masked_source_idx, src_ch=TOKEN_SOURCE_CH):
        super().__init__()
        self.base = base
        self.masked = sorted(set(masked_source_idx))
        self.src_ch = src_ch

    def forward(self, x, return_aux=False):
        pixel, token = x
        if self.masked:
            token = token.clone()
            for i in self.masked:
                token[:, i * self.src_ch:(i + 1) * self.src_ch] = 0
        return self.base((pixel, token), return_aux=return_aux)


# ----------------- Probe C: pathway ablation --------------------------------

def clone_model(model):
    return copy.deepcopy(model).eval()


def ablate_attention(model):
    """Zero the cross-source self-attention output projection."""
    fusion = model.hybrid_fusion
    if not fusion.attn_enabled:
        return model
    with torch.no_grad():
        fusion.cross_source_attn.out_proj.weight.zero_()
        fusion.cross_source_attn.out_proj.bias.zero_()
    return model


def ablate_additive(model):
    """Zero the additive A_i pathway for every source."""
    fusion = model.hybrid_fusion
    if fusion.add_convs is None:
        return model
    with torch.no_grad():
        for c in fusion.add_convs:
            c.weight.zero_()
            c.bias.zero_()
    return model


def ablate_spatial_gate(model):
    """Disable the spatial gate (let all deltas through full strength)."""
    fusion = model.hybrid_fusion
    fusion.gate_convs = None
    return model


# ----------------- Probe B: weight-norm + gate dashboard --------------------

def weight_norm_report(model):
    fusion = model.hybrid_fusion
    n = fusion.n_sources
    report = {"n_sources": n}
    film_norms = []
    add_norms = []
    gate_bias = []
    proj_norms = []
    for i in range(n):
        proj_norms.append(
            sum(
                p.detach().float().norm().item() ** 2
                for p in fusion.token_projs[i].parameters() if p.requires_grad
            ) ** 0.5
        )
        film_norms.append(fusion.film_convs[i].weight.detach().float().norm().item())
        if fusion.add_convs is not None:
            add_norms.append(fusion.add_convs[i].weight.detach().float().norm().item())
        else:
            add_norms.append(None)
        if fusion.gate_convs is not None:
            gate_bias.append(fusion.gate_convs[i].bias.detach().float().item())
        else:
            gate_bias.append(None)

    report["per_source"] = [
        {
            "idx": i,
            "name": SOURCE_NAMES[i] if i < len(SOURCE_NAMES) else f"src{i}",
            "token_proj_norm": proj_norms[i],
            "film_conv_norm": film_norms[i],
            "add_conv_norm": add_norms[i],
            "gate_bias": gate_bias[i],
        }
        for i in range(n)
    ]
    if fusion.attn_enabled:
        op = fusion.cross_source_attn.out_proj.weight.detach().float()
        report["attn_out_proj_norm"] = op.norm().item()
        report["modality_embed_norms"] = [
            fusion.modality_embed.detach().float()[i].norm().item()
            for i in range(n)
        ]
    return report


# ----------------- Probe B': mean spatial gate over val ---------------------

def collect_spatial_gate_stats(model, val_loader, device, max_batches=8):
    """Hook gate_convs[i] and accumulate sigmoid-output mean/std over batches."""
    fusion = model.hybrid_fusion
    if fusion.gate_convs is None:
        return None

    sums = [0.0 for _ in range(fusion.n_sources)]
    sq_sums = [0.0 for _ in range(fusion.n_sources)]
    counts = [0 for _ in range(fusion.n_sources)]
    handles = []

    def make_hook(i):
        def _h(_mod, _inp, out):
            g = torch.sigmoid(out.detach().float())
            sums[i] += g.sum().item()
            sq_sums[i] += (g ** 2).sum().item()
            counts[i] += g.numel()
        return _h

    for i, c in enumerate(fusion.gate_convs):
        handles.append(c.register_forward_hook(make_hook(i)))

    try:
        model.eval()
        with torch.no_grad():
            for b, (imgs, _, _) in enumerate(val_loader):
                if b >= max_batches:
                    break
                if isinstance(imgs, (tuple, list)):
                    imgs = tuple(x.to(device, non_blocking=True) for x in imgs)
                else:
                    imgs = imgs.to(device, non_blocking=True)
                _ = model(imgs, return_aux=False)
    finally:
        for h in handles:
            h.remove()

    return [
        {
            "idx": i,
            "name": SOURCE_NAMES[i] if i < len(SOURCE_NAMES) else f"src{i}",
            "mean_sigma_g": sums[i] / max(1, counts[i]),
            "std_sigma_g": (
                (sq_sums[i] / max(1, counts[i])) - (sums[i] / max(1, counts[i])) ** 2
            ) ** 0.5,
            "n_pixels": counts[i],
        }
        for i in range(fusion.n_sources)
    ]


# ----------------- Probe D: cross-source attention map ----------------------

def collect_attention_map(model, val_loader, device, max_batches=4):
    """Run cross_source_attn with need_weights=True on a few batches and average.

    Returns an N x N matrix where entry (i, j) is the mean attention weight from
    source i (query) to source j (key/value), block-averaged over spatial tokens.
    """
    fusion = model.hybrid_fusion
    if not fusion.attn_enabled:
        return None

    n = fusion.n_sources
    accum = torch.zeros(n, n, dtype=torch.float64)
    n_batches = 0

    # Monkey-patch _refine_sources to capture attention weights once per batch.
    original = fusion._refine_sources
    captured = {}

    def patched(ctx_list):
        b, _, h, w = ctx_list[0].shape
        pos = fusion._pos_tokens(h, w, ctx_list[0].device, ctx_list[0].dtype)
        seqs = []
        for i, ctx in enumerate(ctx_list):
            tokens = ctx.flatten(2).transpose(1, 2)
            tokens = tokens + pos
            tokens = tokens + fusion.modality_embed[i].view(1, 1, -1)
            seqs.append(tokens)
        x = torch.cat(seqs, dim=1)
        x_norm = fusion.attn_norm(x)
        with torch.amp.autocast("cuda", enabled=False):
            attn_out, attn_weights = fusion.cross_source_attn(
                x_norm.float(), x_norm.float(), x_norm.float(),
                need_weights=True, average_attn_weights=True,
            )
        captured["w"] = attn_weights.detach().float().cpu()  # (B, L, L)
        captured["L_per_src"] = h * w
        chunks = attn_out.to(x.dtype).split(h * w, dim=1)
        refined = []
        for i, ck in enumerate(chunks):
            delta = ck.transpose(1, 2).reshape(b, fusion.ctx_ch, h, w)
            refined.append(ctx_list[i] + delta)
        return refined

    fusion._refine_sources = patched
    try:
        model.eval()
        with torch.no_grad():
            for b, (imgs, _, _) in enumerate(val_loader):
                if b >= max_batches:
                    break
                if isinstance(imgs, (tuple, list)):
                    imgs = tuple(x.to(device, non_blocking=True) for x in imgs)
                else:
                    imgs = imgs.to(device, non_blocking=True)
                _ = model(imgs, return_aux=False)
                if "w" not in captured:
                    continue
                w = captured["w"]  # (B, L, L) where L = n*Lps
                Lps = captured["L_per_src"]
                # Block-mean: avg over spatial positions inside each (src_q, src_k) block
                for qi in range(n):
                    for kj in range(n):
                        blk = w[
                            :, qi * Lps:(qi + 1) * Lps, kj * Lps:(kj + 1) * Lps
                        ]
                        accum[qi, kj] += blk.mean().item()
                n_batches += 1
                captured.clear()
    finally:
        fusion._refine_sources = original

    if n_batches == 0:
        return None
    mat = (accum / n_batches).tolist()
    return {
        "source_names": SOURCE_NAMES[:n],
        "matrix": mat,
        "n_batches_averaged": n_batches,
        "note": "entry (i,j) = mean attention weight from source i (query) to source j (key)",
    }


# ----------------- Evaluation driver ----------------------------------------

def evaluate_loss(model, val_loader, criterion, device, *, desc=""):
    """Compute val_loss + components for a model on the val_loader (no AMP)."""
    avg_loss, comp_avg = run_epoch(
        model, val_loader, criterion,
        optimizer=None, scaler=None, device=device,
        train=False, use_amp=False, desc=desc,
    )
    return {"val_loss": avg_loss, "components": comp_avg}


def summarize_components(comp):
    """Pick a small set of headline components for readable tables."""
    keys = [
        "mae", "presence_tversky", "height_boost", "water_empty_topk",
        "fraction_mae", "aux_height_building", "aux_height_vegetation",
    ]
    return {k: comp.get(k) for k in keys}


# ----------------- Main -----------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Path to runs/<exp>/ containing resolved_config.yml + model_best.pth")
    p.add_argument("--ckpt", default="model_best.pth",
                   help="Filename of checkpoint inside run-dir.")
    p.add_argument("--output", default=None,
                   help="Output JSON path. Defaults to <run-dir>/attribution.json.")
    p.add_argument("--gate-stat-batches", type=int, default=8)
    p.add_argument("--attn-batches", type=int, default=4)
    p.add_argument("--skip-loso", action="store_true",
                   help="Skip Probe A (LOSO) - the most expensive probe.")
    p.add_argument("--skip-pathways", action="store_true",
                   help="Skip Probe C (pathway ablation).")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    ckpt_path = os.path.join(run_dir, args.ckpt)
    output_path = args.output or os.path.join(run_dir, "attribution.json")

    print(f">>> Loading resolved_config from {run_dir}")
    cfg = load_resolved_config(run_dir)
    run_args = args_from_resolved(cfg, run_dir)

    device = select_device()
    seed_everything(run_args.seed)
    print(f">>> Device: {device}")

    print(">>> Building data loaders (this also loads the val split)")
    _, val_loader, _, _, n_channels = make_dataloaders(run_args, device)
    print(f"    val batches: {len(val_loader)}  n_channels: {n_channels}")

    print(">>> Loading model + checkpoint")
    model, name = build_loaded_model(run_args, n_channels, ckpt_path, device)
    print(f"    model: {name}  ckpt: {ckpt_path}")

    criterion = build_criterion(run_args, device)

    results = {
        "run_dir": run_dir,
        "ckpt": ckpt_path,
        "experiment_name": run_args.experiment_name,
        "model_type": run_args.model_type,
    }

    print("\n=== Baseline (full model, no ablation) ===")
    baseline = evaluate_loss(model, val_loader, criterion, device, desc="baseline")
    print(f"    val_loss = {baseline['val_loss']:.6f}")
    results["baseline"] = baseline

    print("\n=== Probe B: per-source weight norms ===")
    wn = weight_norm_report(model)
    results["weight_norms"] = wn
    for s in wn["per_source"]:
        add_str = "None" if s["add_conv_norm"] is None else f"{s['add_conv_norm']:.4f}"
        gate_str = "None" if s["gate_bias"] is None else f"{s['gate_bias']:+.4f}"
        print(
            f"    {s['name']:>13s}  proj={s['token_proj_norm']:6.3f}  "
            f"film={s['film_conv_norm']:6.4f}  "
            f"add={add_str}  gate_bias={gate_str}"
        )
    if "attn_out_proj_norm" in wn:
        print(f"    cross_attn out_proj_norm = {wn['attn_out_proj_norm']:.4f}")
        print(f"    modality_embed_norms = {[f'{x:.3f}' for x in wn['modality_embed_norms']]}")

    print("\n=== Probe B': mean spatial gate sigma over val ===")
    gate_stats = collect_spatial_gate_stats(
        model, val_loader, device, max_batches=args.gate_stat_batches
    )
    results["spatial_gate_stats"] = gate_stats
    if gate_stats is not None:
        for g in gate_stats:
            print(
                f"    {g['name']:>13s}  mean sigma(g) = {g['mean_sigma_g']:.4f}  "
                f"std = {g['std_sigma_g']:.4f}"
            )

    print("\n=== Probe D: cross-source attention map ===")
    attn_map = collect_attention_map(
        model, val_loader, device, max_batches=args.attn_batches
    )
    results["attention_map"] = attn_map
    if attn_map is not None:
        print("    Row=query source, Col=key source. Higher = more attention.")
        print("    " + "        ".join(f"{n[:11]:>11}" for n in attn_map["source_names"]))
        for i, row in enumerate(attn_map["matrix"]):
            print(
                f"    {attn_map['source_names'][i][:11]:>11}  "
                + "  ".join(f"{v:9.4f}" for v in row)
            )

    if not args.skip_loso:
        print("\n=== Probe A: Leave-one-source-out (LOSO) ===")
        loso = []
        n_sources = wn["n_sources"]
        for i in range(n_sources):
            masked = TokenMaskedModel(model, [i]).to(device)
            r = evaluate_loss(masked, val_loader, criterion, device, desc=f"mask_{SOURCE_NAMES[i]}")
            delta = r["val_loss"] - baseline["val_loss"]
            print(
                f"    mask {SOURCE_NAMES[i]:>13s}:  val_loss = {r['val_loss']:.6f}  "
                f"(+{delta:.6f}, +{100*delta/baseline['val_loss']:.2f}%)"
            )
            loso.append({
                "masked_source": SOURCE_NAMES[i],
                "val_loss": r["val_loss"],
                "delta": delta,
                "headline_components": summarize_components(r["components"]),
            })
        # Also mask ALL tokens to upper-bound token contribution
        all_masked = TokenMaskedModel(model, list(range(n_sources))).to(device)
        r_all = evaluate_loss(all_masked, val_loader, criterion, device, desc="mask_all_tokens")
        d_all = r_all["val_loss"] - baseline["val_loss"]
        print(
            f"    mask ALL tokens:        val_loss = {r_all['val_loss']:.6f}  "
            f"(+{d_all:.6f}, +{100*d_all/baseline['val_loss']:.2f}%)"
        )
        results["loso"] = {
            "per_source": loso,
            "all_tokens_zero": {
                "val_loss": r_all["val_loss"], "delta": d_all,
                "headline_components": summarize_components(r_all["components"]),
            },
        }

    if not args.skip_pathways:
        print("\n=== Probe C: pathway ablation (FiLM / additive / gate / attn) ===")
        pathways = []
        for label, mut in (
            ("kill_self_attn", ablate_attention),
            ("kill_additive_A", ablate_additive),
            ("kill_spatial_gate", ablate_spatial_gate),
        ):
            m = clone_model(model).to(device)
            mut(m)
            r = evaluate_loss(m, val_loader, criterion, device, desc=label)
            delta = r["val_loss"] - baseline["val_loss"]
            print(
                f"    {label:>20s}:  val_loss = {r['val_loss']:.6f}  "
                f"(+{delta:+.6f}, +{100*delta/baseline['val_loss']:.2f}%)"
            )
            pathways.append({
                "ablation": label,
                "val_loss": r["val_loss"],
                "delta": delta,
                "headline_components": summarize_components(r["components"]),
            })
        results["pathway_ablation"] = pathways

    print(f"\n>>> Writing report to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(">>> Done.")


if __name__ == "__main__":
    main()
