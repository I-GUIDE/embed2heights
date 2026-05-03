"""
Self-supervised AlphaEarth+Tessera pretraining.

This script uses label-free train embeddings plus optional label-free test
embeddings. It writes a checkpoint whose ``alpha_unet`` and ``tessera_stem``
weights can be loaded by supervised ``tessera_iou_fusion`` runs with
``train.py --init-from-pretrain``.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = SCRIPT_DIR
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from core.pretrain import (
    BlockMask2d,
    PixelFusionPretrainDataset,
    PixelFusionPretrainModel,
    apply_mask_strategy,
    find_pixel_pretrain_pairs,
    masked_reconstruction_loss,
    save_pretrain_config,
)
from core.data.training import build_data_loader


DEFAULT_DATA = os.path.abspath(os.path.join(REPO_DIR, "..", "data"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-alpha-dir", default=os.path.join(DEFAULT_DATA, "train", "alphaearth_emb"))
    p.add_argument("--train-tessera-dir", default=os.path.join(DEFAULT_DATA, "train", "tessera_emb"))
    p.add_argument("--test-alpha-dir", default=None,
                   help="Optional label-free test AlphaEarth embeddings for transductive pretraining.")
    p.add_argument("--test-tessera-dir", default=None,
                   help="Optional label-free test Tessera embeddings for transductive pretraining.")
    p.add_argument("--output-dir", default=os.path.join(REPO_DIR, "runs"))
    p.add_argument("--experiment-name", default="pretrain_ae_tessera")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--mask-ratio", type=float, default=0.55)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--alpha-loss-weight", type=float, default=1.0)
    p.add_argument("--tessera-loss-weight", type=float, default=1.0)
    p.add_argument("--cosine-weight", type=float, default=0.05)
    p.add_argument("--lightunet-base-ch", type=int, default=48)
    p.add_argument("--tessera-presence-ch", type=int, default=16)
    p.add_argument("--tessera-hidden-ch", type=int, default=96)
    p.add_argument("--tessera-hidden-depth", type=int, default=2)
    p.add_argument("--fusion-ch", type=int, default=None)
    p.add_argument("--norm-kind", default="bn", choices=["bn", "gn"],
                   help="LightUNet normalization layer. Default 'bn' matches "
                        "the only positive pretrain result documented in "
                        "logs/PRETRAIN_AE_TESSERA_REPORT.md. The 'gn' variant "
                        "was tested in logs/PRETRAIN_GN_XMODAL_REPORT.md and "
                        "regressed downstream score.")
    p.add_argument("--mask-strategy", default="independent",
                   choices=["complementary", "dual", "independent", "mixed"],
                   help="How alpha and tessera masks relate per batch. "
                        "Default 'independent' matches the positive BN "
                        "pretrain recipe. Cross-modal/mixed masking was tested "
                        "with GN and closed after downstream regression.")
    p.add_argument("--modality-dropout", type=float, default=0.0,
                   help="Probability that the 'visible' side is also zeroed "
                        "in a complementary batch. Forces robustness to a "
                        "missing modality at finetune time. 0.0 disables.")
    return p.parse_args()


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_pair(batch, device):
    alpha, tessera = batch
    non_blocking = device.type == "cuda"
    return (
        alpha.to(device, non_blocking=non_blocking),
        tessera.to(device, non_blocking=non_blocking),
    )


def make_loaders(args, device):
    pairs = find_pixel_pretrain_pairs(
        args.train_alpha_dir,
        args.train_tessera_dir,
        args.test_alpha_dir,
        args.test_tessera_dir,
    )
    if not pairs:
        raise ValueError("No AlphaEarth/Tessera pretrain pairs found.")

    val_fraction = min(max(args.val_fraction, 0.0), 0.5)
    if val_fraction > 0 and len(pairs) > 1:
        split_labels = [p["split"] for p in pairs]
        split_counts = {label: split_labels.count(label) for label in set(split_labels)}
        stratify = (
            split_labels
            if len(split_counts) > 1 and min(split_counts.values()) >= 2
            else None
        )
        train_pairs, val_pairs = train_test_split(
            pairs,
            test_size=val_fraction,
            random_state=args.seed,
            stratify=stratify,
        )
    else:
        train_pairs, val_pairs = pairs, []

    train_ds = PixelFusionPretrainDataset(train_pairs, patch_size=args.patch_size, is_train=True)
    val_ds = PixelFusionPretrainDataset(val_pairs, patch_size=args.patch_size, is_train=False)

    train_loader = build_data_loader(train_ds, args, device, shuffle=True)
    val_loader = build_data_loader(val_ds, args, device, shuffle=False) if val_pairs else None
    return train_loader, val_loader, train_pairs, val_pairs


def run_epoch(model, loader, masker, optimizer, scaler, device, args, *, train):
    model.train(train)
    running = 0.0
    parts = {"alpha": 0.0, "tessera": 0.0, "alpha_mse": 0.0, "tessera_mse": 0.0}
    samples = 0
    grad_accum = max(1, args.grad_accum_steps)
    use_amp = args.amp and device.type == "cuda"
    desc = "train" if train else "val"

    if train:
        optimizer.zero_grad(set_to_none=True)

    # Validation always uses the legacy independent-mask sampling so that the
    # reported val_loss is comparable across mask strategies.
    epoch_strategy = args.mask_strategy if train else "independent"

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=desc, leave=False)
        for step, batch in enumerate(pbar, start=1):
            alpha, tessera = move_pair(batch, device)
            (
                alpha_in, alpha_mask,
                tessera_in, tessera_mask,
                alpha_w_mask, tessera_w_mask,
            ) = apply_mask_strategy(
                masker, alpha, tessera, epoch_strategy,
                modality_dropout=args.modality_dropout if train else 0.0,
            )

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(alpha_in, tessera_in)
                alpha_loss, alpha_parts = masked_reconstruction_loss(
                    out["alpha"], alpha, alpha_mask, cosine_weight=args.cosine_weight
                )
                tessera_loss, tessera_parts = masked_reconstruction_loss(
                    out["tessera"], tessera, tessera_mask, cosine_weight=args.cosine_weight
                )
                loss = (
                    args.alpha_loss_weight * alpha_w_mask * alpha_loss
                    + args.tessera_loss_weight * tessera_w_mask * tessera_loss
                )
                step_loss = loss / grad_accum if train else loss

            if not torch.isfinite(loss):
                print(f"\nWARN: non-finite pretrain loss at step {step}; skipping batch.")
                if train:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if train:
                scaler.scale(step_loss).backward()
                should_step = step % grad_accum == 0 or step == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            bs = alpha.size(0)
            running += loss.detach().item() * bs
            parts["alpha"] += alpha_loss.detach().item() * bs
            parts["tessera"] += tessera_loss.detach().item() * bs
            parts["alpha_mse"] += alpha_parts["mse"].item() * bs
            parts["tessera_mse"] += tessera_parts["mse"].item() * bs
            samples += bs
            avg = running / max(1, samples)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")

    return running / max(1, samples), {k: v / max(1, samples) for k, v in parts.items()}


def write_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_checkpoint(path, model, optimizer, epoch, best_loss, args, train_pairs, val_pairs):
    torch.save({
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": vars(args),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
    }, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    history_path = os.path.join(exp_dir, "pretrain_history.jsonl")
    best_path = os.path.join(exp_dir, "pretrain_best.pth")
    last_path = os.path.join(exp_dir, "pretrain_last.pth")
    config_path = os.path.join(exp_dir, "pretrain_params.json")
    open(history_path, "w").close()

    train_loader, val_loader, train_pairs, val_pairs = make_loaders(args, device)
    save_pretrain_config(config_path, {
        **vars(args),
        "device": str(device),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "uses_test_embeddings": bool(args.test_alpha_dir or args.test_tessera_dir),
    })
    print(f"Pretrain pairs: train={len(train_pairs)} val={len(val_pairs)}")
    print(f"Experiment folder: {exp_dir}")

    model = PixelFusionPretrainModel(
        alpha_channels=64,
        tessera_channels=128,
        base_ch=args.lightunet_base_ch,
        tessera_presence_ch=args.tessera_presence_ch,
        tessera_hidden_ch=args.tessera_hidden_ch,
        tessera_hidden_depth=args.tessera_hidden_depth,
        fusion_ch=args.fusion_ch,
        norm_kind=args.norm_kind,
    ).to(device)
    masker = BlockMask2d(mask_ratio=args.mask_ratio, block_size=args.block_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_parts = run_epoch(
            model, train_loader, masker, optimizer, scaler, device, args, train=True
        )
        if val_loader is not None:
            val_loss, val_parts = run_epoch(
                model, val_loader, masker, optimizer, scaler, device, args, train=False
            )
            score_loss = val_loss
        else:
            val_loss, val_parts = None, {}
            score_loss = train_loss

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_components": train_parts,
            "val_components": val_parts,
            "lr": optimizer.param_groups[0]["lr"],
        }
        write_jsonl(history_path, record)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f}" if val_loss is not None else
            f"Epoch {epoch:03d}/{args.epochs} train={train_loss:.4f}"
        )

        if score_loss < best_loss:
            best_loss = score_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_loss, args, train_pairs, val_pairs)
            print(f"  saved best: {best_path} ({best_loss:.4f})")
        save_checkpoint(last_path, model, optimizer, epoch, best_loss, args, train_pairs, val_pairs)


if __name__ == "__main__":
    main()
