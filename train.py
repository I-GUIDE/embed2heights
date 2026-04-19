"""
Train a single emb2heights backbone.

The script is deliberately monolithic: one process trains one model on one
embedding source and writes all artifacts under `runs/<experiment_name>/`.
Compose multiple training runs in a shell script or slurm array — there is no
multi-baseline driver.
"""
import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import rasterio
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from core.model import build_model
from core.dataset import (
    find_file_pairs, save_split, load_split,
    pick_dataset_class,
)
from core.losses import ImprovedCompositeLoss


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_EMB = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "train", "alphaearth_emb"))
DEFAULT_TRAIN_TAR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "train", "labels"))

MODEL_CHOICES = [
    "auto", "lightunet", "decoder_residual", "embedding_refiner",
    "hrnet_w18", "hrnet_w32",
]

# Defaults — every one is overridable from the CLI.
DEFAULTS = {
    "experiment_name": "run01",
    "output_dir":      os.path.join(SCRIPT_DIR, "runs"),
    "batch_size":      32,
    "patch_size":      256,
    "epochs":          30,
    "lr":              2e-4,
    "weight_decay":    1e-4,
    "val_split":       0.2,
    "lambdas":         [1.0, 0.5, 0.5, 2.0],   # [MAE, SSIM, Gradient, Tversky]
    # Presence head is now the submission output for land-cover channels,
    # so its BCE supervision is primary, not auxiliary. Bumped from 0.25.
    "aux_weight":      1.0,
    "seed":            42,
    "model_type":      "auto",
    "amp":             True,
    "grad_accum":      1,
    "num_workers":     2,
}


RAW_COMPONENTS = (
    "mae",
    "ssim",
    "grad",
    "tversky",
    "height_boost",
    "presence_bce",
    "aux_height_building",
    "aux_height_vegetation",
)

WEIGHTED_COMPONENTS = (
    "weighted_mae",
    "weighted_ssim",
    "weighted_grad",
    "weighted_tversky",
    "weighted_height_boost",
    "weighted_presence_bce",
    "weighted_aux_height",
)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type",           default=DEFAULTS["model_type"], choices=MODEL_CHOICES)
    p.add_argument("--output-dir",           default=DEFAULTS["output_dir"])
    p.add_argument("--train-embeddings-dir", default=DEFAULT_TRAIN_EMB)
    p.add_argument("--train-targets-dir",    default=DEFAULT_TRAIN_TAR)
    p.add_argument("--experiment-name",      default=DEFAULTS["experiment_name"])
    p.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--patch-size",     type=int,   default=DEFAULTS["patch_size"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay",   type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=DEFAULTS["amp"],
                   help="Use CUDA automatic mixed precision when available.")
    p.add_argument("--grad-accum-steps", type=int, default=DEFAULTS["grad_accum"],
                   help="Accumulate gradients over N mini-batches before optimizer step.")
    p.add_argument("--num-workers",    type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--aux-weight",     type=float, default=DEFAULTS["aux_weight"],
                   help="Weight for auxiliary multi-head supervision.")
    p.add_argument("--seed",           type=int, default=DEFAULTS["seed"])
    p.add_argument("--split-file",     default=None,
                   help="Path to a JSON split file. Loaded if present, else a new split is saved there.")
    return p.parse_args()


def make_dataloaders(args):
    all_pairs = find_file_pairs(args.train_embeddings_dir, args.train_targets_dir)
    if not all_pairs:
        raise ValueError(
            f"No (embedding, label) pairs found.\n"
            f"  train_embeddings_dir='{args.train_embeddings_dir}'\n"
            f"  train_targets_dir='{args.train_targets_dir}'\n"
            "Check filename conventions and directory paths."
        )

    if args.split_file and os.path.exists(args.split_file):
        train_pairs, val_pairs = load_split(args.split_file, all_pairs)
    else:
        train_pairs, val_pairs = train_test_split(
            all_pairs, test_size=DEFAULTS["val_split"], random_state=args.seed
        )
        if args.split_file:
            save_split(args.split_file, train_pairs, val_pairs)

    with rasterio.open(train_pairs[0][0]) as src:
        n_channels = src.count

    DatasetCls = pick_dataset_class(args.model_type, n_channels)
    if DatasetCls.__name__ == "LatentTokenDataset":
        train_ds = DatasetCls(train_pairs, patch_size=args.patch_size, scale_factor=16, is_train=True)
        val_ds   = DatasetCls(val_pairs,   patch_size=args.patch_size, scale_factor=16, is_train=False)
    else:
        train_ds = DatasetCls(train_pairs, patch_size=args.patch_size, is_train=True)
        val_ds   = DatasetCls(val_pairs,   patch_size=args.patch_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, train_ds, val_ds, n_channels


def forward_for_training(model, imgs):
    if getattr(model, "supports_aux_outputs", False):
        return model(imgs, return_aux=True)
    return model(imgs)


def save_experiment_config(exp_dir, args, device, use_amp):
    os.makedirs(exp_dir, exist_ok=True)
    cfg = {
        "experiment_name": args.experiment_name,
        "model_type":      args.model_type,
        "output_dir":      args.output_dir,
        "batch_size":      args.batch_size,
        "patch_size":      args.patch_size,
        "epochs":          args.epochs,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "loss_lambdas":    DEFAULTS["lambdas"],
        "aux_weight":      args.aux_weight,
        "amp":             use_amp,
        "grad_accum":      args.grad_accum_steps,
        "num_workers":     args.num_workers,
        "seed":            args.seed,
        "train_embeddings_dir": args.train_embeddings_dir,
        "train_targets_dir":    args.train_targets_dir,
        "val_split":       DEFAULTS["val_split"],
        "device":          str(device),
        "optimizer":       "AdamW",
        "scheduler":       "ReduceLROnPlateau(factor=0.5, patience=2)",
        "grad_clip":       1.0,
    }
    with open(os.path.join(exp_dir, "training_params.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Created experiment folder: {exp_dir}")


def run_epoch(model, loader, criterion, optimizer, scaler, device, *, train,
              grad_accum_steps=1, use_amp=False, desc=""):
    """Train or eval one epoch. Returns (avg_loss, component_avgs)."""
    model.train(train)
    running_loss = 0.0
    component_sums = {}
    samples_seen = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    if train:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, (imgs, targets, masks) in enumerate(pbar, start=1):
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = forward_for_training(model, imgs)
                loss, loss_components = criterion(outputs, targets, masks)
                step_loss = loss / grad_accum_steps if train else loss

            if train:
                scaler.scale(step_loss).backward()
                should_step = step % grad_accum_steps == 0 or step == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            for name, value in loss_components.items():
                component_sums[name] = component_sums.get(name, 0.0) + value.detach().item() * bs
            samples_seen += bs
            avg = running_loss / max(1, samples_seen)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")

    avg_loss = running_loss / max(1, samples_seen)
    comp_avg = {
        name: value / max(1, samples_seen)
        for name, value in component_sums.items()
    }
    return avg_loss, comp_avg


def format_components(components, names):
    parts = []
    for name in names:
        if name in components:
            parts.append(f"{name}:{components[name]:.3f}")
    return " | ".join(parts)


def write_history_record(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def plot_loss_curve(train_losses, val_losses, out_path, experiment_name):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Validation Loss")
    plt.title(f"Training Loss Curve ({experiment_name})")
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    device = select_device()
    seed_everything(args.seed)

    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    best_model_path = os.path.join(exp_dir, "model_best.pth")
    last_model_path = os.path.join(exp_dir, "model_last.pth")
    loss_curve_path = os.path.join(exp_dir, "loss_curve.png")
    loss_history_path = os.path.join(exp_dir, "loss_history.jsonl")

    use_amp = args.amp and device.type == "cuda"
    grad_accum_steps = max(1, args.grad_accum_steps)
    save_experiment_config(exp_dir, args, device, use_amp)
    open(loss_history_path, "w").close()

    print("--- 1. Data Setup ---")
    train_loader, val_loader, train_ds, val_ds, n_channels = make_dataloaders(args)

    print("--- 2. Model Init ---")
    model, selected_model = build_model(args.model_type, n_channels, n_classes=4)
    model = model.to(device)
    print(f"Using model: {selected_model} (input channels={n_channels})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = ImprovedCompositeLoss(lambdas=DEFAULTS["lambdas"], aux_weight=args.aux_weight).to(device)

    print(f"Starting training on {device}...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        tr_loss, tr_comp = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, grad_accum_steps=grad_accum_steps, use_amp=use_amp,
            desc=f"Epoch {epoch + 1}/{args.epochs} [train]",
        )
        val_loss, val_comp = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False, use_amp=use_amp,
            desc=f"Epoch {epoch + 1}/{args.epochs} [val]",
        )
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        write_history_record(loss_history_path, {
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "train_components": tr_comp,
            "val_components": val_comp,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   >> New best val loss {best_val_loss:.4f} — saved.")

        print(f"Epoch {epoch + 1}/{args.epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
        print(f"   >> Train raw: {format_components(tr_comp, RAW_COMPONENTS)}")
        print(f"   >> Train weighted: {format_components(tr_comp, WEIGHTED_COMPONENTS)}")
        print(f"   >> Val raw:   {format_components(val_comp, RAW_COMPONENTS)}")
        print(f"   >> Val weighted: {format_components(val_comp, WEIGHTED_COMPONENTS)}")

    print("--- 3. Saving ---")
    torch.save(model.state_dict(), last_model_path)
    plot_loss_curve(train_losses, val_losses, loss_curve_path, args.experiment_name)


if __name__ == "__main__":
    main()
