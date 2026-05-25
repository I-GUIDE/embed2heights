"""Training loop and run artifact helpers."""

import json
import os
import sys

import torch
from tqdm.auto import tqdm

from .device import move_to_device


def batch_size_of(batch):
    if torch.is_tensor(batch):
        return batch.size(0)
    if isinstance(batch, (tuple, list)) and batch:
        return batch_size_of(batch[0])
    raise TypeError(f"Unsupported batch type for batch size: {type(batch)!r}")


def forward_for_training(model, imgs):
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if getattr(base_model, "supports_aux_outputs", False):
        return model(imgs, return_aux=True)
    return model(imgs)


def run_epoch(model, loader, criterion, optimizer, scaler, device, *, train,
              grad_accum_steps=1, use_amp=False, desc="",
              deep_supervision_weight=0.0):
    """Train or eval one epoch. Returns (avg_loss, component_avgs)."""
    model.train(train)
    running_loss = 0.0
    component_sums = {}
    samples_seen = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    if train:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if train else torch.no_grad()
    non_blocking = device.type == "cuda"
    with context:
        for step, (imgs, targets, masks) in enumerate(pbar, start=1):
            imgs = move_to_device(imgs, device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            masks = masks.to(device, non_blocking=non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = forward_for_training(model, imgs)
                loss, loss_components = criterion(outputs, targets, masks)
                if (deep_supervision_weight > 0.0
                        and isinstance(outputs, dict)
                        and "branch_outs" in outputs
                        and outputs["branch_outs"] is not None):
                    branch_outs = outputs["branch_outs"]
                    branch_losses = []
                    for b_out in branch_outs:
                        bl, _ = criterion(b_out, targets, masks)
                        branch_losses.append(bl)
                    if branch_losses:
                        ds_loss = sum(branch_losses) / len(branch_losses)
                        loss = loss + deep_supervision_weight * ds_loss
                        loss_components["deep_supervision"] = ds_loss.detach()
                step_loss = loss / grad_accum_steps if train else loss

            if not torch.isfinite(loss):
                print(f"\nWARN: non-finite loss at step {step}; skipping batch.")
                if train:
                    optimizer.zero_grad(set_to_none=True)
                del outputs, loss, loss_components, step_loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            if train:
                scaler.scale(step_loss).backward()
                should_step = step % grad_accum_steps == 0 or step == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            bs = batch_size_of(imgs)
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


def save_experiment_config(exp_dir, args, device, use_amp, *, resolved_config_path=None):
    os.makedirs(exp_dir, exist_ok=True)
    resolved_config_path = resolved_config_path or os.path.join(exp_dir, "resolved_config.yml")
    run_metadata_path = os.path.join(exp_dir, "run_metadata.json")
    metadata = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "config": {
            "source_config": args.config,
            "resolved_config_path": resolved_config_path,
        },
        "runtime": {
            "device": str(device),
            "use_amp": bool(use_amp),
            "data_parallel": bool(args.data_parallel),
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args.num_workers > 0,
            "non_blocking_transfer": device.type == "cuda",
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2)",
            "grad_clip": 1.0,
        },
    }
    with open(run_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created experiment folder: {exp_dir}")
    if args.config:
        print(f"Config source: {args.config}")
    print(f"Resolved config: {resolved_config_path}")
    print(f"Run metadata: {run_metadata_path}")


def save_metrics_summary(exp_dir, *, args, selected_model, n_channels,
                         train_history, best_epoch, best_val_loss,
                         resolved_config_path=None):
    """Write a compact run-level training summary for experiment indexing."""
    metrics_path = os.path.join(exp_dir, "metrics_summary.json")
    last = train_history[-1] if train_history else {}
    best = (
        train_history[best_epoch - 1]
        if train_history and best_epoch is not None and 1 <= best_epoch <= len(train_history)
        else {}
    )
    summary = {
        "schema_version": 1,
        "experiment_name": args.experiment_name,
        "source_config": args.config,
        "selected_model": selected_model,
        "model_type": args.model_type,
        "input_channels": list(n_channels) if isinstance(n_channels, tuple) else n_channels,
        "best": {
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "train_loss": best.get("train_loss"),
            "lr": best.get("lr"),
            "train_components": best.get("train_components", {}),
            "val_components": best.get("val_components", {}),
        },
        "last": {
            "epoch": last.get("epoch"),
            "train_loss": last.get("train_loss"),
            "val_loss": last.get("val_loss"),
            "lr": last.get("lr"),
            "train_components": last.get("train_components", {}),
            "val_components": last.get("val_components", {}),
        },
        "artifacts": {
            "best_model": os.path.join(exp_dir, "model_best.pth"),
            "last_model": os.path.join(exp_dir, "model_last.pth"),
            "loss_history": os.path.join(exp_dir, "loss_history.jsonl"),
            "loss_curve": os.path.join(exp_dir, "loss_curve.png"),
            "resolved_config": resolved_config_path or os.path.join(exp_dir, "resolved_config.yml"),
            "run_metadata": os.path.join(exp_dir, "run_metadata.json"),
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics summary: {metrics_path}")
    return metrics_path


def plot_loss_curve(train_losses, val_losses, out_path, experiment_name):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN: matplotlib unavailable; skipping loss curve: {exc}")
        return

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Training Loss Curve ({experiment_name})")
    plt.legend()
    plt.savefig(out_path)
    plt.close()
