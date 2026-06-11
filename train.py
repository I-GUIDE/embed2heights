"""Train one active emb2heights experiment.

Experiment settings live in YAML presets under ``configs``.
This file is intentionally a thin entrypoint: config parsing, data loading,
and epoch mechanics live under ``core.engine``.
"""

import os

import torch
import torch.optim as optim

from core.losses import ImprovedCompositeLoss
from core.engine import (
    ModelEMA,
    format_components,
    plot_loss_curve,
    run_epoch,
    save_experiment_config,
    save_metrics_summary,
    seed_everything,
    select_device,
    state_dict_for_save,
    write_history_record,
)
from core.models import build_model
from core.config import (
    RAW_COMPONENTS,
    WEIGHTED_COMPONENTS,
    parse_args,
    write_resolved_config,
)
from core.data.training import make_dataloaders


def build_active_model(args, n_channels):
    return build_model(
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
        attn_heads=getattr(args, "attn_heads", 4),
        token_calibration=getattr(args, "token_calibration", False),
        token_calibration_source_indices=getattr(
            args, "token_calibration_source_indices", None),
        use_additive=getattr(args, "use_additive", True),
        token_ctx_ch=getattr(args, "token_ctx_ch", 96),
        token_proj_depth=getattr(args, "token_proj_depth", 1) or 1,
        token_in_source_attn=getattr(args, "token_in_source_attn", False),
        token_cross_source_attn=getattr(args, "token_cross_source_attn", True),
        pixel_noise_std=getattr(args, "pixel_noise_std", 0.0),
        n_head_replicas=getattr(args, "n_head_replicas", 1),
        symmetric_modality_dropout=getattr(args, "symmetric_modality_dropout", 0.0),
        symmetric_modality_dropout_alpha_share=getattr(args, "symmetric_modality_dropout_alpha_share", 0.5),
        height_from_pixel=getattr(args, "height_from_pixel", False),
        feat_aggregation=getattr(args, "feat_aggregation", "mean"),
        token_input_clamp=getattr(args, "token_input_clamp", None),
        pixel_backbone_kind=getattr(args, "pixel_backbone_kind", "unet"),
        use_boundary_head=float(getattr(args, "building_boundary_weight", 0.0) or 0.0) > 0,
        presence_tower_depth=getattr(args, "presence_tower_depth", 0),
        split_trunk=bool(getattr(args, "split_trunk", False)),
    )


def build_loss(args, device):
    criterion = ImprovedCompositeLoss(
        weight_mae=args.weight_mae,
        weight_presence_tversky=args.weight_presence_tversky,
        weight_fraction_mae=args.weight_fraction_mae,
        weight_height_boost=args.weight_height_boost,
        aux_weight=args.aux_weight,
        loss_preset="presence_centered",
        height_loss_kind=args.height_loss_kind,
        huber_delta=args.huber_delta,
        pinball_tau=getattr(args, "pinball_tau", 0.5),
        build_height_boost=args.build_height_boost,
        veg_height_boost=args.veg_height_boost,
        aux_veg_weight=args.aux_veg_weight,
        height_bin_aux_weight=args.height_bin_aux_weight,
        height_bin_sigma_bins=args.height_bin_sigma_bins,
        tversky_building_alpha=args.tversky_building_alpha,
        tversky_water_alpha=args.tversky_water_alpha,
        water_empty_topk=args.water_empty_topk,
        weight_water_empty_topk=args.weight_water_empty_topk,
        building_presence_pos_weight=getattr(args, "building_presence_pos_weight", 1.0),
        small_building_presence_weight=getattr(args, "small_building_presence_weight", 1.0),
        small_building_max_pixels=getattr(args, "small_building_max_pixels", 0),
        building_boundary_weight=getattr(args, "building_boundary_weight", 0.0),
        building_ring_presence_alpha=getattr(args, "building_ring_presence_alpha", 0.0),
        building_ring_kernel=getattr(args, "building_ring_kernel", 5),
    ).to(device)
    print(
        "Using loss: "
        f"preset=presence_centered, "
        f"weight_mae={args.weight_mae}, "
        f"weight_presence_tversky={args.weight_presence_tversky}, "
        f"weight_fraction_mae={args.weight_fraction_mae}, "
        f"weight_height_boost={args.weight_height_boost}, "
        f"aux_weight={args.aux_weight}, "
        f"height_loss_kind={args.height_loss_kind}, "
        f"huber_delta={args.huber_delta}, "
        f"pinball_tau={getattr(args, 'pinball_tau', 0.5)}, "
        f"build_height_boost={args.build_height_boost}, "
        f"veg_height_boost={args.veg_height_boost}, "
        f"aux_veg_weight={args.aux_veg_weight}, "
        f"height_head_kind={args.height_head_kind}, "
        f"height_n_bins={args.height_n_bins}, "
        f"height_bin_max_m={args.height_bin_max_m}, "
        f"tversky_building_alpha={args.tversky_building_alpha}, "
        f"tversky_water_alpha={args.tversky_water_alpha}, "
        f"water_empty_topk={args.water_empty_topk}, "
        f"weight_water_empty_topk={args.weight_water_empty_topk}, "
        f"building_presence_pos_weight={getattr(args, 'building_presence_pos_weight', 1.0)}, "
        f"small_building_presence_weight={getattr(args, 'small_building_presence_weight', 1.0)}, "
        f"small_building_max_pixels={getattr(args, 'small_building_max_pixels', 0)}"
    )
    return criterion


def main():
    args = parse_args()
    device = select_device()
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    best_model_path = os.path.join(exp_dir, "model_best.pth")
    last_model_path = os.path.join(exp_dir, "model_last.pth")
    ema_model_path = os.path.join(exp_dir, "model_ema.pth")
    loss_curve_path = os.path.join(exp_dir, "loss_curve.png")
    loss_history_path = os.path.join(exp_dir, "loss_history.jsonl")

    use_amp = args.amp and device.type == "cuda"
    grad_accum_steps = max(1, args.grad_accum_steps)
    resolved_config_path = write_resolved_config(exp_dir, args, device=device, use_amp=use_amp)
    save_experiment_config(
        exp_dir,
        args,
        device,
        use_amp,
        resolved_config_path=resolved_config_path,
    )
    open(loss_history_path, "w").close()

    print("--- 1. Data Setup ---")
    train_loader, val_loader, _, _, n_channels = make_dataloaders(args, device)

    print("--- 2. Model Init ---")
    model, selected_model = build_active_model(args, n_channels)
    model = model.to(device)
    if getattr(args, "compile", False) and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        model = model.to(memory_format=torch.channels_last)
        model = torch.compile(model, mode="default")
        print("torch.compile enabled (mode='default')")
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(f"Using model: {selected_model} (input channels={n_channels})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # ReduceLROnPlateau params are configurable so noisy training signals
    # (e.g. d4 aug) can use a longer patience to avoid premature LR cascade.
    # Or switch to cosine annealing for noisy regimes that need full-budget decay.
    lr_scheduler_kind = str(getattr(args, "lr_scheduler", "plateau") or "plateau").lower()
    if lr_scheduler_kind == "cosine":
        eta_min = float(getattr(args, "lr_eta_min", 1e-5) or 1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=eta_min
        )
        print(f"LR scheduler: CosineAnnealingLR(T_max={args.epochs}, eta_min={eta_min})")
    else:
        lr_patience = int(getattr(args, "lr_patience", 2) or 2)
        lr_factor = float(getattr(args, "lr_factor", 0.5) or 0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience
        )
        print(f"LR scheduler: ReduceLROnPlateau(factor={lr_factor}, patience={lr_patience})")
    criterion = build_loss(args, device)

    ema_decay = float(getattr(args, "ema_decay", 0.0) or 0.0)
    ema = ModelEMA(model, ema_decay) if ema_decay > 0.0 else None
    if ema is not None:
        print(f"EMA enabled: decay={ema_decay}")

    print(f"Starting training on {device}...")
    train_losses, val_losses = [], []
    train_history = []
    best_val_loss = float("inf")
    best_epoch = None

    ds_weight = float(getattr(args, "deep_supervision_weight", 0.0) or 0.0)
    for epoch in range(args.epochs):
        tr_loss, tr_comp = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, grad_accum_steps=grad_accum_steps, use_amp=use_amp,
            desc=f"Epoch {epoch + 1}/{args.epochs} [train]",
            deep_supervision_weight=ds_weight,
            ema=ema,
        )
        val_loss, val_comp = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False, use_amp=use_amp,
            desc=f"Epoch {epoch + 1}/{args.epochs} [val]",
            deep_supervision_weight=0.0,
        )
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        record = {
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "train_components": tr_comp,
            "val_components": val_comp,
            "lr": optimizer.param_groups[0]["lr"],
        }
        train_history.append(record)
        write_history_record(loss_history_path, record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(state_dict_for_save(model), best_model_path)
            print(f"   >> New best val loss {best_val_loss:.4f} - saved.")

        print(f"Epoch {epoch + 1}/{args.epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
        print(f"   >> Train raw: {format_components(tr_comp, RAW_COMPONENTS)}")
        print(f"   >> Train weighted: {format_components(tr_comp, WEIGHTED_COMPONENTS)}")
        print(f"   >> Val raw:   {format_components(val_comp, RAW_COMPONENTS)}")
        print(f"   >> Val weighted: {format_components(val_comp, WEIGHTED_COMPONENTS)}")

    print("--- 3. Saving ---")
    torch.save(state_dict_for_save(model), last_model_path)
    if ema is not None:
        torch.save(ema.state_dict(), ema_model_path)
        print(f"   >> EMA weights saved to {ema_model_path}")
    plot_loss_curve(train_losses, val_losses, loss_curve_path, args.experiment_name)
    save_metrics_summary(
        exp_dir,
        args=args,
        selected_model=selected_model,
        n_channels=n_channels,
        train_history=train_history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        resolved_config_path=resolved_config_path,
    )


if __name__ == "__main__":
    main()
