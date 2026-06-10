"""Training CLI and YAML recipe configuration."""

import argparse
import os

try:
    import yaml
except ImportError:
    yaml = None


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_DIR, "configs", "defaults.yml")

MODEL_CHOICES = [
    "ae_only",
    "ae_tessera_gated",
    "xfusion_unet_hybrid_cross_source",
    "xfusion_unet_per_source_ensemble",
    "auto",
]
CONFIG_SECTIONS = ("data", "model", "training", "runtime")
RECIPE_METADATA_KEYS = ("name", "description", "reference")

RAW_COMPONENTS = (
    "mae",
    "fraction_mae",
    "ssim",
    "grad",
    "tversky",
    "height_boost",
    "presence_bce",
    "presence_tversky",
    "water_empty_topk",
    "aux_height_building",
    "aux_height_vegetation",
    "height_bin_ce",
    "building_boundary",
    "building_smooth",
    "height_error_bce",
    "delta_sparsity",
    "token_aux",
    "token_aux_fraction_mae",
    "token_aux_height_boost",
    "token_aux_presence_bce",
    "token_aux_presence_tversky",
    "token_aux_height_building",
    "token_aux_height_vegetation",
)

WEIGHTED_COMPONENTS = (
    "weighted_mae",
    "weighted_ssim",
    "weighted_grad",
    "weighted_tversky",
    "weighted_height_boost",
    "weighted_presence_bce",
    "weighted_presence_tversky",
    "weighted_water_empty_topk",
    "weighted_aux_height",
    "weighted_height_bin_ce",
    "weighted_building_boundary",
    "weighted_building_smooth",
    "weighted_height_error_bce",
    "weighted_delta_sparsity",
    "weighted_token_aux",
)


def _read_yaml_mapping(config_path):
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for --config. Install pyyaml or run with CLI arguments only."
        )
    with open(config_path, "r") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return loaded


def _resolve_placeholders(value):
    if not isinstance(value, str):
        return value
    value = value.replace("${REPO_DIR}", REPO_DIR)
    if "/" in value or value.startswith("."):
        return os.path.abspath(value)
    return value


def _flatten_config(loaded, *, source_config=None):
    defaults = {}
    if source_config is not None:
        defaults["config"] = source_config
    for section in CONFIG_SECTIONS:
        values = loaded.get(section, {})
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
        defaults.update({
            key: _resolve_placeholders(value)
            for key, value in values.items()
        })
    return defaults


def load_config_defaults(config_path):
    """Load a YAML run recipe and flatten supported sections into arg defaults."""
    if not config_path:
        return {}
    return _flatten_config(_read_yaml_mapping(config_path), source_config=config_path)


def load_recipe_metadata(config_path):
    if not config_path:
        return {}
    loaded = _read_yaml_mapping(config_path)
    return {
        key: loaded[key]
        for key in RECIPE_METADATA_KEYS
        if key in loaded
    }


DEFAULTS = _flatten_config(_read_yaml_mapping(DEFAULT_CONFIG_PATH))


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,
                     help="YAML run recipe. CLI runtime flags override values from the file.")
    config_args, _ = pre.parse_known_args()
    config_defaults = load_config_defaults(config_args.config)

    p = argparse.ArgumentParser(description="Train a single emb2heights backbone.", parents=[pre])
    p.add_argument("--experiment-name")
    p.add_argument("--output-dir")
    p.add_argument("--model-type", choices=MODEL_CHOICES)
    p.add_argument("--train-embeddings-dir")
    p.add_argument("--secondary-train-embeddings-dir",
                   help="Second pixel-aligned embedding dir, e.g. Tessera.")
    p.add_argument("--token-train-embeddings-dir",
                   help="Optional 16x16 token embedding dir for xfusion.")
    p.add_argument("--secondary-token-train-embeddings-dir",
                   help="Optional second 16x16 token embedding dir for xfusion.")
    p.add_argument("--third-token-train-embeddings-dir",
                   help="Optional third 16x16 token embedding dir for xfusion.")
    p.add_argument("--fourth-token-train-embeddings-dir",
                   help="Optional fourth 16x16 token embedding dir for xfusion.")
    p.add_argument("--token-normalization",
                   choices=("none", "train_channel_zscore"),
                   help="Optional token source normalization applied after reading rasters.")
    p.add_argument("--token-normalization-source-indices",
                   help="Comma-separated 0-based token source indices to normalize. "
                        "Empty means all token sources.")
    p.add_argument("--token-normalization-stats-path",
                   help="Path to token z-score stats .npz. Defaults to the run directory.")
    p.add_argument("--train-targets-dir")
    p.add_argument("--split-file",
                   help="Path to a JSON split file. Loaded if present, else saved there.")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--grad-accum-steps", type=int,
                   help="Accumulate gradients over N mini-batches before optimizer step.")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--prefetch-factor", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="Apply torch.compile(mode='default') + channels_last + high matmul "
                        "precision for a balanced (~20-30%%) speedup. Requires PyTorch >= 2.0.")
    p.add_argument("--data-parallel", action=argparse.BooleanOptionalAction)
    p.add_argument("--freeze-except",
                   help="Freeze all params whose name does NOT contain this substring. "
                        "Used for two-stage training: freeze pixel path, train token path only.")
    p.add_argument("--freeze-epochs", type=int,
                   help="Number of epochs to keep pixel path frozen (Stage 1). "
                        "After this, all params are unfrozen for Stage 2 fine-tuning.")
    p.add_argument("--stage2-lr", type=float,
                   help="Learning rate for Stage 2 (all params unfrozen). "
                        "Defaults to lr * 0.1 if not set.")
    p.add_argument("--boundary-building-weight", type=float,
                   help="Upweight BCE for background pixels within boundary_kernel_size "
                        "of a GT building. 1.0 = disabled (default).")
    p.add_argument("--boundary-kernel-size", type=int,
                   help="Dilation kernel size (pixels) for boundary building weight.")
    p.add_argument("--tversky-building-alpha", type=float,
                   help="Tversky alpha (FP penalty) for building class only. "
                        "Lower = more recall-focused. Default 0.3.")
    p.add_argument("--tversky-water-alpha", type=float,
                   help="Tversky alpha (FP penalty) for water class only.")
    p.add_argument("--water-empty-topk", type=int,
                   help="Top-k water probabilities to penalize on patches with no water.")
    p.add_argument("--weight-water-empty-topk", type=float,
                   help="Weight for the empty-water top-k penalty.")
    p.add_argument("--use-fraction-film", action=argparse.BooleanOptionalAction,
                   help="Use predicted fractions as FiLM conditioning for height.")
    p.add_argument("--use-fraction-aux", action=argparse.BooleanOptionalAction,
                   help="Keep the fraction auxiliary head/loss even when height FiLM is disabled.")
    p.add_argument("--deep-supervision-weight", type=float,
                   help="Per-branch aux-loss weight for per-source ensemble models. "
                        "0 disables; typical 0.2-0.5.")
    p.add_argument("--token-proj-depth", type=int,
                   help="Per-source-ensemble: depth of each branch's token "
                        "projection (768 -> ctx_ch). 1=linear (default), "
                        "2=GN+GELU+linear.")
    p.add_argument("--token-in-source-attn", action=argparse.BooleanOptionalAction,
                   help="Run per-source spatial self-attention before the "
                        "cross-source token attention in hybrid token fusion.")
    p.add_argument("--token-cross-source-attn", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Run cross-source token attention after optional per-source "
                        "spatial self-attention in hybrid token fusion.")
    p.add_argument("--height-from-pixel", action=argparse.BooleanOptionalAction,
                   help="Per-source-ensemble: route the height trunk through the "
                        "pre-token-fusion pixel feature (alpha+tessera gate output) "
                        "while presence still uses the token-fused feat_i. "
                        "Decouples height regression from token smoothing.")
    p.add_argument("--building-presence-pos-weight", type=float,
                   help="Extra BCE weight for positive building pixels in the presence head.")
    p.add_argument("--small-building-presence-weight", type=float,
                   help="Additional BCE weight for positive building pixels on small-building tiles.")
    p.add_argument("--small-building-max-pixels", type=int,
                   help="Tile-level positive building pixel cutoff for small-building BCE weighting.")
    p.add_argument("--building-boundary-weight", type=float,
                   help="Stage D: weight for the building-boundary auxiliary head "
                        "(BCE+Dice on the GT building edge ring). >0 activates the "
                        "boundary head and its loss; 0 disables (default).")
    p.add_argument("--presence-tower-depth", type=int,
                   help="P3 head: number of ConvGNAct blocks in the presence-only tower.")

    p.set_defaults(**DEFAULTS)
    p.set_defaults(**config_defaults)
    args = p.parse_args()
    if args.presence_tower_depth is None:
        args.presence_tower_depth = 0
    args.presence_tversky_weight = args.weight_presence_tversky
    args.fraction_mae_weight = args.weight_fraction_mae
    return args


def build_resolved_config(args, *, device=None, use_amp=None):
    """Return the final nested run recipe after YAML defaults and CLI overrides."""
    runtime = {
        "experiment_name": args.experiment_name,
        "output_dir": args.output_dir,
        "amp": args.amp,
        "data_parallel": args.data_parallel,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
    }
    if device is not None:
        runtime["device"] = str(device)
    if use_amp is not None:
        runtime["use_amp"] = bool(use_amp)

    return {
        "schema_version": 2,
        "source_config": args.config,
        "recipe": load_recipe_metadata(args.config),
        "data": {
            "train_embeddings_dir": args.train_embeddings_dir,
            "secondary_train_embeddings_dir": args.secondary_train_embeddings_dir,
            "token_train_embeddings_dir": args.token_train_embeddings_dir,
            "secondary_token_train_embeddings_dir": args.secondary_token_train_embeddings_dir,
            "third_token_train_embeddings_dir": args.third_token_train_embeddings_dir,
            "fourth_token_train_embeddings_dir": args.fourth_token_train_embeddings_dir,
            "token_normalization": args.token_normalization,
            "token_normalization_source_indices": args.token_normalization_source_indices,
            "token_normalization_stats_path": args.token_normalization_stats_path,
            "train_targets_dir": args.train_targets_dir,
            "split_file": args.split_file,
            "patch_size": args.patch_size,
            "d4_aug": getattr(args, "d4_aug", False),
        },
        "model": {
            "model_type": args.model_type,
            "tessera_presence_ch": args.tessera_presence_ch,
            "tessera_hidden_ch": args.tessera_hidden_ch,
            "tessera_hidden_depth": args.tessera_hidden_depth,
            "height_specialist_depth": args.height_specialist_depth,
            "height_gate_source": args.height_gate_source,
            "height_hidden_ch": args.height_hidden_ch,
            "height_trunk_depth": args.height_trunk_depth,
            "height_independent_branches": args.height_independent_branches,
            "height_head_kind": args.height_head_kind,
            "height_n_bins": args.height_n_bins,
            "height_bin_max_m": args.height_bin_max_m,
            "lightunet_base_ch": args.lightunet_base_ch,
            "lightunet_norm_kind": args.lightunet_norm_kind,
            "gate_mode": args.gate_mode,
            "gate_untied": args.gate_untied,
            "gate_init_bias": args.gate_init_bias,
            "modality_dropout": args.modality_dropout,
            "token_calibration": getattr(args, "token_calibration", False),
            "token_calibration_source_indices": getattr(args, "token_calibration_source_indices", None),
            "pixel_noise_std": getattr(args, "pixel_noise_std", 0.0),
            "n_head_replicas": getattr(args, "n_head_replicas", 1),
            "symmetric_modality_dropout": getattr(args, "symmetric_modality_dropout", 0.0),
            "symmetric_modality_dropout_alpha_share": getattr(args, "symmetric_modality_dropout_alpha_share", 0.5),
            "token_ctx_ch": getattr(args, "token_ctx_ch", 96),
            "presence_head_kind": args.presence_head_kind,
            "presence_head_depth": args.presence_head_depth,
            "presence_branch_ch": args.presence_branch_ch,
            "use_fraction_film": args.use_fraction_film,
            "use_fraction_aux": args.use_fraction_aux,
            "attn_heads": getattr(args, "attn_heads", 4),
            "use_additive": getattr(args, "use_additive", True),
            "token_proj_depth": getattr(args, "token_proj_depth", 1) or 1,
            "token_in_source_attn": getattr(args, "token_in_source_attn", False),
            "token_cross_source_attn": getattr(args, "token_cross_source_attn", True),
            "height_from_pixel": getattr(args, "height_from_pixel", False),
            "feat_aggregation": getattr(args, "feat_aggregation", "mean"),
            "token_input_clamp": getattr(args, "token_input_clamp", None),
            "pixel_backbone_kind": getattr(args, "pixel_backbone_kind", "unet"),
            "use_boundary_head": float(getattr(args, "building_boundary_weight", 0.0) or 0.0) > 0,
            "presence_tower_depth": getattr(args, "presence_tower_depth", 0),
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_patience": getattr(args, "lr_patience", 2),
            "lr_factor": getattr(args, "lr_factor", 0.5),
            "seed": args.seed,
            "grad_accum_steps": args.grad_accum_steps,
            "freeze_except": args.freeze_except,
            "freeze_epochs": args.freeze_epochs,
            "stage2_lr": args.stage2_lr,
            "loss_preset": "presence_centered",
            "weight_mae": args.weight_mae,
            "weight_presence_tversky": args.weight_presence_tversky,
            "weight_fraction_mae": args.weight_fraction_mae,
            "weight_height_boost": args.weight_height_boost,
            "aux_weight": args.aux_weight,
            "height_loss_kind": args.height_loss_kind,
            "huber_delta": args.huber_delta,
            "pinball_tau": getattr(args, "pinball_tau", 0.5),
            "build_height_boost": args.build_height_boost,
            "veg_height_boost": args.veg_height_boost,
            "aux_veg_weight": args.aux_veg_weight,
            "height_bin_aux_weight": args.height_bin_aux_weight,
            "height_bin_sigma_bins": args.height_bin_sigma_bins,
            "boundary_building_weight": args.boundary_building_weight,
            "boundary_kernel_size": args.boundary_kernel_size,
            "tversky_building_alpha": args.tversky_building_alpha,
            "tversky_water_alpha": args.tversky_water_alpha,
            "water_empty_topk": args.water_empty_topk,
            "weight_water_empty_topk": args.weight_water_empty_topk,
            "deep_supervision_weight": getattr(args, "deep_supervision_weight", 0.0),
            "building_presence_pos_weight": getattr(args, "building_presence_pos_weight", 1.0),
            "small_building_presence_weight": getattr(args, "small_building_presence_weight", 1.0),
            "small_building_max_pixels": getattr(args, "small_building_max_pixels", 0),
            "building_boundary_weight": getattr(args, "building_boundary_weight", 0.0),
            "ema_decay": getattr(args, "ema_decay", 0.0),
            "lr_scheduler": getattr(args, "lr_scheduler", "plateau"),
            "lr_eta_min": getattr(args, "lr_eta_min", 1e-5),
            "lr_patience": getattr(args, "lr_patience", 2),
            "lr_factor": getattr(args, "lr_factor", 0.5),
        },
        "runtime": runtime,
    }


def write_resolved_config(exp_dir, args, *, device=None, use_amp=None):
    """Write the final run recipe after YAML defaults and CLI overrides are merged."""
    if yaml is None:
        return None
    resolved = build_resolved_config(args, device=device, use_amp=use_amp)
    os.makedirs(exp_dir, exist_ok=True)
    out_path = os.path.join(exp_dir, "resolved_config.yml")
    with open(out_path, "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    return out_path
