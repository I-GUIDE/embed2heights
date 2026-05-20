"""Training CLI and YAML recipe configuration."""

import argparse
import os

try:
    import yaml
except ImportError:
    yaml = None


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_DIR, "configs", "defaults.yml")

MODEL_CHOICES = ["ae_only", "ae_tessera", "ae_tessera_gated", "ae_tessera_mlp", "ae_tessera_moe", "xfusion_crosslevel", "xfusion_pp", "hier_gated", "auto"]
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
    "aux_height_building",
    "aux_height_vegetation",
    "height_bin_ce",
    "building_smooth",
)

WEIGHTED_COMPONENTS = (
    "weighted_mae",
    "weighted_ssim",
    "weighted_grad",
    "weighted_tversky",
    "weighted_height_boost",
    "weighted_presence_bce",
    "weighted_presence_tversky",
    "weighted_aux_height",
    "weighted_height_bin_ce",
    "weighted_building_smooth",
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
    p.add_argument("--data-parallel", action=argparse.BooleanOptionalAction)
    p.add_argument("--init-from-pretrain",
                   help="Optional self-supervised pretrain checkpoint.")

    p.set_defaults(**DEFAULTS)
    p.set_defaults(**config_defaults)
    args = p.parse_args()
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
        "init_from_pretrain": args.init_from_pretrain,
        "init_pretrain_strict": args.init_pretrain_strict,
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
            "train_targets_dir": args.train_targets_dir,
            "split_file": args.split_file,
            "patch_size": args.patch_size,
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
            "presence_head_kind": args.presence_head_kind,
            "presence_head_depth": args.presence_head_depth,
            "presence_branch_ch": args.presence_branch_ch,
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "grad_accum_steps": args.grad_accum_steps,
            "loss_preset": "presence_centered",
            "weight_mae": args.weight_mae,
            "weight_presence_tversky": args.weight_presence_tversky,
            "weight_fraction_mae": args.weight_fraction_mae,
            "weight_height_boost": args.weight_height_boost,
            "aux_weight": args.aux_weight,
            "height_loss_kind": args.height_loss_kind,
            "huber_delta": args.huber_delta,
            "build_height_boost": args.build_height_boost,
            "veg_height_boost": args.veg_height_boost,
            "aux_veg_weight": args.aux_veg_weight,
            "height_bin_aux_weight": args.height_bin_aux_weight,
            "height_bin_sigma_bins": args.height_bin_sigma_bins,
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
