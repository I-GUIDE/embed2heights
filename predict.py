"""Load an active checkpoint and write per-patch .npy predictions."""

import argparse
import json
import os

import numpy as np
import rasterio
import torch
from tqdm.auto import tqdm

try:
    import yaml
except ImportError:
    yaml = None

from core.data.datasets import PixelMultiTokenEmbeddingDataset
from core.data.discovery import find_source_pairs, find_source_tuples
from core.models import build_model
from core.engine import move_to_device, select_device
from core.inference import (
    batched,
    input_channels,
    predict_batch,
    prediction_output_id,
    prediction_to_numpy,
    write_prediction_config,
)
from core.config import DEFAULTS as TRAIN_DEFAULTS
from core.config import MODEL_CHOICES


CONFIG_SECTIONS = ("data", "model", "training", "runtime")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="run01")
    parser.add_argument("--base-dir", default=TRAIN_DEFAULTS["output_dir"],
                        help="Root directory containing experiment subfolders.")
    parser.add_argument("--model-type", default=None, choices=MODEL_CHOICES,
                        help="Override model type. Defaults to resolved_config.yml, else auto.")
    parser.add_argument("--model-path", default=None,
                        help="Defaults to <base-dir>/<experiment-name>/model_best.pth.")
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--secondary-test-embeddings-dir", default=None,
                        help="Second pixel-aligned embedding dir, e.g. Tessera.")
    parser.add_argument("--token-test-embeddings-dir", default=None,
                        help="Optional 16x16 token embedding dir for xfusion.")
    parser.add_argument("--secondary-token-test-embeddings-dir", default=None,
                        help="Optional second 16x16 token embedding dir for xfusion.")
    parser.add_argument("--third-token-test-embeddings-dir", default=None,
                        help="Optional third 16x16 token embedding dir for xfusion.")
    parser.add_argument("--fourth-token-test-embeddings-dir", default=None,
                        help="Optional fourth 16x16 token embedding dir for xfusion.")
    parser.add_argument("--test-targets-dir", default=None,
                        help="When omitted, writes label-free submission ids with year suffix.")
    parser.add_argument("--predictions-dir", default=None,
                        help="Defaults to <base-dir>/<experiment-name>/predictions.")
    parser.add_argument("--patch-size", type=int, default=TRAIN_DEFAULTS["patch_size"])
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit inference to N samples (0 = all).")
    parser.add_argument("--thresholds", type=float, nargs=3, default=None,
                        metavar=("BLD", "VEG", "WAT"),
                        help="Optional per-class thresholds baked into channels 0-2.")
    parser.add_argument("--restrict-val-split", default=None,
                        help="Keep only inputs whose core (first two underscore fields, e.g. "
                             "'0041_FQ') is in this split.json's 'val' list. Used to write "
                             "out-of-fold validation predictions for threshold tuning.")
    return parser.parse_args()


def _core_of_input(inp, label_mode):
    """First two underscore fields of the prediction id, e.g. '0041_FQ'."""
    emb_path = inp[0] if isinstance(inp, (tuple, list)) else inp
    oid = prediction_output_id(emb_path, label_mode=label_mode)
    return "_".join(oid.split("_")[:2])


def load_legacy_training_params(exp_dir):
    cfg_path = os.path.join(exp_dir, "training_params.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r") as f:
        return json.load(f)


def flatten_run_config(config):
    flat = {}
    for section in CONFIG_SECTIONS:
        values = config.get(section, {})
        if isinstance(values, dict):
            flat.update(values)
    flat["_config_source"] = "resolved_config.yml"
    flat["_source_config"] = config.get("source_config")
    flat["_recipe"] = config.get("recipe", {})
    return flat


def load_training_config(exp_dir):
    resolved_path = os.path.join(exp_dir, "resolved_config.yml")
    if yaml is not None and os.path.exists(resolved_path):
        with open(resolved_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            return flatten_run_config(loaded), loaded

    flat = load_legacy_training_params(exp_dir)
    flat["_config_source"] = "legacy training_params.json" if flat else "defaults"
    flat["_source_config"] = flat.get("source_config")
    flat["_recipe"] = {}
    return flat, None


def model_kwargs_from_run_config(cfg):
    default_base_ch = TRAIN_DEFAULTS["lightunet_base_ch"]
    return {
        "height_specialist_depth": cfg.get("height_specialist_depth", TRAIN_DEFAULTS["height_specialist_depth"]),
        "height_hidden_ch": cfg.get("height_hidden_ch", TRAIN_DEFAULTS["height_hidden_ch"]),
        "height_trunk_depth": cfg.get("height_trunk_depth", TRAIN_DEFAULTS["height_trunk_depth"]),
        "lightunet_base_ch": cfg.get("lightunet_base_ch", default_base_ch),
        "lightunet_norm_kind": cfg.get("lightunet_norm_kind", TRAIN_DEFAULTS["lightunet_norm_kind"]),
        "height_n_bins": cfg.get("height_n_bins", TRAIN_DEFAULTS["height_n_bins"]),
        "height_bin_max_m": cfg.get("height_bin_max_m", TRAIN_DEFAULTS["height_bin_max_m"]),
        "gate_mode": cfg.get("gate_mode", TRAIN_DEFAULTS["gate_mode"]),
        "gate_untied": cfg.get("gate_untied", TRAIN_DEFAULTS["gate_untied"]),
        "gate_init_bias": cfg.get("gate_init_bias", TRAIN_DEFAULTS["gate_init_bias"]),
        "modality_dropout": cfg.get("modality_dropout", TRAIN_DEFAULTS["modality_dropout"]),
        "presence_head_depth": cfg.get("presence_head_depth", TRAIN_DEFAULTS["presence_head_depth"]),
        "presence_branch_ch": cfg.get("presence_branch_ch", TRAIN_DEFAULTS["presence_branch_ch"]),
        "use_fraction_aux": cfg.get("use_fraction_aux", TRAIN_DEFAULTS["use_fraction_aux"]),
        "attn_heads": cfg.get("attn_heads", 4),
        "token_calibration": cfg.get("token_calibration", False),
        "token_calibration_source_indices": cfg.get("token_calibration_source_indices", None),
        "token_ctx_ch": cfg.get("token_ctx_ch", 96),
        "pixel_backbone_kind": cfg.get("pixel_backbone_kind", "unet"),
        "presence_tower_depth": cfg.get("presence_tower_depth", 0),
        "split_trunk": bool(cfg.get("split_trunk", False)),
        "presence_trunk_grad_scale": cfg.get("presence_trunk_grad_scale", 1.0),
        "height_trunk_grad_scale": cfg.get("height_trunk_grad_scale", 1.0),
    }


def token_test_dirs(args):
    return [
        path for path in (
            args.token_test_embeddings_dir,
            args.secondary_token_test_embeddings_dir,
            args.third_token_test_embeddings_dir,
            args.fourth_token_test_embeddings_dir,
        )
        if path
    ]


def resolve_inputs(args):
    """Return source tuples: (AE, Tessera, *tokens[, label]) for the fixed
    2-pixel + N-token config. Includes the label when --test-targets-dir is set
    (validation mode); otherwise label-free (test mode)."""
    token_dirs = token_test_dirs(args)
    if not token_dirs or not args.secondary_test_embeddings_dir:
        raise RuntimeError(
            "This pipeline needs --secondary-test-embeddings-dir (Tessera) and "
            "the token sources (--token-test-embeddings-dir ...)."
        )
    if args.test_targets_dir:
        pairs = find_source_pairs(
            args.test_embeddings_dir, args.secondary_test_embeddings_dir,
            token_dirs, args.test_targets_dir,
        )
    else:
        pairs = find_source_tuples(
            args.test_embeddings_dir, args.secondary_test_embeddings_dir, token_dirs,
        )
    if not pairs:
        raise RuntimeError("No matching source tuples found.")
    return pairs


def infer_channels_and_dataset(args, inputs):
    """Resolve ((pixel_channels, token_channels), dataset_cls) for the fixed
    2-pixel + N-token config."""
    sample_tuple = inputs[0]
    with rasterio.open(sample_tuple[0]) as src:
        pixel_channels = src.count
    with rasterio.open(sample_tuple[1]) as src:
        pixel_channels += src.count
    token_slice = sample_tuple[2:-1] if args.test_targets_dir else sample_tuple[2:]
    token_channels = 0
    for path in token_slice:
        with rasterio.open(path) as src:
            token_channels += src.count
    return (pixel_channels, token_channels), PixelMultiTokenEmbeddingDataset


def build_dataset(dataset_cls, inputs, patch_size):
    return dataset_cls(inputs, patch_size=patch_size, scale_factor=16, is_train=False)


def main():
    args = parse_args()
    device = select_device()

    exp_dir = os.path.join(args.base_dir, args.experiment_name)
    model_path = args.model_path or os.path.join(exp_dir, "model_best.pth")
    predictions_dir = args.predictions_dir or os.path.join(exp_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    train_cfg, train_config_raw = load_training_config(exp_dir)
    model_type = args.model_type or train_cfg.get("model_type", "auto")

    inputs = resolve_inputs(args)
    if args.restrict_val_split:
        val = set(json.load(open(args.restrict_val_split))["val"])
        label_mode = args.test_targets_dir is not None
        before = len(inputs)
        inputs = [x for x in inputs if _core_of_input(x, label_mode) in val]
        print(f"restrict-val-split: kept {len(inputs)}/{before} inputs in val of {args.restrict_val_split}")
    if args.max_samples > 0:
        inputs = inputs[:args.max_samples]

    n_channels, dataset_cls = infer_channels_and_dataset(args, inputs)
    test_ds = build_dataset(dataset_cls, inputs, args.patch_size)
    sample_img, _, _ = test_ds[0]

    model, selected_model = build_model(
        model_type,
        n_channels=input_channels(sample_img),
        n_classes=4,
        **model_kwargs_from_run_config(train_cfg),
    )
    model = model.to(device)
    sd = torch.load(model_path, map_location=device)
    # Strip torch.compile's _orig_mod. prefix if the checkpoint was saved from a compiled model.
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        print(f"WARNING: missing keys in checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"INFO: extra keys in checkpoint (unused by current arch): {result.unexpected_keys}")
    model.eval()

    print(f"Loaded model: {selected_model} from {model_path} (input channels={input_channels(sample_img)})")
    if args.thresholds is not None:
        print(f"Baking thresholds into output: bld={args.thresholds[0]}, "
              f"veg={args.thresholds[1]}, wat={args.thresholds[2]}")
    print(f"Training config source: {train_cfg.get('_config_source')}")

    print(f"Running inference on {len(test_ds)} samples...")
    with torch.no_grad():
        pred_shape = None
        for i in tqdm(range(len(test_ds)), desc="Predicting"):
            img_tensor, _, _ = test_ds[i]
            img_batch = move_to_device(batched(img_tensor), device)

            pred = prediction_to_numpy(
                predict_batch(model, img_batch),
                thresholds=args.thresholds,
            )

            emb_path = test_ds.file_pairs[i][0]
            out_id = prediction_output_id(
                emb_path,
                label_mode=args.test_targets_dir is not None,
            )
            np.save(os.path.join(predictions_dir, f"{out_id}.npy"), pred)
            pred_shape = pred.shape

    print(f"Predictions saved to: {predictions_dir}")
    if pred_shape is not None:
        print(f"Output shape per file: {pred_shape}  [building%, veg%, water%, height_m]")
    prediction_config_path = os.path.join(predictions_dir, "prediction_config.json")
    write_prediction_config(
        prediction_config_path,
        args=args,
        exp_dir=exp_dir,
        model_path=model_path,
        predictions_dir=predictions_dir,
        selected_model=selected_model,
        n_channels=n_channels,
        dataset_cls=dataset_cls,
        train_cfg=train_cfg,
        train_config_raw=train_config_raw,
        sample_count=len(test_ds),
        pred_shape=pred_shape,
    )
    print(f"Prediction config saved to: {prediction_config_path}")


if __name__ == "__main__":
    main()
