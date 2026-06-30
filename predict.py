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

from core.data.datasets import (
    MultiPixelEmbeddingDataset,
    PixelTokenEmbeddingDataset,
    pick_dataset_class,
)
from core.data.discovery import (
    find_embedding_files,
    find_file_pairs,
    find_multisource_embedding_files,
    find_multisource_file_pairs,
    find_trisource_embedding_files,
    find_trisource_file_pairs,
)
from core.models import build_model
from core.engine import move_to_device, select_device
from core.inference import (
    batched,
    input_channels,
    predict_batch,
    prediction_output_id,
    prediction_to_numpy,
    tta_views,
    write_prediction_config,
)
from core.config import DEFAULTS as TRAIN_DEFAULTS
from core.config import MODEL_CHOICES


TTA_CHOICES = ("none", "flip", "d4")
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
    parser.add_argument("--tta", default="none", choices=TTA_CHOICES,
                        help="Test-time augmentation mode. 'flip' uses identity + h/v flips; "
                             "'d4' uses rotations plus mirrored rotations.")
    parser.add_argument("--adabn", action="store_true",
                        help="Adaptive Batch Normalization: do a no-grad forward pass over the "
                             "test/inference inputs with BN layers in training mode (updates "
                             "running stats to match the inference distribution) before the "
                             "final eval-mode pass. Parameter-free domain adaptation that "
                             "neutralizes regional style shifts (per-region illumination, "
                             "sensor noise, etc.) without risking catastrophic divergence.")
    return parser.parse_args()


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
    model_type = str(cfg.get("model_type", "")).lower()
    default_base_ch = 32 if model_type in {"lightunet", "ae_only"} else TRAIN_DEFAULTS["lightunet_base_ch"]
    return {
        "tessera_presence_ch": cfg.get("tessera_presence_ch", TRAIN_DEFAULTS["tessera_presence_ch"]),
        "tessera_hidden_ch": cfg.get("tessera_hidden_ch", TRAIN_DEFAULTS["tessera_hidden_ch"]),
        "tessera_hidden_depth": cfg.get("tessera_hidden_depth", TRAIN_DEFAULTS["tessera_hidden_depth"]),
        "height_specialist_depth": cfg.get("height_specialist_depth", TRAIN_DEFAULTS["height_specialist_depth"]),
        "height_gate_source": cfg.get("height_gate_source", TRAIN_DEFAULTS["height_gate_source"]),
        "height_hidden_ch": cfg.get("height_hidden_ch", TRAIN_DEFAULTS["height_hidden_ch"]),
        "height_trunk_depth": cfg.get("height_trunk_depth", TRAIN_DEFAULTS["height_trunk_depth"]),
        "height_independent_branches": cfg.get(
            "height_independent_branches", TRAIN_DEFAULTS["height_independent_branches"]
        ),
        "lightunet_base_ch": cfg.get("lightunet_base_ch", default_base_ch),
        "lightunet_norm_kind": cfg.get("lightunet_norm_kind", TRAIN_DEFAULTS["lightunet_norm_kind"]),
        "height_head_kind": cfg.get("height_head_kind", TRAIN_DEFAULTS["height_head_kind"]),
        "height_n_bins": cfg.get("height_n_bins", TRAIN_DEFAULTS["height_n_bins"]),
        "height_bin_max_m": cfg.get("height_bin_max_m", TRAIN_DEFAULTS["height_bin_max_m"]),
        "gate_mode": cfg.get("gate_mode", TRAIN_DEFAULTS["gate_mode"]),
        "gate_untied": cfg.get("gate_untied", TRAIN_DEFAULTS["gate_untied"]),
        "gate_init_bias": cfg.get("gate_init_bias", TRAIN_DEFAULTS["gate_init_bias"]),
        "modality_dropout": cfg.get("modality_dropout", TRAIN_DEFAULTS["modality_dropout"]),
        "presence_head_kind": cfg.get("presence_head_kind", TRAIN_DEFAULTS["presence_head_kind"]),
        "presence_head_depth": cfg.get("presence_head_depth", TRAIN_DEFAULTS["presence_head_depth"]),
        "presence_branch_ch": cfg.get("presence_branch_ch", TRAIN_DEFAULTS["presence_branch_ch"]),
        "bidirectional_ctask": cfg.get("bidirectional_ctask", False),
        "height_blend_mode": cfg.get("height_blend_mode", "presence_gated"),
        "dual_presence": cfg.get("dual_presence", False),
        "ae_only_supervision": (cfg.get("ae_only_deep_sup_weight", 0.0) or 0.0) > 0.0,
        "use_se": cfg.get("use_se", False),
        "use_coord_attn": cfg.get("use_coord_attn", False),
        "use_bottleneck_attn": cfg.get("use_bottleneck_attn", False),
        "use_mixstyle": cfg.get("use_mixstyle", False),
        "use_attn_gates": cfg.get("use_attn_gates", False),
        "use_aspp": cfg.get("use_aspp", False),
        "bottleneck_attn_depth": cfg.get("bottleneck_attn_depth", 1),
        "use_modern": cfg.get("use_modern", False),
        "detail_bypass": cfg.get("detail_bypass", False),
        "sharp_upsample": cfg.get("sharp_upsample", False),
        "scene_film": cfg.get("scene_film", False),
        "encoder_arch": cfg.get("encoder_arch", "unet"),
        "disable_head_film": cfg.get("disable_head_film", False),
        "use_xsource_fusion": cfg.get("use_xsource_fusion", False),
        "token_source_ch": cfg.get("token_source_ch", 768),
        "token_ctx_ch": cfg.get("token_ctx_ch", 96),
        "xsource_attn_heads": cfg.get("xsource_attn_heads", 4),
        "xsource_token_calibration": cfg.get("xsource_token_calibration", False),
        "use_spatial_token_film": cfg.get("use_spatial_token_film", False),
        "use_shape_queries": cfg.get("use_shape_queries", False),
        "shape_n_queries": cfg.get("shape_n_queries", 32),
        "shape_depth": cfg.get("shape_depth", 2),
        # MultiBackboneFusion: needed to reconstruct the identical architecture
        # (proj_ch drives the stem in_chans / adapter; input_norm changes the
        # forward; source/freeze are inert at predict but kept for fidelity).
        "pretrained_backbone_path": cfg.get("pretrained_backbone_path", None),
        "backbone_input_proj_ch": cfg.get("backbone_input_proj_ch", None),
        "backbone_input_norm": cfg.get("backbone_input_norm", None),
        "backbone_pretrained_source": cfg.get("backbone_pretrained_source", None),
        "freeze_backbone_stages": cfg.get("freeze_backbone_stages", 0),
    }


def resolve_inputs(args):
    """Return embedding paths, or tuples ending in labels for validation mode."""
    if args.token_test_embeddings_dir:
        if not args.secondary_test_embeddings_dir:
            raise RuntimeError("--token-test-embeddings-dir requires --secondary-test-embeddings-dir")
        if args.test_targets_dir:
            pairs = find_trisource_file_pairs(
                args.test_embeddings_dir,
                args.secondary_test_embeddings_dir,
                args.token_test_embeddings_dir,
                args.test_targets_dir,
            )
            if not pairs:
                raise RuntimeError("No matching tri-source file pairs found.")
            return pairs
        pairs = find_trisource_embedding_files(
            args.test_embeddings_dir,
            args.secondary_test_embeddings_dir,
            args.token_test_embeddings_dir,
        )
        if not pairs:
            raise RuntimeError("No matching tri-source .tif files found.")
        return pairs

    if args.secondary_test_embeddings_dir:
        if args.test_targets_dir:
            pairs = find_multisource_file_pairs(
                args.test_embeddings_dir,
                args.secondary_test_embeddings_dir,
                args.test_targets_dir,
            )
            if not pairs:
                raise RuntimeError("No matching multi-source file pairs found.")
            return pairs
        pairs = find_multisource_embedding_files(
            args.test_embeddings_dir,
            args.secondary_test_embeddings_dir,
        )
        if not pairs:
            raise RuntimeError("No matching multi-source .tif files found.")
        return pairs

    if args.test_targets_dir:
        pairs = find_file_pairs(args.test_embeddings_dir, args.test_targets_dir)
        if not pairs:
            raise RuntimeError("No matching file pairs found.")
        return pairs
    emb_files = find_embedding_files(args.test_embeddings_dir)
    if not emb_files:
        raise RuntimeError(f"No .tif files found in {args.test_embeddings_dir}")
    return emb_files


def infer_channels_and_dataset(args, inputs):
    sample_emb_path = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
    with rasterio.open(sample_emb_path) as src:
        n_channels = src.count

    if args.token_test_embeddings_dir:
        with rasterio.open(inputs[0][1]) as src:
            pixel_channels = n_channels + src.count
        with rasterio.open(inputs[0][2]) as src:
            token_channels = src.count
        return (pixel_channels, token_channels), PixelTokenEmbeddingDataset

    if args.secondary_test_embeddings_dir:
        with rasterio.open(inputs[0][1]) as src:
            n_channels += src.count
        return n_channels, MultiPixelEmbeddingDataset

    return n_channels, pick_dataset_class(args.model_type or "auto", n_channels)


def build_dataset(dataset_cls, inputs, patch_size):
    if dataset_cls.__name__ == "PixelTokenEmbeddingDataset":
        return dataset_cls(inputs, patch_size=patch_size, scale_factor=16, is_train=False)
    return dataset_cls(inputs, patch_size=patch_size, is_train=False)


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
    raw_sd = torch.load(model_path, map_location=device)
    # Strip torch.compile prefix (_orig_mod.*) if present
    if any(k.startswith("_orig_mod.") for k in raw_sd):
        raw_sd = {k[len("_orig_mod."):]: v for k, v in raw_sd.items()}
    # Remap legacy key names: UpsampleBlock renamed self.norm → self.bn
    sd = {k.replace(".norm.", ".bn."): v for k, v in raw_sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected checkpoint keys (ignored): {unexpected[:3]}...")
    if missing:
        bn_missing = [k for k in missing if "running_mean" in k or "running_var" in k or "num_batches" in k]
        other_missing = [k for k in missing if k not in bn_missing]
        if bn_missing:
            print(f"  INFO: {len(bn_missing)} BN running-stat keys not in checkpoint (legacy GroupNorm ckpt) — using defaults")
        if other_missing:
            print(f"  WARN: {len(other_missing)} non-stat keys missing: {other_missing[:3]}...")
    if args.adabn:
        print(f"AdaBN: running no-grad forward pass over {len(test_ds)} samples to update BN stats...")
        # Reset all BN running statistics so they accumulate purely from
        # inference distribution. Then run forward in train() mode under
        # no_grad to update running_mean / running_var per batch without
        # touching the learned affine parameters or any other weight.
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.reset_running_stats()
        model.train()
        with torch.no_grad():
            for i in tqdm(range(len(test_ds)), desc="AdaBN BN-update"):
                img_tensor, _, _ = test_ds[i]
                img_batch = move_to_device(batched(img_tensor), device)
                model(img_batch)
    model.eval()

    print(f"Loaded model: {selected_model} from {model_path} (input channels={input_channels(sample_img)})")
    if args.thresholds is not None:
        print(f"Baking thresholds into output: bld={args.thresholds[0]}, "
              f"veg={args.thresholds[1]}, wat={args.thresholds[2]}")
    views = tta_views(args.tta)
    if args.tta != "none":
        print(f"TTA enabled: {args.tta} ({len(views)} views)")
    print(f"Training config source: {train_cfg.get('_config_source')}")

    print(f"Running inference on {len(test_ds)} samples...")
    with torch.no_grad():
        pred_shape = None
        for i in tqdm(range(len(test_ds)), desc="Predicting"):
            img_tensor, _, _ = test_ds[i]
            img_batch = move_to_device(batched(img_tensor), device)

            pred = prediction_to_numpy(
                predict_batch(model, img_batch, views),
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
