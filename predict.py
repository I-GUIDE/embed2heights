"""
Load a trained model and write per-patch predictions as .npy files.

Two modes:
  - Paired (validation): --test-targets-dir given, filenames use `core_id.npy`.
  - Label-free (test set submission): no --test-targets-dir, filenames include
    the year suffix required by the leaderboard (`<core>_<region>_<year>.npy`).
"""
import os
import json
import argparse
import numpy as np
import torch
import rasterio
from tqdm.auto import tqdm

from core.model import build_model
from core.dataset import (
    find_file_pairs,
    find_embedding_files,
    find_multisource_file_pairs,
    find_multisource_embedding_files,
    normalize_core_id,
    submission_id,
    pick_dataset_class,
    MultiPixelEmbeddingDataset,
    HEIGHT_NORM_CONSTANT,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "experiment_name": "terramind_decoder_run01",
    "base_dir": os.path.join(SCRIPT_DIR, "runs"),
    "model_type": "decoder_residual",
    "patch_size": 256,
    "max_samples": 0,
}

MODEL_CHOICES = [
    "auto", "lightunet", "decoder", "decoder_residual", "token_neck",
    "embedding_refiner", "hrnet_w18", "hrnet_w32", "tessera_iou_fusion",
]


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DEFAULTS["experiment_name"])
    parser.add_argument("--base-dir", default=DEFAULTS["base_dir"],
                        help="Root directory containing experiment subfolders.")
    parser.add_argument("--model-type", default=DEFAULTS["model_type"], choices=MODEL_CHOICES,
                        help="Model architecture used during training.")
    parser.add_argument("--model-path", default=None,
                        help="Path to the .pth checkpoint. Defaults to <base-dir>/<experiment-name>/model_best.pth.")
    parser.add_argument("--test-embeddings-dir", required=True,
                        help="Directory containing embedding .tif files.")
    parser.add_argument("--secondary-test-embeddings-dir", default=None,
                        help="Optional second pixel-aligned embedding dir to concatenate with "
                             "--test-embeddings-dir, e.g. Tessera with AlphaEarth.")
    parser.add_argument("--tessera-presence-ch", type=int, default=None,
                        help="Compressed Tessera channels used at training time. "
                             "Defaults to training_params.json when available, else 16.")
    parser.add_argument("--tessera-hidden-ch", type=int, default=None,
                        help="Tessera compressor hidden width used at training time. "
                             "Defaults to training_params.json when available.")
    parser.add_argument("--tessera-hidden-depth", type=int, default=None,
                        help="Extra Tessera compressor hidden depth used at training time. "
                             "Defaults to training_params.json when available, else 0.")
    parser.add_argument("--height-specialist-depth", type=int, default=None,
                        help="Depth of per-class height specialist projections used at "
                             "training time. Defaults to training_params.json when "
                             "available, else 0.")
    parser.add_argument("--lightunet-base-ch", type=int, default=None,
                        help="LightUNet base channel width used at training time. "
                             "Defaults to training_params.json when available, else 32.")
    parser.add_argument("--test-targets-dir", default=None,
                        help="Optional directory of label .tif files. When omitted, "
                             "runs label-free inference (competition test set).")
    parser.add_argument("--predictions-dir", default=None,
                        help="Output directory for .npy predictions. Defaults to <base-dir>/<experiment-name>/predictions.")
    parser.add_argument("--patch-size", type=int, default=DEFAULTS["patch_size"])
    parser.add_argument("--max-samples", type=int, default=DEFAULTS["max_samples"],
                        help="Limit inference to N samples (0 = all).")
    parser.add_argument("--thresholds", type=float, nargs=3, default=None,
                        metavar=("BLD", "VEG", "WAT"),
                        help="Optional per-class thresholds to bake into the output. "
                             "When set, class channels (0-2) are written as {0.0, 1.0} "
                             "using pred > threshold. Default keeps raw sigmoid probs "
                             "(recommended — lets you sweep thresholds later).")
    return parser.parse_args()


def resolve_tessera_model_kwargs(args, exp_dir):
    cfg = {}
    cfg_path = os.path.join(exp_dir, "training_params.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

    return {
        "tessera_presence_ch": (
            args.tessera_presence_ch
            if args.tessera_presence_ch is not None
            else cfg.get("tessera_presence_ch", 16)
        ),
        "tessera_hidden_ch": (
            args.tessera_hidden_ch
            if args.tessera_hidden_ch is not None
            else cfg.get("tessera_hidden_ch", None)
        ),
        "tessera_hidden_depth": (
            args.tessera_hidden_depth
            if args.tessera_hidden_depth is not None
            else cfg.get("tessera_hidden_depth", 0)
        ),
        "height_specialist_depth": (
            args.height_specialist_depth
            if args.height_specialist_depth is not None
            else cfg.get("height_specialist_depth", 0)
        ),
        "lightunet_base_ch": (
            args.lightunet_base_ch
            if args.lightunet_base_ch is not None
            else cfg.get("lightunet_base_ch", 32)
        ),
        # Pulled from training_params.json so the architecture matches what
        # was trained; no CLI flag since it must mirror training exactly.
        "fusion_mode": cfg.get("fusion_mode", "residual_presence"),
    }


def resolve_inputs(args):
    """Return a list of embedding paths (label-free) or (emb, label) tuples."""
    if args.secondary_test_embeddings_dir:
        if args.test_targets_dir:
            pairs = find_multisource_file_pairs(
                args.test_embeddings_dir,
                args.secondary_test_embeddings_dir,
                args.test_targets_dir,
            )
            if not pairs:
                raise RuntimeError("No matching multi-source file pairs found. Check embedding and target dirs.")
            return pairs
        pairs = find_multisource_embedding_files(
            args.test_embeddings_dir,
            args.secondary_test_embeddings_dir,
        )
        if not pairs:
            raise RuntimeError("No matching multi-source .tif files found. Check embedding dirs.")
        return pairs

    if args.test_targets_dir:
        pairs = find_file_pairs(args.test_embeddings_dir, args.test_targets_dir)
        if not pairs:
            raise RuntimeError("No matching file pairs found. Check --test-embeddings-dir and --test-targets-dir.")
        return pairs
    emb_files = find_embedding_files(args.test_embeddings_dir)
    if not emb_files:
        raise RuntimeError(f"No .tif files found in {args.test_embeddings_dir}")
    return emb_files


def main():
    args = parse_args()
    device = select_device()

    exp_dir = os.path.join(args.base_dir, args.experiment_name)
    model_path = args.model_path or os.path.join(exp_dir, "model_best.pth")
    predictions_dir = args.predictions_dir or os.path.join(exp_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    inputs = resolve_inputs(args)
    if args.max_samples > 0:
        inputs = inputs[:args.max_samples]

    sample_emb_path = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
    with rasterio.open(sample_emb_path) as src:
        n_channels = src.count
    if args.secondary_test_embeddings_dir:
        with rasterio.open(inputs[0][1]) as src:
            n_channels += src.count

    DatasetCls = MultiPixelEmbeddingDataset if args.secondary_test_embeddings_dir else pick_dataset_class(args.model_type, n_channels)
    if DatasetCls.__name__ == "LatentTokenDataset":
        test_ds = DatasetCls(inputs, patch_size=args.patch_size, scale_factor=16, is_train=False)
    else:
        test_ds = DatasetCls(inputs, patch_size=args.patch_size, is_train=False)

    sample_img, _, _ = test_ds[0]
    tessera_kwargs = resolve_tessera_model_kwargs(args, exp_dir)
    model, selected_model = build_model(
        args.model_type,
        n_channels=sample_img.shape[0],
        n_classes=4,
        **tessera_kwargs,
    )
    model = model.to(device)
    # torch.compile wraps the model in _orig_mod; strip that prefix at load
    # time so checkpoints from compiled training runs load into the plain
    # (uncompiled) inference model.
    state = torch.load(model_path, map_location=device)
    state = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {selected_model} from {model_path} (input channels={sample_img.shape[0]})")
    if args.thresholds is not None:
        print(f"Baking per-class thresholds into output: bld={args.thresholds[0]}, "
              f"veg={args.thresholds[1]}, wat={args.thresholds[2]}")

    print(f"Running inference on {len(test_ds)} samples...")
    with torch.no_grad():
        pred_shape = None
        for i in tqdm(range(len(test_ds)), desc="Predicting"):
            img_tensor, _, _ = test_ds[i]
            img_batch = img_tensor.unsqueeze(0).to(device)

            output_batch = model(img_batch)
            pred = output_batch.squeeze().cpu().numpy().astype(np.float32)

            # Model emits height / HEIGHT_NORM_CONSTANT; rescale to meters.
            pred[3] = pred[3] * HEIGHT_NORM_CONSTANT

            if args.thresholds is not None:
                for c, t in enumerate(args.thresholds):
                    pred[c] = (pred[c] > t).astype(np.float32)

            emb_path = test_ds.file_pairs[i][0]
            # Submission format keeps the year suffix; val mode normalizes it
            # away so predictions can be matched to labels by core id.
            out_id = submission_id(emb_path) if args.test_targets_dir is None else normalize_core_id(emb_path)
            np.save(os.path.join(predictions_dir, f"{out_id}.npy"), pred)
            pred_shape = pred.shape

    print(f"Predictions saved to: {predictions_dir}")
    if pred_shape is not None:
        print(f"Output shape per file: {pred_shape}  [building%, veg%, water%, height_m]")


if __name__ == "__main__":
    main()
