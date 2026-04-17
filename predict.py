"""
Load a trained model and write per-patch predictions as .npy files.

Two modes:
  - Paired (validation): --test-targets-dir given, filenames use `core_id.npy`.
  - Label-free (test set submission): no --test-targets-dir, filenames include
    the year suffix required by the leaderboard (`<core>_<region>_<year>.npy`).
"""
import os
import argparse
import numpy as np
import torch
import rasterio
from tqdm.auto import tqdm

from core.model import build_model
from core.dataset import (
    find_file_pairs,
    find_embedding_files,
    normalize_core_id,
    submission_id,
    pick_dataset_class,
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
    "auto", "lightunet", "decoder", "decoder_residual",
    "embedding_refiner", "hrnet_w18", "hrnet_w32",
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
    parser.add_argument("--test-targets-dir", default=None,
                        help="Optional directory of label .tif files. When omitted, "
                             "runs label-free inference (competition test set).")
    parser.add_argument("--predictions-dir", default=None,
                        help="Output directory for .npy predictions. Defaults to <base-dir>/<experiment-name>/predictions.")
    parser.add_argument("--patch-size", type=int, default=DEFAULTS["patch_size"])
    parser.add_argument("--max-samples", type=int, default=DEFAULTS["max_samples"],
                        help="Limit inference to N samples (0 = all).")
    return parser.parse_args()


def resolve_inputs(args):
    """Return a list of embedding paths (label-free) or (emb, label) tuples."""
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

    DatasetCls = pick_dataset_class(args.model_type, n_channels)
    if DatasetCls.__name__ == "LatentTokenDataset":
        test_ds = DatasetCls(inputs, patch_size=args.patch_size, scale_factor=16, is_train=False)
    else:
        test_ds = DatasetCls(inputs, patch_size=args.patch_size, is_train=False)

    sample_img, _, _ = test_ds[0]
    model, selected_model = build_model(args.model_type, n_channels=sample_img.shape[0], n_classes=4)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {selected_model} from {model_path} (input channels={sample_img.shape[0]})")

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
