"""
Test-Time Augmentation (TTA) inference.

Runs the trained model on N geometric views of each input, inverts the view on
the output, and averages the views. Model is a fully-convolutional U-Net-style
network, so it is equivariant to the D4 group (4 rotations x optional hflip):
input rot/flip -> output rot/flip. Averaging the inverted outputs is therefore
exact (not a heuristic).

Output layout matches predict.py: (4, H, W) with channels
  0: building presence prob (averaged sigmoid)
  1: vegetation presence prob
  2: water presence prob
  3: height in meters (averaged, then rescaled by HEIGHT_NORM_CONSTANT)

Usage (validation on the E-specialist run):
    python predict_tta.py \
        --experiment-name alphaearth_tessera_iou_fusion_E_specialist_d2 \
        --model-type tessera_iou_fusion \
        --test-embeddings-dir /u/dingqi2/workspace/esa/data/train/alphaearth_emb \
        --secondary-test-embeddings-dir /u/dingqi2/workspace/esa/data/train/tessera_emb \
        --test-targets-dir /u/dingqi2/workspace/esa/data/train/labels \
        --tta d4
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
    "tta": "d4",
}

MODEL_CHOICES = [
    "auto", "lightunet", "decoder", "decoder_residual", "token_neck",
    "embedding_refiner", "hrnet_w18", "hrnet_w32", "tessera_iou_fusion",
]

TTA_CHOICES = ["none", "flip", "d4"]


# ---------------- TTA view spec -----------------
# Each view is (k_rot90, do_hflip). Apply: hflip first (if requested), then
# rot90 k times. Inverse: rot90 (-k) times, then hflip again.
# D4 = 8 views; flip = 4 views (rot0/180 x hflip off/on); none = 1 view.

def tta_views(mode: str):
    if mode == "none":
        return [(0, False)]
    if mode == "flip":
        # Identity, hflip, vflip (== rot180 + hflip), rot180
        return [(0, False), (0, True), (2, True), (2, False)]
    if mode == "d4":
        return [(k, h) for k in (0, 1, 2, 3) for h in (False, True)]
    raise ValueError(f"Unknown TTA mode: {mode}")


def apply_view(x: torch.Tensor, k_rot: int, hflip: bool) -> torch.Tensor:
    if hflip:
        x = torch.flip(x, dims=[-1])
    if k_rot:
        x = torch.rot90(x, k=k_rot, dims=[-2, -1])
    return x


def invert_view(x: torch.Tensor, k_rot: int, hflip: bool) -> torch.Tensor:
    if k_rot:
        x = torch.rot90(x, k=-k_rot, dims=[-2, -1])
    if hflip:
        x = torch.flip(x, dims=[-1])
    return x


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DEFAULTS["experiment_name"])
    parser.add_argument("--base-dir", default=DEFAULTS["base_dir"])
    parser.add_argument("--model-type", default=DEFAULTS["model_type"], choices=MODEL_CHOICES)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--secondary-test-embeddings-dir", default=None)
    parser.add_argument("--tessera-presence-ch", type=int, default=None)
    parser.add_argument("--tessera-hidden-ch", type=int, default=None)
    parser.add_argument("--tessera-hidden-depth", type=int, default=None)
    parser.add_argument("--height-specialist-depth", type=int, default=None)
    parser.add_argument("--test-targets-dir", default=None)
    parser.add_argument("--predictions-dir", default=None,
                        help="Output dir for .npy predictions. "
                             "Defaults to <base-dir>/<experiment-name>/predictions_tta.")
    parser.add_argument("--patch-size", type=int, default=DEFAULTS["patch_size"])
    parser.add_argument("--max-samples", type=int, default=DEFAULTS["max_samples"])
    parser.add_argument("--tta", choices=TTA_CHOICES, default=DEFAULTS["tta"],
                        help="TTA mode: none (1 view), flip (4), d4 (8, default).")
    parser.add_argument("--thresholds", type=float, nargs=3, default=None,
                        metavar=("BLD", "VEG", "WAT"),
                        help="Optional per-class thresholds to bake into the output. "
                             "Default keeps raw probs so you can sweep thresholds later.")
    return parser.parse_args()


def resolve_tessera_model_kwargs(args, exp_dir):
    cfg = {}
    cfg_path = os.path.join(exp_dir, "training_params.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    return {
        "tessera_presence_ch": (
            args.tessera_presence_ch if args.tessera_presence_ch is not None
            else cfg.get("tessera_presence_ch", 16)
        ),
        "tessera_hidden_ch": (
            args.tessera_hidden_ch if args.tessera_hidden_ch is not None
            else cfg.get("tessera_hidden_ch", None)
        ),
        "tessera_hidden_depth": (
            args.tessera_hidden_depth if args.tessera_hidden_depth is not None
            else cfg.get("tessera_hidden_depth", 0)
        ),
        "height_specialist_depth": (
            args.height_specialist_depth if args.height_specialist_depth is not None
            else cfg.get("height_specialist_depth", 0)
        ),
    }


def resolve_inputs(args):
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
            args.test_embeddings_dir, args.secondary_test_embeddings_dir,
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


def main():
    args = parse_args()
    device = select_device()

    exp_dir = os.path.join(args.base_dir, args.experiment_name)
    model_path = args.model_path or os.path.join(exp_dir, "model_best.pth")
    predictions_dir = args.predictions_dir or os.path.join(exp_dir, "predictions_tta")
    os.makedirs(predictions_dir, exist_ok=True)

    views = tta_views(args.tta)

    inputs = resolve_inputs(args)
    if args.max_samples > 0:
        inputs = inputs[:args.max_samples]

    sample_emb_path = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
    with rasterio.open(sample_emb_path) as src:
        n_channels = src.count
    if args.secondary_test_embeddings_dir:
        with rasterio.open(inputs[0][1]) as src:
            n_channels += src.count

    DatasetCls = (
        MultiPixelEmbeddingDataset if args.secondary_test_embeddings_dir
        else pick_dataset_class(args.model_type, n_channels)
    )
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {selected_model} from {model_path} (input channels={sample_img.shape[0]})")
    print(f"TTA mode: {args.tta}  ({len(views)} views per sample)")
    if args.thresholds is not None:
        print(f"Baking per-class thresholds: bld={args.thresholds[0]}, "
              f"veg={args.thresholds[1]}, wat={args.thresholds[2]}")

    print(f"Running inference on {len(test_ds)} samples...")
    with torch.no_grad():
        pred_shape = None
        for i in tqdm(range(len(test_ds)), desc="TTA-predicting"):
            img_tensor, _, _ = test_ds[i]
            img_batch = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)

            # Stack views into a batch for one forward pass per view count.
            view_inputs = torch.cat(
                [apply_view(img_batch, k, h) for (k, h) in views], dim=0
            )
            view_outputs = model(view_inputs)  # (V, 4, H, W)

            # Invert each view and average in raw output space (sigmoid for 0-2,
            # normalized height for 3 -- both linear, so the mean is meaningful).
            inverted = torch.stack([
                invert_view(view_outputs[v:v + 1], k, h)[0]
                for v, (k, h) in enumerate(views)
            ], dim=0)  # (V, 4, H, W)
            pred_tensor = inverted.mean(dim=0)  # (4, H, W)

            pred = pred_tensor.cpu().numpy().astype(np.float32)
            pred[3] = pred[3] * HEIGHT_NORM_CONSTANT

            if args.thresholds is not None:
                for c, t in enumerate(args.thresholds):
                    pred[c] = (pred[c] > t).astype(np.float32)

            emb_path = test_ds.file_pairs[i][0]
            out_id = submission_id(emb_path) if args.test_targets_dir is None else normalize_core_id(emb_path)
            np.save(os.path.join(predictions_dir, f"{out_id}.npy"), pred)
            pred_shape = pred.shape

    print(f"Predictions saved to: {predictions_dir}")
    if pred_shape is not None:
        print(f"Output shape per file: {pred_shape}  [building%, veg%, water%, height_m]")


if __name__ == "__main__":
    main()
