"""Training data discovery, Dataset selection, and DataLoader assembly."""

import os
from dataclasses import dataclass
from typing import Any, Dict, Type

import rasterio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import PixelMultiTokenEmbeddingDataset
from .discovery import find_source_pairs, load_split, save_split

TOKEN_SCALE_FACTOR = 16


def token_train_dirs(args):
    return [
        path for path in (
            args.token_train_embeddings_dir,
            getattr(args, "secondary_token_train_embeddings_dir", None),
            getattr(args, "third_token_train_embeddings_dir", None),
            getattr(args, "fourth_token_train_embeddings_dir", None),
        )
        if path
    ]


@dataclass(frozen=True)
class TrainingDatasetSpec:
    dataset_cls: Type
    n_channels: Any
    extra_kwargs: Dict[str, Any]


def discover_training_pairs(args):
    """Find all trainable tuples (2 pixel sources + token sources + label)."""
    token_dirs = token_train_dirs(args)
    if not token_dirs or not args.secondary_train_embeddings_dir:
        raise ValueError(
            "This pipeline needs AlphaEarth (--train-embeddings-dir), Tessera "
            "(--secondary-train-embeddings-dir) and the token sources "
            "(--token-train-embeddings-dir ...)."
        )
    all_pairs = find_source_pairs(
        args.train_embeddings_dir,
        args.secondary_train_embeddings_dir,
        token_dirs,
        args.train_targets_dir,
    )
    if not all_pairs:
        raise ValueError(
            f"No (embedding, label) tuples found.\n"
            f"  train_embeddings_dir='{args.train_embeddings_dir}'\n"
            f"  secondary_train_embeddings_dir='{args.secondary_train_embeddings_dir}'\n"
            f"  token_train_embeddings_dir='{args.token_train_embeddings_dir}'\n"
            f"  train_targets_dir='{args.train_targets_dir}'\n"
            "Check the directory paths."
        )
    return all_pairs


def split_training_pairs(all_pairs, args):
    """Load or create the train/validation split for already-discovered pairs."""
    if args.split_file and os.path.exists(args.split_file):
        return load_split(args.split_file, all_pairs)

    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=args.val_split,
        random_state=args.seed,
    )
    if args.split_file:
        save_split(args.split_file, train_pairs, val_pairs)
    return train_pairs, val_pairs


def raster_channel_count(path):
    """Return the band count for a raster without loading its pixel data."""
    with rasterio.open(path) as src:
        return src.count


def infer_training_dataset_spec(train_pairs, args):
    """Resolve model input channels for the fixed 2-pixel + N-token config."""
    if not train_pairs:
        raise ValueError("Cannot infer dataset spec from an empty training split.")
    sample_pair = train_pairs[0]
    primary_channels = raster_channel_count(sample_pair[0])
    secondary_channels = raster_channel_count(sample_pair[1])
    token_channels = sum(raster_channel_count(path) for path in sample_pair[2:-1])
    return TrainingDatasetSpec(
        dataset_cls=PixelMultiTokenEmbeddingDataset,
        n_channels=(primary_channels + secondary_channels, token_channels),
        extra_kwargs={"scale_factor": TOKEN_SCALE_FACTOR},
    )


def build_training_dataset(dataset_cls, pairs, args, *, is_train, extra_kwargs=None):
    """Instantiate a training Dataset with common patching options."""
    dataset_kwargs = {
        "patch_size": args.patch_size,
        "is_train": is_train,
    }
    if extra_kwargs:
        dataset_kwargs.update(extra_kwargs)
    # Missing-building loss masking: drop the presence/seg loss on flagged
    # (we believe human-deleted) building footprints. Training split only; the
    # val split is scored against the real labels so it must stay unmasked.
    missing_mask_dir = getattr(args, "missing_building_mask_dir", None)
    if missing_mask_dir and is_train:
        dataset_kwargs["missing_mask_dir"] = missing_mask_dir
    return dataset_cls(pairs, **dataset_kwargs)


def build_train_val_datasets(train_pairs, val_pairs, args):
    """Build train/validation Dataset instances and report model input channels."""
    spec = infer_training_dataset_spec(train_pairs, args)
    train_ds = build_training_dataset(
        spec.dataset_cls, train_pairs, args, is_train=True, extra_kwargs=spec.extra_kwargs,
    )
    val_ds = build_training_dataset(
        spec.dataset_cls, val_pairs, args, is_train=False, extra_kwargs=spec.extra_kwargs,
    )
    return train_ds, val_ds, spec.n_channels


def loader_kwargs(args, device):
    """Translate runtime args into PyTorch DataLoader keyword arguments."""
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def build_data_loader(dataset, args, device, *, shuffle):
    return DataLoader(dataset, shuffle=shuffle, **loader_kwargs(args, device))


def build_train_val_loaders(train_ds, val_ds, args, device):
    train_loader = build_data_loader(train_ds, args, device, shuffle=True)
    val_loader = build_data_loader(val_ds, args, device, shuffle=False)
    return train_loader, val_loader


def make_dataloaders(args, device):
    """Build training/validation loaders for the configured data sources."""
    all_pairs = discover_training_pairs(args)
    train_pairs, val_pairs = split_training_pairs(all_pairs, args)
    train_ds, val_ds, n_channels = build_train_val_datasets(train_pairs, val_pairs, args)
    train_loader, val_loader = build_train_val_loaders(train_ds, val_ds, args, device)
    return train_loader, val_loader, train_ds, val_ds, n_channels
