"""Training data discovery, Dataset selection, and DataLoader assembly."""

import os
from dataclasses import dataclass
from typing import Any, Dict, Type

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import (
    MultiPixelEmbeddingDataset,
    PixelMultiTokenEmbeddingDataset,
    PixelTokenEmbeddingDataset,
    pick_dataset_class,
)
from .discovery import (
    find_file_pairs,
    find_multisource_file_pairs,
    find_multitoken_file_pairs,
    find_trisource_file_pairs,
    load_split,
    save_split,
)

TOKEN_SCALE_FACTOR = 16
TOKEN_ZSCORE_EPS = 1e-6


def _parse_token_source_indices(value, n_sources):
    if value is None or value == "":
        return list(range(n_sources))
    if isinstance(value, (list, tuple)):
        indices = [int(v) for v in value]
    else:
        indices = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    bad = [idx for idx in indices if idx < 0 or idx >= n_sources]
    if bad:
        raise ValueError(
            f"token_normalization_source_indices contains invalid index {bad}; "
            f"configured token source count is {n_sources}"
        )
    return indices


def _read_token_for_stats(path):
    with rasterio.open(path) as src:
        array = src.read().astype(np.float32, copy=False)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _load_token_channel_zscore_stats(path):
    with np.load(path) as data:
        source_indices = [int(v) for v in data["source_indices"].tolist()]
        means = data["means"].astype(np.float32)
        stds = data["stds"].astype(np.float32)
    return {
        "source_indices": source_indices,
        "means": means[:, :, None, None],
        "stds": stds[:, :, None, None],
    }


def _compute_token_channel_zscore_stats(train_pairs, source_indices):
    sums = [None for _ in source_indices]
    sumsq = [None for _ in source_indices]
    counts = [0 for _ in source_indices]

    for pair in train_pairs:
        token_paths = pair[2:-1]
        for stat_idx, source_idx in enumerate(source_indices):
            array = _read_token_for_stats(token_paths[source_idx])
            channels = array.reshape(array.shape[0], -1).astype(np.float64, copy=False)
            channel_sum = channels.sum(axis=1)
            channel_sumsq = np.square(channels).sum(axis=1)
            if sums[stat_idx] is None:
                sums[stat_idx] = channel_sum
                sumsq[stat_idx] = channel_sumsq
            else:
                sums[stat_idx] += channel_sum
                sumsq[stat_idx] += channel_sumsq
            counts[stat_idx] += channels.shape[1]

    means = []
    stds = []
    for stat_idx in range(len(source_indices)):
        mean = sums[stat_idx] / counts[stat_idx]
        var = np.maximum(sumsq[stat_idx] / counts[stat_idx] - np.square(mean), 0.0)
        std = np.sqrt(var)
        std = np.where(std < TOKEN_ZSCORE_EPS, 1.0, std)
        means.append(mean.astype(np.float32))
        stds.append(std.astype(np.float32))
    return np.stack(means, axis=0), np.stack(stds, axis=0)


def prepare_token_normalization(train_pairs, args):
    """Resolve token z-score stats: load if cached, else compute and cache."""
    mode = getattr(args, "token_normalization", "none") or "none"
    if mode == "none":
        return None
    if mode != "train_channel_zscore":
        raise ValueError(f"Unknown token_normalization mode: {mode!r}")

    n_sources = len(token_train_dirs(args))
    if n_sources == 0:
        raise ValueError("token_normalization requires token embedding sources")
    source_indices = _parse_token_source_indices(
        getattr(args, "token_normalization_source_indices", None),
        n_sources,
    )
    stats_path = getattr(args, "token_normalization_stats_path", None)
    if not stats_path:
        stats_path = os.path.join(
            args.output_dir,
            args.experiment_name,
            "token_channel_zscore_stats.npz",
        )
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    if os.path.exists(stats_path):
        print(f"Loading token z-score stats: {stats_path}")
        return _load_token_channel_zscore_stats(stats_path)

    print(
        "Computing train-set token channel z-score stats "
        f"for source indices {source_indices}..."
    )
    means, stds = _compute_token_channel_zscore_stats(train_pairs, source_indices)
    np.savez(
        stats_path,
        source_indices=np.asarray(source_indices, dtype=np.int64),
        means=means,
        stds=stds,
        token_dirs=np.asarray(token_train_dirs(args), dtype=str),
    )
    print(f"Saved token z-score stats: {stats_path}")
    return {
        "source_indices": source_indices,
        "means": means[:, :, None, None],
        "stds": stds[:, :, None, None],
    }


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
    """Find all trainable embedding/label tuples for the configured sources."""
    token_dirs = token_train_dirs(args)
    if token_dirs:
        if not args.secondary_train_embeddings_dir:
            raise ValueError(
                "--token-train-embeddings-dir requires --secondary-train-embeddings-dir"
            )
        if len(token_dirs) == 1:
            all_pairs = find_trisource_file_pairs(
                args.train_embeddings_dir,
                args.secondary_train_embeddings_dir,
                token_dirs[0],
                args.train_targets_dir,
            )
        else:
            all_pairs = find_multitoken_file_pairs(
                args.train_embeddings_dir,
                args.secondary_train_embeddings_dir,
                token_dirs,
                args.train_targets_dir,
            )
    elif args.secondary_train_embeddings_dir:
        all_pairs = find_multisource_file_pairs(
            args.train_embeddings_dir,
            args.secondary_train_embeddings_dir,
            args.train_targets_dir,
        )
    else:
        all_pairs = find_file_pairs(args.train_embeddings_dir, args.train_targets_dir)

    if not all_pairs:
        raise ValueError(
            f"No (embedding, label) pairs found.\n"
            f"  train_embeddings_dir='{args.train_embeddings_dir}'\n"
            f"  secondary_train_embeddings_dir='{args.secondary_train_embeddings_dir}'\n"
            f"  token_train_embeddings_dir='{args.token_train_embeddings_dir}'\n"
            f"  train_targets_dir='{args.train_targets_dir}'\n"
            "Check filename conventions and directory paths."
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
    """Resolve dataset class, model input channels, and dataset-only kwargs."""
    if not train_pairs:
        raise ValueError("Cannot infer dataset spec from an empty training split.")

    sample_pair = train_pairs[0]
    primary_channels = raster_channel_count(sample_pair[0])

    token_dirs = token_train_dirs(args)
    if token_dirs:
        secondary_channels = raster_channel_count(sample_pair[1])
        token_channels = sum(raster_channel_count(path) for path in sample_pair[2:-1])
        dataset_cls = (
            PixelTokenEmbeddingDataset if len(token_dirs) == 1
            else PixelMultiTokenEmbeddingDataset
        )
        return TrainingDatasetSpec(
            dataset_cls=dataset_cls,
            n_channels=(primary_channels + secondary_channels, token_channels),
            extra_kwargs={"scale_factor": TOKEN_SCALE_FACTOR},
        )

    if args.secondary_train_embeddings_dir:
        secondary_channels = raster_channel_count(sample_pair[1])
        return TrainingDatasetSpec(
            dataset_cls=MultiPixelEmbeddingDataset,
            n_channels=primary_channels + secondary_channels,
            extra_kwargs={},
        )

    return TrainingDatasetSpec(
        dataset_cls=pick_dataset_class(args.model_type, primary_channels),
        n_channels=primary_channels,
        extra_kwargs={},
    )


def build_training_dataset(dataset_cls, pairs, args, *, is_train, extra_kwargs=None):
    """Instantiate a training Dataset with common patching options."""
    dataset_kwargs = {
        "patch_size": args.patch_size,
        "is_train": is_train,
    }
    if extra_kwargs:
        dataset_kwargs.update(extra_kwargs)
    # Only pass d4_aug to dataset classes that declare it AND only enable for
    # training. Eval datasets always see d4_aug=False (handled by self.is_train
    # gate inside __getitem__ anyway).
    d4_aug = bool(getattr(args, "d4_aug", False)) and is_train
    if d4_aug and "d4_aug" not in dataset_kwargs:
        # Datasets that don't accept d4_aug will quietly raise during construction;
        # only pass when the requested class supports it.
        try:
            import inspect
            params = inspect.signature(dataset_cls).parameters
            if "d4_aug" in params:
                dataset_kwargs["d4_aug"] = d4_aug
        except (TypeError, ValueError):
            pass
    # Missing-building loss masking: drop the presence/seg loss on flagged
    # (we believe human-deleted) building footprints. Training split only; the
    # val split is scored against the real labels so it must stay unmasked.
    missing_mask_dir = getattr(args, "missing_building_mask_dir", None)
    if missing_mask_dir and is_train and "missing_mask_dir" not in dataset_kwargs:
        try:
            import inspect
            if "missing_mask_dir" in inspect.signature(dataset_cls).parameters:
                dataset_kwargs["missing_mask_dir"] = missing_mask_dir
        except (TypeError, ValueError):
            pass
    return dataset_cls(pairs, **dataset_kwargs)


def build_train_val_datasets(train_pairs, val_pairs, args):
    """Build train/validation Dataset instances and report model input channels."""
    spec = infer_training_dataset_spec(train_pairs, args)
    token_normalization = prepare_token_normalization(train_pairs, args)
    extra_kwargs = dict(spec.extra_kwargs)
    if token_normalization is not None:
        extra_kwargs["token_normalization"] = token_normalization
    train_ds = build_training_dataset(
        spec.dataset_cls,
        train_pairs,
        args,
        is_train=True,
        extra_kwargs=extra_kwargs,
    )
    val_ds = build_training_dataset(
        spec.dataset_cls,
        val_pairs,
        args,
        is_train=False,
        extra_kwargs=extra_kwargs,
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
    return DataLoader(
        dataset,
        shuffle=shuffle,
        **loader_kwargs(args, device),
    )


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
