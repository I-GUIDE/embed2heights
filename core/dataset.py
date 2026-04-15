import os
import glob
import re
import json
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

HEIGHT_NORM_CONSTANT = 30.0

def _normalize_core_id(filename):
    """
    Extracts the pure core ID by stripping all known prefixes,
    embedding suffixes, and year suffixes.

    Handles both train and test naming conventions:
      Train: gee_emb_0000_BE.tif, tessera_emb_0000_BE.tif, s2_0000_BE_2023_embeddings.tif
      Test:  emb_3001_BE_2023_quantized.tif, 3001_BE_2023_merged.tif, s2_3001_BE_2023_embedding.tif
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    # 1. Strip label / prediction prefixes
    if base.startswith("label_"):
        base = base[len("label_"):]
    if base.startswith("pred_"):
        base = base[len("pred_"):]

    # 2. Strip embedding prefixes (order matters: longer prefixes first)
    for prefix in ("gee_emb_", "tessera_emb_", "emb_", "s2_", "s1_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    # 3. Strip trailing embedding/test suffixes
    for suffix in ("_embedding", "_embeddings", "_quantized", "_merged"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]

    # 4. Strip trailing year suffixes (e.g., '_2021', '_2023')
    base = re.sub(r'_\d{4}$', '', base)

    return base


def _submission_id(filename):
    """
    Extracts the submission file id from a test embedding filename,
    preserving the year suffix required by the leaderboard.

    Example: 'emb_3001_BE_2023_quantized.tif' -> '3001_BE_2023'
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    # Strip known prefixes (same set as _normalize_core_id, minus pred_/label_)
    for prefix in ("gee_emb_", "tessera_emb_", "emb_", "s2_", "s1_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    # Strip trailing embedding/test suffixes
    for suffix in ("_embedding", "_embeddings", "_quantized", "_merged"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]

    # Keep the '_YYYY' year suffix — required by submission format.
    return base


def find_file_pairs(emb_dir, tar_dir):
    """
    Fast and robust O(N) file matching using a hash map and regex normalization.
    Searches recursively and guarantees a match regardless of prefixes/suffixes.
    """
    pairs = []

    # 1. Grab ALL files from the disk exactly ONCE
    emb_files = glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True)
    label_files = glob.glob(os.path.join(tar_dir, "**", "label_*.tif"), recursive=True)

    # 2. Build a fast lookup dictionary for the labels: {normalized_id: full_path}
    label_map = {}
    for l_path in label_files:
        norm_id = _normalize_core_id(l_path)
        label_map[norm_id] = l_path

    # 3. Match embeddings to the lookup dictionary instantly
    for e_path in emb_files:
        norm_id = _normalize_core_id(e_path)

        if norm_id in label_map:
            pairs.append((e_path, label_map[norm_id]))

    return pairs


def find_embedding_files(emb_dir):
    """
    List all embedding .tif files without requiring labels.
    Used for competition test-set prediction where no ground truth exists.
    Returns list of embedding file paths.
    """
    emb_files = sorted(glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True))
    return emb_files


def save_split(split_path, train_pairs, val_pairs):
    """Save train/val split to a JSON file using normalized core IDs."""
    data = {
        "train": [_normalize_core_id(e) for e, _ in train_pairs],
        "val": [_normalize_core_id(e) for e, _ in val_pairs],
    }
    with open(split_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Split saved to {split_path} (train={len(data['train'])}, val={len(data['val'])})")


def load_split(split_path, all_pairs):
    """Load a saved split and reconstruct file pair lists."""
    with open(split_path) as f:
        data = json.load(f)

    train_ids = set(data["train"])
    val_ids = set(data["val"])

    train_pairs, val_pairs = [], []
    for pair in all_pairs:
        core_id = _normalize_core_id(pair[0])
        if core_id in train_ids:
            train_pairs.append(pair)
        elif core_id in val_ids:
            val_pairs.append(pair)

    print(f"Split loaded from {split_path} (train={len(train_pairs)}, val={len(val_pairs)})")
    return train_pairs, val_pairs


# ---------------------------------------------------------
# DATASET 1: Pixel-Based (Alpha Earth, Tessera)
# 1:1 Spatial Resolution (e.g., 256x256 -> 256x256)
# ---------------------------------------------------------
class PixelEmbeddingDataset(Dataset):
    """
    For pixel-level embeddings (AlphaEarth 64ch, Tessera 128ch).
    file_pairs: list of (emb_path, label_path) tuples, OR list of emb_path strings (label-free mode).
    """
    def __init__(self, file_pairs, patch_size=128, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train
        # Support both paired and label-free inputs
        if file_pairs and isinstance(file_pairs[0], str):
            self.file_pairs = [(p, None) for p in file_pairs]
        else:
            self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        emb_path, tar_path = self.file_pairs[idx]

        with rasterio.open(emb_path) as src:
            image = src.read().astype(np.float32)
        image = np.nan_to_num(image)

        if tar_path is not None:
            with rasterio.open(tar_path) as src:
                target = src.read().astype(np.float32)
            raw_target = np.nan_to_num(target)
            # Global validity mask: exclude pixels where all 4 bands are zero (nodata)
            global_valid = ~np.all(raw_target == 0, axis=0)  # (H, W) bool
            # nDSM-specific mask: additionally exclude pixels where nDSM==0
            # but land cover is present (nDSM data hole, not true zero height)
            has_landcover = (raw_target[0] > 0) | (raw_target[1] > 0) | (raw_target[2] > 0)
            ndsm_hole = (raw_target[3] == 0) & has_landcover
            height_valid = global_valid & ~ndsm_hole  # (H, W) bool
            # Pack into (2, H, W): channel 0 = global, channel 1 = height-specific
            valid_mask = np.stack([global_valid, height_valid], axis=0).astype(np.float32)
            target = raw_target
            target[3, :, :] = np.maximum(target[3, :, :], 0.0) / HEIGHT_NORM_CONSTANT
        else:
            target = np.zeros((4, image.shape[1], image.shape[2]), dtype=np.float32)
            valid_mask = np.ones((2, image.shape[1], image.shape[2]), dtype=np.float32)

        # 1:1 Padding
        c, h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            valid_mask = np.pad(valid_mask, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            h, w = image.shape[1], image.shape[2]

        # 1:1 Random Cropping
        if self.is_train:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        image = image[:, top:top + self.patch_size, left:left + self.patch_size]
        target = target[:, top:top + self.patch_size, left:left + self.patch_size]
        valid_mask = valid_mask[:, top:top + self.patch_size, left:left + self.patch_size]

        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(valid_mask)

# ---------------------------------------------------------
# DATASET 2: Latent Token-Based (TerraMind, Thor)
# Upscaled Spatial Resolution (e.g., 16x16 -> 256x256)
# ---------------------------------------------------------
class LatentTokenDataset(Dataset):
    """
    For patch-level embeddings (TerraMind 768ch@16x16, THOR 768ch@16x16).
    file_pairs: list of (emb_path, label_path) tuples, OR list of emb_path strings (label-free mode).
    """
    def __init__(self, file_pairs, patch_size=256, scale_factor=16, is_train=True):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        # Support both paired and label-free inputs
        if file_pairs and isinstance(file_pairs[0], str):
            self.file_pairs = [(p, None) for p in file_pairs]
        else:
            self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        emb_path, tar_path = self.file_pairs[idx]

        with rasterio.open(emb_path) as src:
            image = src.read().astype(np.float32)
        image = np.nan_to_num(image)

        emb_patch_size = self.patch_size // self.scale_factor

        if tar_path is not None:
            with rasterio.open(tar_path) as src:
                target = src.read().astype(np.float32)
            raw_target = np.nan_to_num(target)
            global_valid = ~np.all(raw_target == 0, axis=0)
            has_landcover = (raw_target[0] > 0) | (raw_target[1] > 0) | (raw_target[2] > 0)
            ndsm_hole = (raw_target[3] == 0) & has_landcover
            height_valid = global_valid & ~ndsm_hole
            valid_mask = np.stack([global_valid, height_valid], axis=0).astype(np.float32)
            target = raw_target
            target[3, :, :] = np.maximum(target[3, :, :], 0.0) / HEIGHT_NORM_CONSTANT
        else:
            target = np.zeros((4, self.patch_size, self.patch_size), dtype=np.float32)
            valid_mask = np.ones((2, self.patch_size, self.patch_size), dtype=np.float32)

        # Pad Embedding to its specific small size
        c, h_emb, w_emb = image.shape
        if h_emb < emb_patch_size or w_emb < emb_patch_size:
            pad_h = max(0, emb_patch_size - h_emb)
            pad_w = max(0, emb_patch_size - w_emb)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            h_emb, w_emb = image.shape[1], image.shape[2]

        # Pad Target to full size
        _, h_tar, w_tar = target.shape
        if h_tar < self.patch_size or w_tar < self.patch_size:
            pad_h = max(0, self.patch_size - h_tar)
            pad_w = max(0, self.patch_size - w_tar)
            target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            valid_mask = np.pad(valid_mask, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        # Multi-scale Cropping
        if self.is_train:
            top_emb = np.random.randint(0, h_emb - emb_patch_size + 1)
            left_emb = np.random.randint(0, w_emb - emb_patch_size + 1)
        else:
            top_emb = (h_emb - emb_patch_size) // 2
            left_emb = (w_emb - emb_patch_size) // 2

        top_tar = top_emb * self.scale_factor
        left_tar = left_emb * self.scale_factor

        image = image[:, top_emb:top_emb + emb_patch_size, left_emb:left_emb + emb_patch_size]
        target = target[:, top_tar:top_tar + self.patch_size, left_tar:left_tar + self.patch_size]
        valid_mask = valid_mask[:, top_tar:top_tar + self.patch_size, left_tar:left_tar + self.patch_size]

        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(valid_mask)
