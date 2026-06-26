"""PyTorch datasets for competition raster embeddings."""

import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


HEIGHT_NORM_CONSTANT = 30.0


def clean_raster_array(array):
    """Convert raster data to finite float32 values."""
    array = array.astype(np.float32, copy=False)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _read_raster(path):
    with rasterio.open(path) as src:
        return clean_raster_array(src.read())


def _normalize_token_arrays(token_arrays, token_normalization):
    """Apply per-channel z-score to selected token sources in-place by index."""
    if not token_normalization:
        return token_arrays
    source_indices = token_normalization["source_indices"]
    means = token_normalization["means"]
    stds = token_normalization["stds"]
    arrays = list(token_arrays)
    for stat_idx, source_idx in enumerate(source_indices):
        arrays[source_idx] = (arrays[source_idx] - means[stat_idx]) / stds[stat_idx]
    return arrays


def _assert_same_spatial(left, right, left_path, right_path, kind):
    if left.shape[1:] != right.shape[1:]:
        raise ValueError(
            f"{kind} shapes do not align for {left_path} and {right_path}: "
            f"{left.shape[1:]} vs {right.shape[1:]}"
        )


def _pad_to_min_shape(array, min_shape, *, mode, constant_values=0):
    h, w = array.shape[-2:]
    min_h, min_w = min_shape
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return array

    pad_width = ((0, 0), (0, pad_h), (0, pad_w))
    if mode == "constant":
        return np.pad(array, pad_width, mode=mode, constant_values=constant_values)
    return np.pad(array, pad_width, mode=mode)


def _pad_pixel_training_tensors(image, target, valid_mask, patch_size):
    image = _pad_to_min_shape(image, (patch_size, patch_size), mode="reflect")
    target = _pad_to_min_shape(target, (patch_size, patch_size), mode="reflect")
    valid_mask = _pad_to_min_shape(
        valid_mask,
        (patch_size, patch_size),
        mode="constant",
        constant_values=0,
    )
    return image, target, valid_mask


def _crop_chw(array, top, left, height, width=None):
    if width is None:
        width = height
    return array[:, top:top + height, left:left + width]


def _apply_d4(array, rot_k, mirror):
    """Apply a D4 symmetry (rot ∈ {0..3} × mirror ∈ {True,False}) to a CHW
    numpy array. Used for training-time geometric augmentation.
    Both pixel images and tokens must share the same (rot_k, mirror) draw so
    the spatial correspondence is preserved.
    """
    if rot_k:
        array = np.rot90(array, k=rot_k, axes=(-2, -1))
    if mirror:
        array = np.flip(array, axis=-1)
    # rot90/flip return non-contiguous views; copy so torch.from_numpy is happy
    return np.ascontiguousarray(array)


def _sample_or_center_origin(height, width, crop_size, is_train):
    if is_train:
        return (
            np.random.randint(0, height - crop_size + 1),
            np.random.randint(0, width - crop_size + 1),
        )
    return (height - crop_size) // 2, (width - crop_size) // 2


def pick_dataset_class(model_type, n_channels):
    """Single-source datasets only handle pixel rasters in the clean repo."""
    return PixelEmbeddingDataset


def _tile_core_from_path(path):
    """'.../label_1597_GD_2023.tif' -> '1597_GD' (matches missing-mask filenames)."""
    import re
    m = re.search(r"(\d{4}_[A-Z]{2})", os.path.basename(path))
    return m.group(1) if m else None


def _apply_missing_mask(valid_mask, tar_path, missing_mask_dir):
    """Zero the GLOBAL (presence/seg) validity channel on flagged missing-building
    pixels so the model is NOT penalized for predicting building where the footprint
    label was deleted. Height validity (channel 1) is left intact — the nDSM there is
    real building height and we keep supervising it (mirrors the ndsm_hole convention,
    but for the presence label instead of the height label).
    `valid_mask` is (2, H, W); edited in place and returned."""
    core = _tile_core_from_path(tar_path)
    if core is None:
        return valid_mask
    mpath = os.path.join(missing_mask_dir, f"{core}.npy")
    if not os.path.exists(mpath):
        return valid_mask
    mm = np.load(mpath)
    h = min(mm.shape[0], valid_mask.shape[1])
    w = min(mm.shape[1], valid_mask.shape[2])
    valid_mask[0, :h, :w] = valid_mask[0, :h, :w] * (1.0 - mm[:h, :w].astype(np.float32))
    return valid_mask


def _prepare_target(tar_path, image_shape, patch_size=None):
    if tar_path is not None:
        with rasterio.open(tar_path) as src:
            raw_target = clean_raster_array(src.read())
        global_valid = ~np.all(raw_target == 0, axis=0)
        has_landcover = (raw_target[0] > 0) | (raw_target[1] > 0) | (raw_target[2] > 0)
        ndsm_hole = (raw_target[3] == 0) & has_landcover
        height_valid = global_valid & ~ndsm_hole
        valid_mask = np.stack([global_valid, height_valid], axis=0).astype(np.float32)
        target = raw_target
        target[3, :, :] = np.maximum(target[3, :, :], 0.0) / HEIGHT_NORM_CONSTANT
        return target, valid_mask

    h, w = image_shape if patch_size is None else (patch_size, patch_size)
    target = np.zeros((4, h, w), dtype=np.float32)
    valid_mask = np.ones((2, h, w), dtype=np.float32)
    return target, valid_mask


class PixelEmbeddingDataset(Dataset):
    """
    For pixel-level embeddings (AlphaEarth 64ch, Tessera 128ch).
    file_pairs: list of (emb_path, label_path) tuples, OR list of emb_path strings (label-free mode).
    """
    def __init__(self, file_pairs, patch_size=128, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train
        if file_pairs and isinstance(file_pairs[0], str):
            self.file_pairs = [(p, None) for p in file_pairs]
        else:
            self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        emb_path, tar_path = self.file_pairs[idx]

        image = _read_raster(emb_path)

        target, valid_mask = _prepare_target(tar_path, image.shape[1:])

        image, target, valid_mask = _pad_pixel_training_tensors(
            image,
            target,
            valid_mask,
            self.patch_size,
        )
        _, h, w = image.shape
        top, left = _sample_or_center_origin(h, w, self.patch_size, self.is_train)

        image = _crop_chw(image, top, left, self.patch_size)
        target = _crop_chw(target, top, left, self.patch_size)
        valid_mask = _crop_chw(valid_mask, top, left, self.patch_size)

        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(valid_mask)


class MultiPixelEmbeddingDataset(Dataset):
    """
    For concatenating two pixel-aligned embedding sources, e.g. AlphaEarth
    64ch + Tessera 128ch -> 192ch at 256x256.

    file_pairs may contain:
      - (primary_emb_path, secondary_emb_path, label_path)
      - (primary_emb_path, secondary_emb_path) for label-free inference
    """
    def __init__(self, file_pairs, patch_size=128, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train
        self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pair = self.file_pairs[idx]
        if len(pair) == 3:
            primary_path, secondary_path, tar_path = pair
        elif len(pair) == 2:
            primary_path, secondary_path = pair
            tar_path = None
        else:
            raise ValueError("MultiPixelEmbeddingDataset expects 2- or 3-item tuples")

        primary = _read_raster(primary_path)
        secondary = _read_raster(secondary_path)

        _assert_same_spatial(primary, secondary, primary_path, secondary_path, "Embedding")
        image = np.concatenate([primary, secondary], axis=0)
        target, valid_mask = _prepare_target(tar_path, image.shape[1:])

        image, target, valid_mask = _pad_pixel_training_tensors(
            image,
            target,
            valid_mask,
            self.patch_size,
        )
        _, h, w = image.shape
        top, left = _sample_or_center_origin(h, w, self.patch_size, self.is_train)

        image = _crop_chw(image, top, left, self.patch_size)
        target = _crop_chw(target, top, left, self.patch_size)
        valid_mask = _crop_chw(valid_mask, top, left, self.patch_size)

        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(valid_mask)


class PixelTokenEmbeddingDataset(Dataset):
    """
    For probing one 16x16 token source against the AlphaEarth+Tessera champion.

    file_pairs may contain:
      - (primary_emb_path, secondary_emb_path, token_emb_path, label_path)
      - (primary_emb_path, secondary_emb_path, token_emb_path) for label-free inference

    Returns ((pixel_image, token_image), target, valid_mask), where pixel_image is
    AlphaEarth+Tessera concatenated at 256x256 and token_image is 768x16x16 for
    patch_size=256, scale_factor=16.
    """
    def __init__(self, file_pairs, patch_size=128, scale_factor=16, is_train=True,
                 token_normalization=None, d4_aug=False):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.file_pairs = file_pairs
        self.token_normalization = token_normalization
        # D4 dihedral augmentation: see PixelMultiTokenEmbeddingDataset doc.
        self.d4_aug = bool(d4_aug)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pair = self.file_pairs[idx]
        if len(pair) == 4:
            primary_path, secondary_path, token_path, tar_path = pair
        elif len(pair) == 3:
            primary_path, secondary_path, token_path = pair
            tar_path = None
        else:
            raise ValueError("PixelTokenEmbeddingDataset expects 3- or 4-item tuples")

        primary = _read_raster(primary_path)
        secondary = _read_raster(secondary_path)
        token = _read_raster(token_path)
        # Single-token dataset: normalize as a 1-element list and unpack.
        token = _normalize_token_arrays([token], self.token_normalization)[0]

        _assert_same_spatial(primary, secondary, primary_path, secondary_path, "Embedding")
        pixel = np.concatenate([primary, secondary], axis=0)
        target, valid_mask = _prepare_target(tar_path, pixel.shape[1:], patch_size=self.patch_size)

        emb_patch_size = self.patch_size // self.scale_factor

        pixel, target, valid_mask = _pad_pixel_training_tensors(
            pixel,
            target,
            valid_mask,
            self.patch_size,
        )
        token = _pad_to_min_shape(token, (emb_patch_size, emb_patch_size), mode="reflect")

        _, h_pix, w_pix = pixel.shape
        _, h_tok, w_tok = token.shape
        max_top_emb = min(h_tok - emb_patch_size, (h_pix - self.patch_size) // self.scale_factor)
        max_left_emb = min(w_tok - emb_patch_size, (w_pix - self.patch_size) // self.scale_factor)
        if max_top_emb < 0 or max_left_emb < 0:
            raise ValueError(
                f"Pixel/token shapes are incompatible for {primary_path} and {token_path}: "
                f"pixel={pixel.shape[1:]}, token={token.shape[1:]}"
            )

        if self.is_train:
            top_emb = np.random.randint(0, max_top_emb + 1)
            left_emb = np.random.randint(0, max_left_emb + 1)
        else:
            top_emb = max_top_emb // 2
            left_emb = max_left_emb // 2

        top_pix = top_emb * self.scale_factor
        left_pix = left_emb * self.scale_factor

        pixel = _crop_chw(pixel, top_pix, left_pix, self.patch_size)
        token = _crop_chw(token, top_emb, left_emb, emb_patch_size)
        target = _crop_chw(target, top_pix, left_pix, self.patch_size)
        valid_mask = _crop_chw(valid_mask, top_pix, left_pix, self.patch_size)

        # D4 augmentation (training only). Same draw applied to pixel/token/target
        # so spatial correspondence (and 16:1 token-pixel scale) is preserved.
        if self.is_train and self.d4_aug:
            rot_k = np.random.randint(0, 4)
            mirror = bool(np.random.randint(0, 2))
            pixel      = _apply_d4(pixel,      rot_k, mirror)
            token      = _apply_d4(token,      rot_k, mirror)
            target     = _apply_d4(target,     rot_k, mirror)
            valid_mask = _apply_d4(valid_mask, rot_k, mirror)

        return (
            torch.from_numpy(pixel),
            torch.from_numpy(token),
        ), torch.from_numpy(target), torch.from_numpy(valid_mask)


class PixelMultiTokenEmbeddingDataset(Dataset):
    """
    For probing one or more token sources against the AlphaEarth+Tessera champion.

    file_pairs may contain:
      - (primary_emb_path, secondary_emb_path, *token_paths, label_path)
      - (primary_emb_path, secondary_emb_path, *token_paths) for label-free inference

    Returns ((pixel_image, token_image), target, valid_mask), where pixel_image
    is AlphaEarth+Tessera concatenated at 256x256 and token_image is
    channel-concatenated token sources at 16x16.
    """
    def __init__(self, file_pairs, patch_size=128, scale_factor=16, is_train=True,
                 token_normalization=None, d4_aug=False, missing_mask_dir=None):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.file_pairs = file_pairs
        self.token_normalization = token_normalization
        # D4 dihedral augmentation: 8 orientations (4 rotations × {identity, mirror}).
        # Applied AFTER cropping; pixel + token + target share the same draw so
        # the 16:1 spatial correspondence is preserved. Training only.
        self.d4_aug = bool(d4_aug)
        # Optional dir of per-tile missing-building masks; when set (training only)
        # the presence/seg loss is dropped on flagged pixels (see _apply_missing_mask).
        self.missing_mask_dir = missing_mask_dir

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pair = self.file_pairs[idx]
        if len(pair) < 4:
            raise ValueError("PixelMultiTokenEmbeddingDataset expects at least 4-item tuples")
        primary_path, secondary_path = pair[:2]
        if len(pair) % 2 == 1:
            token_paths = pair[2:-1]
            tar_path = pair[-1]
        else:
            token_paths = pair[2:]
            tar_path = None
        if not token_paths:
            raise ValueError("PixelMultiTokenEmbeddingDataset requires at least one token path")

        primary = _read_raster(primary_path)
        secondary = _read_raster(secondary_path)
        token_arrays = [_read_raster(path) for path in token_paths]
        token_arrays = _normalize_token_arrays(token_arrays, self.token_normalization)

        _assert_same_spatial(primary, secondary, primary_path, secondary_path, "Embedding")
        for token_path, token_array in zip(token_paths[1:], token_arrays[1:]):
            _assert_same_spatial(
                token_arrays[0],
                token_array,
                token_paths[0],
                token_path,
                "Token",
            )
        pixel = np.concatenate([primary, secondary], axis=0)
        token = np.concatenate(token_arrays, axis=0)
        target, valid_mask = _prepare_target(tar_path, pixel.shape[1:], patch_size=self.patch_size)
        if self.is_train and self.missing_mask_dir and tar_path is not None:
            valid_mask = _apply_missing_mask(valid_mask, tar_path, self.missing_mask_dir)

        emb_patch_size = self.patch_size // self.scale_factor

        pixel, target, valid_mask = _pad_pixel_training_tensors(
            pixel,
            target,
            valid_mask,
            self.patch_size,
        )
        token = _pad_to_min_shape(token, (emb_patch_size, emb_patch_size), mode="reflect")

        _, h_pix, w_pix = pixel.shape
        _, h_tok, w_tok = token.shape
        max_top_emb = min(h_tok - emb_patch_size, (h_pix - self.patch_size) // self.scale_factor)
        max_left_emb = min(w_tok - emb_patch_size, (w_pix - self.patch_size) // self.scale_factor)
        if max_top_emb < 0 or max_left_emb < 0:
            raise ValueError(
                f"Pixel/token shapes are incompatible for {primary_path} and {token_paths[0]}: "
                f"pixel={pixel.shape[1:]}, token={token.shape[1:]}"
            )

        if self.is_train:
            top_emb = np.random.randint(0, max_top_emb + 1)
            left_emb = np.random.randint(0, max_left_emb + 1)
        else:
            top_emb = max_top_emb // 2
            left_emb = max_left_emb // 2

        top_pix = top_emb * self.scale_factor
        left_pix = left_emb * self.scale_factor

        pixel = _crop_chw(pixel, top_pix, left_pix, self.patch_size)
        token = _crop_chw(token, top_emb, left_emb, emb_patch_size)
        target = _crop_chw(target, top_pix, left_pix, self.patch_size)
        valid_mask = _crop_chw(valid_mask, top_pix, left_pix, self.patch_size)

        # D4 augmentation (training only). Same draw applied to pixel/token/target
        # so spatial correspondence (and 16:1 token-pixel scale) is preserved.
        if self.is_train and self.d4_aug:
            rot_k = np.random.randint(0, 4)
            mirror = bool(np.random.randint(0, 2))
            pixel      = _apply_d4(pixel,      rot_k, mirror)
            token      = _apply_d4(token,      rot_k, mirror)
            target     = _apply_d4(target,     rot_k, mirror)
            valid_mask = _apply_d4(valid_mask, rot_k, mirror)

        return (
            torch.from_numpy(pixel),
            torch.from_numpy(token),
        ), torch.from_numpy(target), torch.from_numpy(valid_mask)


