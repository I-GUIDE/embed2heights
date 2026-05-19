"""PyTorch datasets for competition raster embeddings."""

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


def _sample_or_center_origin(height, width, crop_size, is_train):
    if is_train:
        return (
            np.random.randint(0, height - crop_size + 1),
            np.random.randint(0, width - crop_size + 1),
        )
    return (height - crop_size) // 2, (width - crop_size) // 2


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
    def __init__(self, file_pairs, patch_size=128, scale_factor=16, is_train=True):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.file_pairs = file_pairs

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

        return (
            torch.from_numpy(pixel),
            torch.from_numpy(token),
        ), torch.from_numpy(target), torch.from_numpy(valid_mask)


class PixelMultiTokenEmbeddingDataset(Dataset):
    """
    For probing same-model S1/S2 token fusion against the AlphaEarth+Tessera
    champion.

    file_pairs may contain:
      - (primary_emb_path, secondary_emb_path, token_primary_path,
         token_secondary_path, label_path)
      - (primary_emb_path, secondary_emb_path, token_primary_path,
         token_secondary_path) for label-free inference

    Returns ((pixel_image, token_image), target, valid_mask), where pixel_image
    is AlphaEarth+Tessera concatenated at 256x256 and token_image is
    [S1, S2] channel-concatenated at 16x16.
    """
    def __init__(self, file_pairs, patch_size=128, scale_factor=16, is_train=True):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pair = self.file_pairs[idx]
        if len(pair) == 5:
            primary_path, secondary_path, token_primary_path, token_secondary_path, tar_path = pair
        elif len(pair) == 4:
            primary_path, secondary_path, token_primary_path, token_secondary_path = pair
            tar_path = None
        else:
            raise ValueError("PixelMultiTokenEmbeddingDataset expects 4- or 5-item tuples")

        primary = _read_raster(primary_path)
        secondary = _read_raster(secondary_path)
        token_primary = _read_raster(token_primary_path)
        token_secondary = _read_raster(token_secondary_path)

        _assert_same_spatial(primary, secondary, primary_path, secondary_path, "Embedding")
        _assert_same_spatial(
            token_primary,
            token_secondary,
            token_primary_path,
            token_secondary_path,
            "Token",
        )
        pixel = np.concatenate([primary, secondary], axis=0)
        token = np.concatenate([token_primary, token_secondary], axis=0)
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
                f"Pixel/token shapes are incompatible for {primary_path} and {token_primary_path}: "
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

        return (
            torch.from_numpy(pixel),
            torch.from_numpy(token),
        ), torch.from_numpy(target), torch.from_numpy(valid_mask)


