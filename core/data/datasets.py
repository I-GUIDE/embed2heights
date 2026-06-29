"""PyTorch datasets for competition raster embeddings."""

import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


HEIGHT_NORM_CONSTANT = 30.0


def _teacher_path_for(tar_path, teacher_dir):
    """Map a label/embedding path to its teacher-prediction .npy path.

    Teacher predictions are named by the normalized core id, e.g.
    ``runs/.../predictions/0000_BE.npy`` for label ``label_0000_BE.tif``.
    Returns None if no usable path can be derived.
    """
    if tar_path is None or teacher_dir is None:
        return None
    # Local import to avoid a module-level cycle; discovery imports nothing here.
    from .discovery import normalize_core_id
    core_id = normalize_core_id(tar_path)
    return os.path.join(teacher_dir, f"{core_id}.npy")


def _load_teacher_channels(teacher_dir, tar_path, target):
    """Return a (4, H, W) teacher tensor aligned to the *uncropped* target.

    Channels 0-2 are class-presence probabilities in [0,1]; channel 3 is height
    in METERS, converted to the model's NORMALIZED output scale (divide by
    HEIGHT_NORM_CONSTANT) so it matches the student output and the real target.
    If the teacher file is missing/unreadable, falls back to the real target so
    the distillation term contributes ~0 for that tile.
    """
    teacher = None
    tpath = _teacher_path_for(tar_path, teacher_dir)
    if tpath is not None and os.path.exists(tpath):
        try:
            arr = np.load(tpath)
            teacher = clean_raster_array(arr)
        except Exception:
            teacher = None
    if teacher is None or teacher.shape != target.shape:
        # Fallback: copy the real target (already normalized) -> KD ~ 0.
        return target.astype(np.float32, copy=True)
    teacher = teacher.astype(np.float32, copy=True)
    teacher[3, :, :] = teacher[3, :, :] / HEIGHT_NORM_CONSTANT
    return teacher

# Active model types all consume pixel-aligned embeddings; xfusion additionally
# receives a token tensor through PixelTokenEmbeddingDataset.
PIXEL_MODEL_TYPES = (
    "ae_only",
    "ae_tessera",
    "ae_tessera_gated",
    "xfusion_crosslevel",
    "lightunet",
    "tessera_iou_fusion",
    "tessera_iou_fusion_gated",
    "tessera_token_crosslevel_s2_decoder64_presence_3way_deep",
)
TOKEN_MODEL_TYPES = ()


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


def _apply_d4_aug(*arrays):
    """Random D4 (dihedral) transform applied consistently to every input.

    Inputs are CHW float arrays sharing (H, W). Combines independent h-flip,
    v-flip, and transpose to cover all 8 lattice symmetries — valid for
    height regression because per-pixel scalar height is rotation/flip
    invariant.
    """
    flags = np.random.randint(0, 2, size=3)
    out = []
    for a in arrays:
        if flags[0]:
            a = a[..., ::-1]
        if flags[1]:
            a = a[..., ::-1, :]
        if flags[2]:
            a = a.swapaxes(-2, -1)
        out.append(np.ascontiguousarray(a))
    return tuple(out)


def pick_dataset_class(model_type, n_channels):
    """Resolve (model_type, n_channels) to the right Dataset subclass.

    `model_type == 'auto'` routes by channel count (the rough pixel vs ViT-token
    discriminator); an explicit model type routes by the PIXEL_MODEL_TYPES list.
    """
    mt = model_type.lower()
    if mt == "auto":
        is_pixel = n_channels < 512
    else:
        is_pixel = mt in PIXEL_MODEL_TYPES
    return PixelEmbeddingDataset if is_pixel else LatentTokenDataset


# --- Classification-mask handling (set once by train.py before DataLoaders fork) ---
# The organizers redact LANDCOVER labels in blocks while leaving the embedding (and
# often the height) intact, in DIFFERENT locations from the height redactions. A
# "cls-hole" = a pixel with NO landcover label but a real tall structure
# (height > thr) — a classification-masked spot, symmetric to ndsm_hole (height
# masked where landcover present). Modes:
#   "off"     : legacy (no special handling)
#   "exclude" : drop cls-holes from the CLASSIFICATION loss (model neither penalized
#               for predicting nor taught to suppress there) — height stays supervised.
#   "impute"  : height-derived FAKE label — set building=1 in cls-holes (tall+no-class)
#               and keep supervising, to teach the model the masked content back.
_CLS_HOLE_MODE = "off"
_CLS_HOLE_H_THR = 2.0


def set_cls_hole_config(mode="off", h_thr=2.0):
    global _CLS_HOLE_MODE, _CLS_HOLE_H_THR
    _CLS_HOLE_MODE = str(mode); _CLS_HOLE_H_THR = float(h_thr)


def _prepare_target(tar_path, image_shape, patch_size=None):
    if tar_path is not None:
        with rasterio.open(tar_path) as src:
            raw_target = clean_raster_array(src.read())
        raw_global = ~np.all(raw_target == 0, axis=0)
        has_landcover = (raw_target[0] > 0) | (raw_target[1] > 0) | (raw_target[2] > 0)
        ndsm_hole = (raw_target[3] == 0) & has_landcover
        height_valid = raw_global & ~ndsm_hole         # height: unchanged (keeps cls-holes)
        cls_valid = raw_global                          # classification mask (ch0 of valid_mask)
        if _CLS_HOLE_MODE != "off":
            cls_hole = (~has_landcover) & (raw_target[3] > _CLS_HOLE_H_THR)
            if _CLS_HOLE_MODE == "exclude":
                cls_valid = raw_global & ~cls_hole      # drop cls-holes from classification loss
            elif _CLS_HOLE_MODE == "impute":
                raw_target[0] = np.where(cls_hole, 1.0, raw_target[0])  # fake building label
        valid_mask = np.stack([cls_valid, height_valid], axis=0).astype(np.float32)
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
    def __init__(self, file_pairs, patch_size=128, is_train=True, geom_aug=False,
                 distill_teacher_dir=None):
        self.patch_size = patch_size
        self.is_train = is_train
        self.geom_aug = bool(geom_aug)
        self.distill_teacher_dir = distill_teacher_dir
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

        if self.distill_teacher_dir is not None:
            teacher = _load_teacher_channels(self.distill_teacher_dir, tar_path, target)
            target = np.concatenate([target, teacher], axis=0)

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

        if self.is_train and self.geom_aug:
            image, target, valid_mask = _apply_d4_aug(image, target, valid_mask)

        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(valid_mask)


class MultiPixelEmbeddingDataset(Dataset):
    """
    For concatenating two pixel-aligned embedding sources, e.g. AlphaEarth
    64ch + Tessera 128ch -> 192ch at 256x256.

    file_pairs may contain:
      - (primary_emb_path, secondary_emb_path, label_path)
      - (primary_emb_path, secondary_emb_path) for label-free inference
    """
    def __init__(self, file_pairs, patch_size=128, is_train=True, geom_aug=False,
                 cutmix_prob=0.0, cutmix_min_frac=0.15, cutmix_max_frac=0.5,
                 cutmix_density_aware=False, mixup_prob=0.0, mixup_alpha=0.2,
                 bld_copypaste_prob=0.0, bld_copypaste_max_size_frac=0.25,
                 distill_teacher_dir=None):
        self.patch_size = patch_size
        self.is_train = is_train
        self.geom_aug = bool(geom_aug)
        self.distill_teacher_dir = distill_teacher_dir
        self.file_pairs = file_pairs
        self.cutmix_prob = float(cutmix_prob)
        self.cutmix_min_frac = float(cutmix_min_frac)
        self.cutmix_max_frac = float(cutmix_max_frac)
        self.cutmix_density_aware = bool(cutmix_density_aware)
        self.mixup_prob = float(mixup_prob)
        self.mixup_alpha = float(mixup_alpha)
        # Building copy-paste: locate a building region in a donor tile and
        # paste it (with corresponding embedding features) into the current tile.
        # Increases building diversity in low-density tiles. Targets iou_bld.
        self.bld_copypaste_prob = float(bld_copypaste_prob)
        self.bld_copypaste_max_size_frac = float(bld_copypaste_max_size_frac)

    def __len__(self):
        return len(self.file_pairs)

    def _load_tile(self, idx):
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
        if self.distill_teacher_dir is not None:
            teacher = _load_teacher_channels(self.distill_teacher_dir, tar_path, target)
            target = np.concatenate([target, teacher], axis=0)
        image, target, valid_mask = _pad_pixel_training_tensors(
            image, target, valid_mask, self.patch_size,
        )
        _, h, w = image.shape
        top, left = _sample_or_center_origin(h, w, self.patch_size, self.is_train)
        image = _crop_chw(image, top, left, self.patch_size)
        target = _crop_chw(target, top, left, self.patch_size)
        valid_mask = _crop_chw(valid_mask, top, left, self.patch_size)
        return image, target, valid_mask

    def __getitem__(self, idx):
        image, target, valid_mask = self._load_tile(idx)

        # CutMix: with probability cutmix_prob, paste a random box from another
        # tile into this one. Useful for KE-density augmentation since pasting
        # high-density patches creates synthetic urban-dense composites.
        if self.is_train and self.cutmix_prob > 0 and np.random.rand() < self.cutmix_prob:
            # Pick source tile. If density_aware, sample from indices with above-average building label.
            if self.cutmix_density_aware:
                # Try a few candidates and pick the one with highest bld density
                best_idx = None; best_density = -1.0
                for _ in range(3):
                    cand = np.random.randint(len(self.file_pairs))
                    if cand == idx: continue
                    _, t_cand, _ = self._load_tile(cand)
                    d = float((t_cand[0] > 0.5).mean())
                    if d > best_density: best_density = d; best_idx = cand
                src_idx = best_idx if best_idx is not None else np.random.randint(len(self.file_pairs))
            else:
                src_idx = np.random.randint(len(self.file_pairs))
                while src_idx == idx:
                    src_idx = np.random.randint(len(self.file_pairs))
            src_image, src_target, src_valid_mask = self._load_tile(src_idx)
            # Random cut box
            ph = image.shape[1]; pw = image.shape[2]
            frac = np.random.uniform(self.cutmix_min_frac, self.cutmix_max_frac)
            bh = max(1, int(round(np.sqrt(frac) * ph)))
            bw = max(1, int(round(np.sqrt(frac) * pw)))
            y0 = np.random.randint(0, max(1, ph - bh + 1))
            x0 = np.random.randint(0, max(1, pw - bw + 1))
            image[:, y0:y0+bh, x0:x0+bw] = src_image[:, y0:y0+bh, x0:x0+bw]
            target[:, y0:y0+bh, x0:x0+bw] = src_target[:, y0:y0+bh, x0:x0+bw]
            valid_mask[:, y0:y0+bh, x0:x0+bw] = src_valid_mask[:, y0:y0+bh, x0:x0+bw]

        # Building Copy-Paste: find a building region in a donor tile, crop a
        # bounding box around real building pixels, paste it (with corresponding
        # embedding features) into the current tile at a random position. This
        # targets iou_bld by giving the model real buildings in diverse
        # geographic contexts.
        if self.is_train and self.bld_copypaste_prob > 0 and np.random.rand() < self.bld_copypaste_prob:
            # Try up to 5 donor candidates; accept the first with enough building pixels.
            donor_image = None
            donor_target = None
            donor_valid = None
            donor_bbox = None
            for _ in range(5):
                cand = np.random.randint(len(self.file_pairs))
                if cand == idx:
                    continue
                d_image, d_target, d_valid = self._load_tile(cand)
                bld_pix = (d_target[0] > 0.1)  # channel 0 = building presence
                if bld_pix.sum() < 32:  # too few buildings, skip
                    continue
                # Find bounding box of building pixels
                ys, xs = np.where(bld_pix)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                # Limit box size to <= max_size_frac of patch
                ph_p, pw_p = d_image.shape[1], d_image.shape[2]
                max_h = int(ph_p * self.bld_copypaste_max_size_frac)
                max_w = int(pw_p * self.bld_copypaste_max_size_frac)
                bh = min(y1 - y0, max_h)
                bw = min(x1 - x0, max_w)
                # Random crop within the bounding box if larger than max
                if (y1 - y0) > bh:
                    y0 = y0 + np.random.randint(0, (y1 - y0) - bh + 1)
                if (x1 - x0) > bw:
                    x0 = x0 + np.random.randint(0, (x1 - x0) - bw + 1)
                donor_image = d_image
                donor_target = d_target
                donor_valid = d_valid
                donor_bbox = (y0, y0 + bh, x0, x0 + bw)
                break
            if donor_bbox is not None:
                dy0, dy1, dx0, dx1 = donor_bbox
                bh = dy1 - dy0
                bw = dx1 - dx0
                ph_c, pw_c = image.shape[1], image.shape[2]
                # Random paste location in current tile
                py = np.random.randint(0, max(1, ph_c - bh + 1))
                px = np.random.randint(0, max(1, pw_c - bw + 1))
                # Paste both inputs and targets so input-output consistency holds
                image[:, py:py+bh, px:px+bw] = donor_image[:, dy0:dy1, dx0:dx1]
                target[:, py:py+bh, px:px+bw] = donor_target[:, dy0:dy1, dx0:dx1]
                valid_mask[:, py:py+bh, px:px+bw] = donor_valid[:, dy0:dy1, dx0:dx1]

        # Mixup: linear blend with another tile. No discontinuities (unlike CutMix).
        if self.is_train and self.mixup_prob > 0 and np.random.rand() < self.mixup_prob:
            src_idx = np.random.randint(len(self.file_pairs))
            while src_idx == idx:
                src_idx = np.random.randint(len(self.file_pairs))
            src_image, src_target, src_valid_mask = self._load_tile(src_idx)
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5 so the primary tile dominates
            image = (lam * image + (1.0 - lam) * src_image).astype(np.float32)
            target = (lam * target + (1.0 - lam) * src_target).astype(np.float32)
            valid_mask = np.minimum(valid_mask, src_valid_mask)

        if self.is_train and self.geom_aug:
            image, target, valid_mask = _apply_d4_aug(image, target, valid_mask)

        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(valid_mask)


class MultiLatentTokenDataset(Dataset):
    """
    For same-grid token fusion, e.g. TerraMind/THOR S1 768ch@16x16 +
    S2 768ch@16x16 -> 1536ch@16x16.

    file_pairs may contain:
      - (primary_token_path, secondary_token_path, label_path)
      - (primary_token_path, secondary_token_path) for label-free inference
    """
    def __init__(self, file_pairs, patch_size=256, scale_factor=16, is_train=True):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
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
            raise ValueError("MultiLatentTokenDataset expects 2- or 3-item tuples")

        primary = _read_raster(primary_path)
        secondary = _read_raster(secondary_path)

        _assert_same_spatial(primary, secondary, primary_path, secondary_path, "Token")
        image = np.concatenate([primary, secondary], axis=0)
        emb_patch_size = self.patch_size // self.scale_factor

        target, valid_mask = _prepare_target(tar_path, image.shape[1:], patch_size=self.patch_size)

        image = _pad_to_min_shape(image, (emb_patch_size, emb_patch_size), mode="reflect")
        target = _pad_to_min_shape(target, (self.patch_size, self.patch_size), mode="reflect")
        valid_mask = _pad_to_min_shape(
            valid_mask,
            (self.patch_size, self.patch_size),
            mode="constant",
            constant_values=0,
        )

        _, h_emb, w_emb = image.shape
        top_emb, left_emb = _sample_or_center_origin(
            h_emb,
            w_emb,
            emb_patch_size,
            self.is_train,
        )

        top_tar = top_emb * self.scale_factor
        left_tar = left_emb * self.scale_factor

        image = _crop_chw(image, top_emb, left_emb, emb_patch_size)
        target = _crop_chw(target, top_tar, left_tar, self.patch_size)
        valid_mask = _crop_chw(valid_mask, top_tar, left_tar, self.patch_size)

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
                 distill_teacher_dir=None):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.distill_teacher_dir = distill_teacher_dir
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

        if self.distill_teacher_dir is not None:
            teacher = _load_teacher_channels(self.distill_teacher_dir, tar_path, target)
            target = np.concatenate([target, teacher], axis=0)

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


class LatentTokenDataset(Dataset):
    """
    For patch-level embeddings (TerraMind 768ch@16x16, THOR 768ch@16x16).
    file_pairs: list of (emb_path, label_path) tuples, OR list of emb_path strings (label-free mode).
    """
    def __init__(self, file_pairs, patch_size=256, scale_factor=16, is_train=True):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
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

        emb_patch_size = self.patch_size // self.scale_factor
        target, valid_mask = _prepare_target(tar_path, image.shape[1:], patch_size=self.patch_size)

        image = _pad_to_min_shape(image, (emb_patch_size, emb_patch_size), mode="reflect")
        target = _pad_to_min_shape(target, (self.patch_size, self.patch_size), mode="reflect")
        valid_mask = _pad_to_min_shape(
            valid_mask,
            (self.patch_size, self.patch_size),
            mode="constant",
            constant_values=0,
        )

        _, h_emb, w_emb = image.shape
        top_emb, left_emb = _sample_or_center_origin(
            h_emb,
            w_emb,
            emb_patch_size,
            self.is_train,
        )

        top_tar = top_emb * self.scale_factor
        left_tar = left_emb * self.scale_factor

        image = _crop_chw(image, top_emb, left_emb, emb_patch_size)
        target = _crop_chw(target, top_tar, left_tar, self.patch_size)
        valid_mask = _crop_chw(valid_mask, top_tar, left_tar, self.patch_size)

        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(valid_mask)
