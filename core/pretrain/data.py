"""Label-free AlphaEarth/Tessera data helpers for pretraining."""

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from core.data.datasets import clean_raster_array
from core.data.discovery import find_multisource_embedding_files
from core.data.discovery import normalize_core_id


def find_pixel_pretrain_pairs(train_alpha_dir, train_tessera_dir,
                              test_alpha_dir=None, test_tessera_dir=None):
    """Return label-free AlphaEarth/Tessera pairs from train and optional test dirs."""
    pairs = []
    for split, alpha_dir, tessera_dir in (
        ("train", train_alpha_dir, train_tessera_dir),
        ("test", test_alpha_dir, test_tessera_dir),
    ):
        if not alpha_dir and not tessera_dir:
            continue
        if not alpha_dir or not tessera_dir:
            raise ValueError(f"{split} pretrain source requires both alpha and tessera dirs")
        for alpha_path, tessera_path in find_multisource_embedding_files(alpha_dir, tessera_dir):
            pairs.append({
                "alpha": alpha_path,
                "tessera": tessera_path,
                "split": split,
                "core_id": normalize_core_id(alpha_path),
            })
    return pairs


class PixelFusionPretrainDataset(Dataset):
    """Label-free AlphaEarth/Tessera dataset for self-supervised pretraining."""

    def __init__(self, pairs, patch_size=256, is_train=True):
        self.pairs = list(pairs)
        self.patch_size = int(patch_size)
        self.is_train = bool(is_train)

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _read(path):
        with rasterio.open(path) as src:
            return clean_raster_array(src.read())

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        alpha = self._read(rec["alpha"])
        tessera = self._read(rec["tessera"])

        if alpha.shape[1:] != tessera.shape[1:]:
            raise ValueError(
                f"Shape mismatch for {rec['core_id']}: "
                f"alpha={alpha.shape}, tessera={tessera.shape}"
            )

        c, h, w = alpha.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            alpha = np.pad(alpha, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            tessera = np.pad(tessera, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            h, w = alpha.shape[1:]

        if self.is_train:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        alpha = alpha[:, top:top + self.patch_size, left:left + self.patch_size]
        tessera = tessera[:, top:top + self.patch_size, left:left + self.patch_size]
        return torch.from_numpy(alpha), torch.from_numpy(tessera)
