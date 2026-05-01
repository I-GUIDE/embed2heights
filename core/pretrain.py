import os
import json

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.dataset import (
    clean_raster_array,
    find_multisource_embedding_files,
    normalize_core_id,
)
from core.model import ConvGNAct, LightUNet, TesseraCompressionStem


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


class BlockMask2d(nn.Module):
    """Generate spatial block masks and fill masked pixels by channel mean.

    ``forward(x, mask=None)`` accepts an externally-provided mask so callers
    can force shared / complementary masking across modalities (the legacy
    behaviour with ``mask=None`` samples a fresh independent mask).
    """

    def __init__(self, mask_ratio=0.55, block_size=16):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.block_size = int(block_size)

    def sample_mask(self, b, h, w, device):
        gh = max(1, (h + self.block_size - 1) // self.block_size)
        gw = max(1, (w + self.block_size - 1) // self.block_size)
        low = torch.rand(b, 1, gh, gw, device=device) < self.mask_ratio
        return F.interpolate(low.float(), size=(h, w), mode="nearest")

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        if mask is None:
            mask = self.sample_mask(b, h, w, x.device)
        keep = 1.0 - mask
        denom = keep.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        fill = (x * keep).sum(dim=(-2, -1), keepdim=True) / denom
        x_masked = x * keep + fill * mask
        return x_masked, mask


def apply_mask_strategy(masker, alpha, tessera, strategy, *,
                        modality_dropout=0.0, generator=None):
    """Build masked-input pairs and per-modality loss weights.

    Returns ``(alpha_in, alpha_mask, tessera_in, tessera_mask, alpha_w, tessera_w)``.

    Strategies:
      - ``complementary``: per batch, flip a coin. One side gets a fresh mask
        and is reconstructed; the other is left fully visible (zero mask, zero
        loss weight). Maximises cross-modal signal — when alpha is masked, the
        only way to recover it is via the visible tessera at the same pixel.
      - ``dual``: a single shared mask drives both modalities. Pure spatial
        inference.
      - ``independent``: legacy behaviour — fresh independent masks.
      - ``mixed``: per batch, uniformly pick among
        {complementary-alpha, complementary-tessera, dual}.

    ``modality_dropout`` (in [0, 1]): with this probability, the visible
    side in a complementary batch is also zeroed at the input — forces the
    masked-side reconstruction to fall back to the spatial context only.
    The corresponding loss weight is unchanged. Keeps a simple regulariser
    against finetune-time modality dropout.
    """
    if generator is None:
        coin = torch.rand((), device=alpha.device).item()
    else:
        coin = torch.rand((), generator=generator, device=alpha.device).item()

    strategy = (strategy or "complementary").lower()
    if strategy == "mixed":
        if coin < 1.0 / 3.0:
            strategy = "complementary"; force_alpha_mask = True
        elif coin < 2.0 / 3.0:
            strategy = "complementary"; force_alpha_mask = False
        else:
            strategy = "dual"; force_alpha_mask = None
    else:
        force_alpha_mask = coin < 0.5  # used only by complementary

    b, _, h, w = alpha.shape
    if strategy == "complementary":
        m = masker.sample_mask(b, h, w, alpha.device)
        if force_alpha_mask:
            alpha_in, alpha_mask = masker(alpha, mask=m)
            tessera_in = tessera
            tessera_mask = torch.zeros_like(m)
            alpha_w, tessera_w = 1.0, 0.0
        else:
            tessera_in, tessera_mask = masker(tessera, mask=m)
            alpha_in = alpha
            alpha_mask = torch.zeros_like(m)
            alpha_w, tessera_w = 0.0, 1.0
        if modality_dropout > 0:
            drop = torch.rand((), device=alpha.device).item() < modality_dropout
            if drop:
                if force_alpha_mask:
                    tessera_in = torch.zeros_like(tessera_in)
                else:
                    alpha_in = torch.zeros_like(alpha_in)
        return alpha_in, alpha_mask, tessera_in, tessera_mask, alpha_w, tessera_w

    if strategy == "dual":
        m = masker.sample_mask(b, h, w, alpha.device)
        alpha_in, alpha_mask = masker(alpha, mask=m)
        tessera_in, tessera_mask = masker(tessera, mask=m)
        return alpha_in, alpha_mask, tessera_in, tessera_mask, 1.0, 1.0

    # independent (legacy)
    alpha_in, alpha_mask = masker(alpha)
    tessera_in, tessera_mask = masker(tessera)
    return alpha_in, alpha_mask, tessera_in, tessera_mask, 1.0, 1.0


class PixelFusionPretrainModel(nn.Module):
    """Masked cross-reconstruction model with transferable Alpha/Tessera modules.

    The module names ``alpha_unet`` and ``tessera_stem`` intentionally match
    ``TesseraIoUFusionLightUNet`` so supervised training can load them directly.
    Reconstruction heads are pretraining-only and are ignored by train.py.
    """

    def __init__(self, alpha_channels=64, tessera_channels=128, base_ch=48,
                 tessera_presence_ch=16, tessera_hidden_ch=96,
                 tessera_hidden_depth=2, fusion_ch=None, drop=0.05,
                 norm_kind="gn"):
        super().__init__()
        fusion_ch = int(fusion_ch or max(base_ch, tessera_presence_ch * 2))
        self.alpha_channels = int(alpha_channels)
        self.tessera_channels = int(tessera_channels)
        self.norm_kind = norm_kind

        self.alpha_unet = LightUNet(alpha_channels, 4, base_ch=base_ch, norm_kind=norm_kind)
        self.alpha_unet.head = nn.Identity()
        self.tessera_stem = TesseraCompressionStem(
            tessera_channels,
            out_ch=tessera_presence_ch,
            hidden_ch=tessera_hidden_ch,
            hidden_depth=tessera_hidden_depth,
        )
        self.fusion = nn.Sequential(
            ConvGNAct(base_ch + tessera_presence_ch, fusion_ch, kernel_size=1, padding=0),
            nn.Dropout2d(drop),
            ConvGNAct(fusion_ch, fusion_ch, kernel_size=3),
            ConvGNAct(fusion_ch, fusion_ch, kernel_size=3),
        )
        self.alpha_recon = nn.Conv2d(fusion_ch, alpha_channels, kernel_size=1)
        self.tessera_recon = nn.Conv2d(fusion_ch, tessera_channels, kernel_size=1)

    def forward(self, alpha, tessera):
        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_stem(tessera)
        fused = self.fusion(torch.cat([alpha_feat, tessera_feat], dim=1))
        return {
            "alpha": self.alpha_recon(fused),
            "tessera": self.tessera_recon(fused),
        }


def channel_standardize(x, eps=1e-5):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    var = (x - mean).pow(2).mean(dim=(-2, -1), keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps)


def masked_reconstruction_loss(pred, target, mask, cosine_weight=0.05):
    """MSE on masked pixels, with a small per-pixel channel-cosine term."""
    target = channel_standardize(target)
    mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp_min(1.0)
    mse = ((pred - target).pow(2) * mask).sum() / denom

    cosine = pred.new_tensor(0.0)
    if cosine_weight > 0:
        pred_pix = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
        tgt_pix = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
        mask_pix = mask[:, :1].permute(0, 2, 3, 1).reshape(-1) > 0.5
        if mask_pix.any():
            cosine = (1.0 - F.cosine_similarity(
                pred_pix[mask_pix], tgt_pix[mask_pix], dim=1, eps=1e-6
            )).mean()
    return mse + cosine_weight * cosine, {"mse": mse.detach(), "cosine": cosine.detach()}


def save_pretrain_config(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
