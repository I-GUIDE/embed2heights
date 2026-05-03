"""Mask sampling and masking strategies for self-supervised pretraining."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        loss weight). Maximises cross-modal signal -- when alpha is masked, the
        only way to recover it is via the visible tessera at the same pixel.
      - ``dual``: a single shared mask drives both modalities. Pure spatial
        inference.
      - ``independent``: legacy behaviour -- fresh independent masks.
      - ``mixed``: per batch, uniformly pick among
        {complementary-alpha, complementary-tessera, dual}.

    ``modality_dropout`` (in [0, 1]): with this probability, the visible
    side in a complementary batch is also zeroed at the input -- forces the
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
        force_alpha_mask = coin < 0.5

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

    alpha_in, alpha_mask = masker(alpha)
    tessera_in, tessera_mask = masker(tessera)
    return alpha_in, alpha_mask, tessera_in, tessera_mask, 1.0, 1.0
