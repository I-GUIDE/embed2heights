"""Pretraining-only masked cross-reconstruction model."""

import torch
import torch.nn as nn

from core.models import ConvGNAct, LightUNet, TesseraCompressionStem


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
