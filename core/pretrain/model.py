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


class GatedTokenPretrainModel(nn.Module):
    """6-source masked/denoise reconstruction whose submodules match
    ``TesseraIoUFusionGatedLightUNet`` EXACTLY by name + shape, so supervised
    ``--init-from-pretrain`` transfers them cleanly (fixes the old
    ``tessera_stem`` name/width mismatch).

    Encoder mirrors the gated forward: alpha_unet(forward_features) +
    tessera_feature_stem -> per-pixel fusion gate -> xsource_fusion(tokens).
    The dense inputs (alpha, tessera) are corrupted by the pretext; the coarse
    tokens are fed clean, so the cross-source fusion learns token->dense
    modulation. We reconstruct the clean dense embeddings (this drives gradients
    through every transferable module, including xsource_fusion). The recon heads
    are pretraining-only and ignored by train.py.
    """

    def __init__(self, alpha_channels=64, tessera_channels=128,
                 token_channels=3072, token_source_ch=768, token_ctx_ch=96,
                 base_ch=48, tessera_hidden_ch=128, tessera_hidden_depth=2,
                 encoder_arch="unetpp_wave", norm_kind="bn",
                 use_bottleneck_attn=True, gate_mode="simple", gate_init_bias=4.0,
                 xsource_attn_heads=4, xsource_token_calibration=True, drop=0.05):
        super().__init__()
        # Import here to mirror the exact supervised submodules (and avoid any
        # import-order surprises at module load).
        from core.models import LightUNet, TesseraCompressionStem
        from core.models.pixel_fusion import (
            CrossSourceHybridFiLMFusion, _build_fusion_gate,
        )
        self.alpha_channels = int(alpha_channels)
        self.tessera_channels = int(tessera_channels)
        self.token_channels = int(token_channels)
        self.gate_mode = gate_mode

        self.alpha_unet = LightUNet(
            alpha_channels, 4, base_ch=base_ch, norm_kind=norm_kind,
            use_bottleneck_attn=bool(use_bottleneck_attn),
            encoder_arch=str(encoder_arch),
        )
        self.alpha_unet.head = nn.Identity()
        self.tessera_feature_stem = TesseraCompressionStem(
            tessera_channels, out_ch=base_ch,
            hidden_ch=tessera_hidden_ch, hidden_depth=tessera_hidden_depth,
        )
        self.gate_conv = _build_fusion_gate(
            base_ch, mode=gate_mode, untied=False, init_bias=gate_init_bias,
        )
        self.xsource_fusion = CrossSourceHybridFiLMFusion(
            pixel_ch=base_ch, token_channels=token_channels,
            token_source_ch=int(token_source_ch), ctx_ch=int(token_ctx_ch),
            token_calibration=bool(xsource_token_calibration),
            attn_heads=int(xsource_attn_heads),
        )
        # pretraining-only reconstruction heads (discarded at fine-tune)
        self.alpha_recon = nn.Conv2d(base_ch, alpha_channels, kernel_size=1)
        self.tessera_recon = nn.Conv2d(base_ch, tessera_channels, kernel_size=1)

    def forward(self, alpha, tessera, token):
        from core.models.pixel_fusion import _apply_fusion_gate
        alpha_feat = self.alpha_unet.forward_features(alpha)
        tessera_feat = self.tessera_feature_stem(tessera)
        fused = _apply_fusion_gate(
            self.gate_conv, alpha_feat, tessera_feat,
            untied=False, mode=self.gate_mode,
        )
        fused = self.xsource_fusion(fused, token)
        return {
            "alpha": self.alpha_recon(fused),
            "tessera": self.tessera_recon(fused),
        }
