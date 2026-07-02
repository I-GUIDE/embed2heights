import torch
import torch.nn as nn

from .backbones import _group_count


def _build_fusion_gate(channels, mode="simple", untied=False, init_bias=4.0):
    n_out = 2 * channels if untied else channels
    if mode == "simple":
        gate = nn.Conv2d(2 * channels, n_out, kernel_size=1)
        final = gate
    elif mode == "rich":
        hidden = max(channels, 32)
        gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, n_out, 1),
        )
        final = gate[-1]
    else:
        raise ValueError(f"Unknown gate mode: {mode!r}")

    nn.init.zeros_(final.weight)
    if untied:
        bias = torch.empty(n_out)
        bias[:channels].fill_(init_bias)
        bias[channels:].fill_(-init_bias)
        final.bias.data.copy_(bias)
    else:
        nn.init.constant_(final.bias, init_bias)
    return gate


def _apply_fusion_gate(gate_module, ae_feat, tes_feat, untied):
    raw = gate_module(torch.cat([ae_feat, tes_feat], dim=1))
    if untied:
        channels = ae_feat.size(1)
        g_ae = torch.sigmoid(raw[:, :channels])
        g_tes = torch.sigmoid(raw[:, channels:])
        return g_ae * ae_feat + g_tes * tes_feat
    gate = torch.sigmoid(raw)
    return gate * ae_feat + (1.0 - gate) * tes_feat


def _maybe_drop_modality(tes_feat, p, training):
    if not training or p <= 0.0:
        return tes_feat
    batch = tes_feat.size(0)
    keep = (torch.rand(batch, 1, 1, 1, device=tes_feat.device) >= p).float()
    return tes_feat * keep / max(1e-6, 1.0 - p)
