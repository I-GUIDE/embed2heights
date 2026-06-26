"""Smoke test for the wavelet pixel-backbone downsampling (pixel_backbone_kind=wavelet).

Run: PYTHONPATH=. python tools/smoke_wavelet.py
Verifies, on random CPU tensors:
  A. WaveletDownsample halves H,W, is channel-preserving, and warm-starts as the
     LL (average) subband (so it is a gentle drop-in for MaxPool2d at init);
  B. LightUNet(down_kind='wavelet') produces the same feature shape as the
     maxpool variant, and _build_pixel_backbone('wavelet') wires it;
  C. the full hybrid model builds with pixel_backbone_kind='wavelet' and runs a
     forward+backward, with the wavelet projection receiving gradient.
"""
import torch

from core.models import build_model
from core.models.backbones import LightUNet, WaveletDownsample
from core.models.token_fusion import _build_pixel_backbone


# ---- A. WaveletDownsample: shape + warm-start == LL subband ----
wd = WaveletDownsample(5)
x = torch.randn(2, 5, 32, 32)
y = wd(x)
assert y.shape == (2, 5, 16, 16), y.shape
ll_manual = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2]
             + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) * 0.5
assert torch.allclose(y, ll_manual, atol=1e-5), (y - ll_manual).abs().max()
print("A. WaveletDownsample: shape ok, warm-start == LL subband")

# ---- B. LightUNet(down_kind=wavelet) feature shape == maxpool variant ----
lu_mp = LightUNet(64, 4, base_ch=48, down_kind="maxpool")
lu_wv = LightUNet(64, 4, base_ch=48, down_kind="wavelet")
xb = torch.randn(2, 64, 128, 128)
fmp = lu_mp.forward_features(xb)
fwv = lu_wv.forward_features(xb)
assert fmp.shape == fwv.shape == (2, 48, 128, 128), (fmp.shape, fwv.shape)
assert _build_pixel_backbone("wavelet", 64, 4, 48, "bn").down_kind == "wavelet"
print(f"B. LightUNet wavelet feat shape {tuple(fwv.shape)} == maxpool; _build_pixel_backbone ok")

# ---- C. full hybrid model builds with wavelet + forward/backward ----
torch.manual_seed(0)
model = build_model(
    "xfusion_unet_hybrid_cross_source", (192, 3072), n_classes=4,
    tessera_presence_ch=0, tessera_hidden_ch=96, tessera_hidden_depth=2,
    height_specialist_depth=1, lightunet_base_ch=48,
    height_gate_source="alpha", height_hidden_ch=96, height_trunk_depth=2,
    height_independent_branches=True, lightunet_norm_kind="bn",
    gate_mode="rich", gate_init_bias=4.0,
    presence_head_kind="split_all", presence_head_depth=1,
    presence_branch_ch=48, use_fraction_film=False, use_fraction_aux=True,
    token_calibration=True, token_calibration_source_indices=[0, 1],
    use_boundary_head=True, presence_tower_depth=2,
    split_trunk=True, presence_trunk_grad_scale=1.0,
    pixel_backbone_kind="wavelet",
)
model = model[0] if isinstance(model, tuple) else model
model.train()
aux = model((torch.randn(2, 192, 128, 128), torch.randn(2, 3072, 8, 8)), return_aux=True)
assert aux["out"].shape == (2, 4, 128, 128)
aux["out"].abs().mean().backward()
gwave = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
            if "proj" in n and p.grad is not None)
assert gwave > 0.0, "wavelet projection received no gradient"
print(f"C. wavelet hybrid forward+backward ok; wavelet-proj grad={gwave:.2e}")

print("SMOKE OK")
