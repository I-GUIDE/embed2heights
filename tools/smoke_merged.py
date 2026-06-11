"""Smoke test: merged recipe — λ grad-scale decouple + ring-weighted presence BCE.

Checks, on random CPU tensors shaped like a real batch:
1. forward outputs are IDENTICAL for scale 1.0 / 0.2 / 0.0 (same weights) —
   the scale must only affect backward;
2. a presence-only loss puts exactly scale * (full-coupling gradient) into the
   pixel backbone: ratio(0.2) == 0.2, ratio(0.0) == 0;
3. ring weighting changes presence_bce (exp1 behavior survives the merge);
4. full composite loss + optimizer step runs end to end with both knobs on;
5. both λ config YAMLs parse with the expected knob values.
"""
import sys

import torch
import torch.nn.functional as F

CONFIGS = {
    0.2: "configs/active/xfusion_095_p3_merged_l02_fold0.yml",
    0.5: "configs/active/xfusion_095_p3_merged_l05_fold0.yml",
}

sys.argv = ["smoke", "--config", CONFIGS[0.2]]
from core.config import parse_args, load_config_defaults  # noqa: E402
from core.losses import ImprovedCompositeLoss  # noqa: E402
from core.models import build_model  # noqa: E402

args = parse_args()
assert args.presence_trunk_grad_scale == 0.2, args.presence_trunk_grad_scale
assert args.presence_detach_trunk is False
assert args.presence_tower_depth == 3
assert args.building_ring_presence_alpha == 2.0
assert args.building_boundary_weight == 0.25
cfg05 = load_config_defaults(CONFIGS[0.5])
assert cfg05["presence_trunk_grad_scale"] == 0.5
print("config parse ok (l02 + l05)")


def make_model(scale):
    torch.manual_seed(0)
    built = build_model(
        args.model_type, (192, 3072), n_classes=4,
        tessera_presence_ch=0, tessera_hidden_ch=96, tessera_hidden_depth=2,
        height_specialist_depth=1, lightunet_base_ch=48,
        height_gate_source="alpha", height_hidden_ch=96, height_trunk_depth=2,
        height_independent_branches=True, lightunet_norm_kind="bn",
        gate_mode="rich", gate_init_bias=4.0,
        presence_head_kind="split_all", presence_head_depth=1,
        presence_branch_ch=48, use_fraction_film=False, use_fraction_aux=True,
        token_calibration=True, token_calibration_source_indices=[0, 1],
        use_boundary_head=True, presence_tower_depth=3,
        presence_trunk_grad_scale=scale,
    )
    model = built[0] if isinstance(built, tuple) else built
    model.eval()  # freeze dropout/BN randomness for forward-identity check
    return model


torch.manual_seed(1)
pixel = torch.randn(2, 192, 128, 128)
token = torch.randn(2, 3072, 8, 8)
target_presence = torch.zeros(2, 3, 128, 128)
target_presence[:, 0, 30:60, 30:60] = 1.0


def backbone_grad(model):
    total = 0.0
    for p in model.alpha_unet.parameters():
        if p.grad is not None:
            total += p.grad.abs().sum().item()
    return total


# 1+2: forward identity & exact gradient ratio
outs, grads = {}, {}
for scale in (1.0, 0.2, 0.0):
    model = make_model(scale)
    aux = model((pixel, token), return_aux=True)
    outs[scale] = aux["out"].detach()
    model.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(aux["presence_logits"], target_presence).backward()
    grads[scale] = backbone_grad(model)

# s=0 takes the pure detach branch -> bitwise identical; s=0.2 recombines
# s*x + (1-s)*x which is identical up to float rounding (~1e-7 rel).
assert torch.equal(outs[1.0], outs[0.0]), "s=0 forward must be bitwise identical"
max_diff = (outs[1.0] - outs[0.2]).abs().max().item()
assert torch.allclose(outs[1.0], outs[0.2], atol=1e-5, rtol=1e-5), \
    f"s=0.2 forward deviates beyond float rounding (max diff {max_diff})"
print(f"s=0.2 forward max diff vs s=1.0: {max_diff:.2e} (float rounding only)")
ratio = grads[0.2] / grads[1.0]
assert abs(ratio - 0.2) < 1e-4, f"grad ratio {ratio} != 0.2"
assert grads[0.0] == 0.0
print(f"forward identical across scales; backbone presence-grad ratio: "
      f"s=0.2 -> {ratio:.6f}, s=0.0 -> {grads[0.0]:.1f}")

# 3+4: ring weighting + full composite loss + step on the s=0.2 model
model = make_model(0.2)
model.train()
targets = torch.zeros(2, 4, 128, 128)
targets[:, 0, 30:60, 30:60] = 0.9
targets[:, 1, 70:110, 10:50] = 0.8
targets[:, 3, 30:60, 30:60] = 8.0 / 100.0
valid = torch.ones(2, 2, 128, 128)


def make_loss(alpha):
    return ImprovedCompositeLoss(
        weight_mae=1.0, weight_presence_tversky=1.0, weight_fraction_mae=0.1,
        weight_height_boost=2.0, aux_weight=1.0, loss_preset="presence_centered",
        build_height_boost=5.0, veg_height_boost=1.5,
        water_empty_topk=512, weight_water_empty_topk=0.03,
        building_boundary_weight=0.25,
        building_ring_presence_alpha=alpha, building_ring_kernel=5,
    )


aux = model((pixel, token), return_aux=True)
loss_ring, comp_ring = make_loss(2.0)(aux, targets, valid)
_, comp_flat = make_loss(0.0)(aux, targets, valid)
diff = (comp_ring["presence_bce"] - comp_flat["presence_bce"]).abs().item()
assert diff > 1e-6, "ring weighting had no effect after merge"
assert comp_ring["building_boundary"].item() > 0
assert torch.isfinite(loss_ring)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_ring.backward()
opt.step()
print(f"ring presence_bce diff={diff:.2e}  total={loss_ring.item():.4f}  "
      f"boundary={comp_ring['building_boundary'].item():.4f}")
print("SMOKE OK")
