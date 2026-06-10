"""Smoke test: ring-weighted building presence BCE (exp/ring-weighted-presence).

Checks, on random CPU tensors shaped like a real batch:
1. model forward + composite loss + backward + optimizer step run end to end
   with building_boundary_weight=0.25 and building_ring_presence_alpha=2.0;
2. the ring weighting actually changes presence_bce vs alpha=0;
3. config YAML parses and exposes the new knobs.
"""
import sys

import torch

sys.argv = [
    "smoke", "--config",
    "configs/active/xfusion_095_p3_ringw_fold0.yml",
]
from core.config import parse_args  # noqa: E402
from core.losses import ImprovedCompositeLoss  # noqa: E402
from core.models import build_model  # noqa: E402

args = parse_args()
assert args.building_ring_presence_alpha == 2.0, args.building_ring_presence_alpha
assert args.building_ring_kernel == 5
assert args.building_boundary_weight == 0.25
assert args.presence_tower_depth == 2
print("config parse ok")

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
    use_boundary_head=True, presence_tower_depth=2,
)
model = built[0] if isinstance(built, tuple) else built
model.train()

pixel = torch.randn(2, 192, 128, 128)
token = torch.randn(2, 3072, 8, 8)
targets = torch.zeros(2, 4, 128, 128)
targets[:, 0, 30:60, 30:60] = 0.9   # building blob
targets[:, 1, 70:110, 10:50] = 0.8  # veg blob
targets[:, 3, 30:60, 30:60] = 8.0 / 100.0
valid = torch.ones(2, 2, 128, 128)

aux = model((pixel, token), return_aux=True)
assert "building_boundary_logits" in aux, "boundary head missing"

def make_loss(alpha):
    return ImprovedCompositeLoss(
        weight_mae=1.0, weight_presence_tversky=1.0, weight_fraction_mae=0.1,
        weight_height_boost=2.0, aux_weight=1.0, loss_preset="presence_centered",
        build_height_boost=5.0, veg_height_boost=1.5,
        water_empty_topk=512, weight_water_empty_topk=0.03,
        building_boundary_weight=0.25,
        building_ring_presence_alpha=alpha, building_ring_kernel=5,
    )

loss_ring, comp_ring = make_loss(2.0)(aux, targets, valid)
loss_flat, comp_flat = make_loss(0.0)(aux, targets, valid)
assert torch.isfinite(loss_ring), loss_ring
diff = (comp_ring["presence_bce"] - comp_flat["presence_bce"]).abs().item()
assert diff > 1e-6, f"ring weighting had no effect on presence_bce (diff={diff})"
assert comp_ring["building_boundary"].item() > 0
print(f"presence_bce ring={comp_ring['presence_bce']:.5f} "
      f"flat={comp_flat['presence_bce']:.5f} (diff {diff:.2e})")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_ring.backward()
opt.step()
print(f"total={loss_ring.item():.4f}  boundary={comp_ring['building_boundary'].item():.4f}")
print("SMOKE OK")
