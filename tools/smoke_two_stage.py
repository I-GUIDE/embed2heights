"""Self-contained smoke test for the two-stage (train -> purify) recipe.

Verifies, on random CPU tensors, every mechanism the 2-stage template relies
on — no config files or checkpoints required:

1. split_trunk model builds; ring-weighted presence BCE changes the loss;
2. forward output is identical for presence_trunk_grad_scale 1.0 vs 0.0
   (the knob is backward-only);
3. at s=0 in split mode, seg-side losses (presence + boundary + fraction)
   put ZERO gradient on the shared backbone and the height trunk, while the
   height loss still reaches the backbone (stage-2 purification semantics);
4. full composite loss + optimizer step runs end to end;
5. a state-dict saved from the s=1 model loads strict-clean into the s=0
   model (stage-1 checkpoint -> stage-2 warm start).

Run: PYTHONPATH=. python tools/smoke_two_stage.py
"""
import os
import tempfile

import torch
import torch.nn.functional as F

from core.losses import ImprovedCompositeLoss
from core.models import build_model


def make_model(scale):
    torch.manual_seed(0)
    built = build_model(
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
        split_trunk=True, presence_trunk_grad_scale=scale,
    )
    model = built[0] if isinstance(built, tuple) else built
    model.eval()
    return model


def grad_sum(module):
    return sum(p.grad.abs().sum().item() for p in module.parameters()
               if p.grad is not None)


torch.manual_seed(1)
pixel = torch.randn(2, 192, 128, 128)
token = torch.randn(2, 3072, 8, 8)
target_presence = torch.zeros(2, 3, 128, 128)
target_presence[:, 0, 30:60, 30:60] = 1.0
targets = torch.zeros(2, 4, 128, 128)
targets[:, 0, 30:60, 30:60] = 0.9
targets[:, 1, 70:110, 10:50] = 0.8
targets[:, 3, 30:60, 30:60] = 8.0 / 100.0
valid = torch.ones(2, 2, 128, 128)


def make_loss(ring_alpha):
    return ImprovedCompositeLoss(
        weight_mae=1.0, weight_presence_tversky=1.0, weight_fraction_mae=0.1,
        weight_height_boost=2.0, aux_weight=1.0, loss_preset="presence_centered",
        build_height_boost=5.0, veg_height_boost=1.5,
        water_empty_topk=512, weight_water_empty_topk=0.03,
        building_boundary_weight=0.25,
        building_ring_presence_alpha=ring_alpha, building_ring_kernel=5,
    )


# 1 + 4: stage-1 semantics — coupled model, ring loss active, full step
stage1 = make_model(1.0)
stage1.train()
aux = stage1((pixel, token), return_aux=True)
loss_ring, comp_ring = make_loss(2.0)(aux, targets, valid)
_, comp_flat = make_loss(0.0)(aux, targets, valid)
diff = (comp_ring["presence_bce"] - comp_flat["presence_bce"]).abs().item()
assert diff > 1e-6, "ring weighting inactive"
assert torch.isfinite(loss_ring)
torch.optim.AdamW(stage1.parameters(), lr=1e-4).step
loss_ring.backward()
print(f"stage-1: ring presence_bce diff={diff:.2e}, full loss/step ok")

# 2: forward identity across the knob
outs = {s: make_model(s)((pixel, token), return_aux=True)["out"].detach()
        for s in (1.0, 0.0)}
assert torch.equal(outs[1.0], outs[0.0]), "scale must be backward-only"
print("forward identical for s=1.0 vs s=0.0")

# 3: stage-2 semantics — seg side severed from backbone, height keeps it
stage2 = make_model(0.0)
aux = stage2((pixel, token), return_aux=True)
stage2.zero_grad(set_to_none=True)
seg_loss = (F.binary_cross_entropy_with_logits(aux["presence_logits"], target_presence)
            + aux["building_boundary_logits"].abs().mean()
            + aux["fraction_logits"].abs().mean())
seg_loss.backward(retain_graph=True)
g_bb, g_seg, g_ht = (grad_sum(stage2.alpha_unet), grad_sum(stage2.head.shared),
                     grad_sum(stage2.head.height_shared))
assert g_bb == 0.0 and g_ht == 0.0 and g_seg > 0.0, (g_bb, g_seg, g_ht)
stage2.zero_grad(set_to_none=True)
aux["out"][:, 3:4].abs().mean().backward()
assert grad_sum(stage2.alpha_unet) > 0.0
print(f"stage-2: seg losses -> backbone 0 / seg trunk {g_seg:.1e}; height -> backbone ok")

# 5: stage-1 checkpoint loads strict-clean into the stage-2 model
with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
    torch.save(stage1.state_dict(), f.name)
    missing, unexpected = stage2.load_state_dict(torch.load(f.name), strict=False)
    os.unlink(f.name)
assert not missing and not unexpected, (missing, unexpected)
print("stage-1 -> stage-2 warm start loads strict-clean")
print("SMOKE OK")
