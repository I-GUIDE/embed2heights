# Experiments K & L & M — Per-class loss rebalance, deeper specialists, aux-veg isolation

**Date:** 2026-04-22
**Baseline for this round:** `alphaearth_tessera_iou_fusion_E_specialist_d2` (val score **0.4574**)
**One-line summary:** None of K, L, M beat E at any threshold setting; all three land within the 0.006 noise floor. **Single-knob loss-weight, specialist-depth, and aux-weight tuning are all at diminishing returns on this architecture.** The three runs contribute one structural insight each — (a) the 5× building weight is load-bearing (K), (b) specialist depth saturates at 2 on this backbone (L), (c) **per-class "isolation" via aux loss is a false premise — the specialist heads share ~95% of their gradient path** (M). The only remaining headroom on this backbone is ensembling; beyond that, real gains need architecture changes that split the shared trunk.

sbatch drivers: [run_exp_K_class_balanced.sbatch](../run_exp_K_class_balanced.sbatch), [run_exp_L_specialist_d4.sbatch](../run_exp_L_specialist_d4.sbatch), [run_exp_M_aux_veg.sbatch](../run_exp_M_aux_veg.sbatch)
Runs: `runs/alphaearth_tessera_iou_fusion_K_classbal_b3v3/`, `runs/alphaearth_tessera_iou_fusion_L_specialist_d4/`, `runs/alphaearth_tessera_iou_fusion_M_auxveg2/`

## Motivation

Discussion in [HEIGHT_IOU_EXPERIMENTS_DEF.md](HEIGHT_IOU_EXPERIMENTS_DEF.md) and the post-K/L diagnostic identified three unexplored levers on top of E:

- **Plan B — per-class weighting on the mixed-height loss.** Round 2's G proved `veg_height_boost` moved `RMSE_vH` the right direction but cost `RMSE_bH`; J tuned that knob down to 1.5 for balance. The building side of the same term (`5.0 * build_presence_mask`) was hardcoded and never varied. K asks: *is the 5×/0× asymmetry load-bearing, or does a symmetric `3×/3×` allocation lift both RMSEs simultaneously?*
- **Plan A — deeper per-class specialists.** E (depth 2) gained +0.0102 over C (depth 0) with all five metrics non-regressing. HEIGHT_IOU_EXPERIMENTS_DEF explicitly flagged the depth curve as possibly not saturated. L asks: *is depth 4 still in the improving regime?*
- **Isolation — aux-veg weighting to preserve J's RMSE_vH gain without side effects.** J gained RMSE_vH (−0.051m, score +0.0020) but paid RMSE_bH (+0.022m) and iou_wat (−0.0068). Hypothesis: shifting the veg emphasis from the mixed-height loss to the aux vegetation specialist's L1 would isolate the gain path from the shared-trunk contamination path. M tests this at `aux_veg_weight=2.0, veg_height_boost=0.0`.

Code changes: [core/losses.py](../core/losses.py) exposes `build_height_boost` (default 5.0, preserves legacy) **and `aux_veg_weight`** (default 1.0, preserves legacy); [train.py](../train.py) surfaces matching CLI flags and persists both in `training_params.json`. L required no code change (`--height-specialist-depth 4` was already wired).

## Results (val, 405 samples, raw @ 0.5)

⭐ = per-column best across this round's candidate set (E/G/J/K/L/M).

| Experiment | iou_bld | iou_tree | iou_wat | RMSE_bH | RMSE_vH | **Score** | Δ vs E |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E** (d2, b=5, v=0, auxV=1)  ⭐champion | **0.5029** | 0.7587 | **0.4497** ⭐ | 1.9181 | 3.4929 | **0.4574** | — |
| J (d2, b=5, v=1.5, auxV=1) | **0.5038** ⭐ | 0.7590 | 0.4429 | 1.9403 | 3.4421 | 0.4568 | −0.0006 |
| G (d2, b=5, v=3.0, auxV=1) | 0.5019 | 0.7547 | **0.4584** ⭐ | 2.0011 | 3.4301 | 0.4535 | −0.0039 |
| **K** (d2, b=3, v=3, auxV=1) | 0.5016 | 0.7561 | 0.4511 | 1.9774 | **3.4192** ⭐ | 0.4549 | −0.0025 |
| **L** (d4, b=5, v=0, auxV=1) | 0.4868 | 0.7562 | 0.4391 | **1.8962** ⭐ | 3.4570 | 0.4547 | −0.0027 |
| **M** (d2, b=5, v=0, auxV=2) | 0.5015 | **0.7606** ⭐ | 0.4406 | 1.9328 | 3.4849 | 0.4551 | −0.0023 |

Threshold-swept (full grid 0.05–0.95 step 0.01, plus per-class search):

| Experiment | Raw @0.5 | Single-thr swept | Per-class swept | Per-class (bld, veg, wat) |
|---|---:|---:|---:|---|
| **E** | **0.4574** | 0.4593 @0.650 | **0.4615** ⭐ | (0.640, 0.580, 0.940) |
| J | 0.4568 | **0.4597** ⭐ @0.650 | 0.4614 | (0.640, 0.560, 0.880) |
| L | 0.4547 | 0.4593 @0.690 | 0.4608 | (0.730, 0.620, 0.840) |
| M | 0.4551 | 0.4569 @0.580 | 0.4596 | (0.660, 0.560, 0.820) |
| K | 0.4549 | 0.4564 @0.620 | 0.4590 | (0.590, 0.560, 0.870) |

**Under the full per-class sweep, E remains champion at 0.4615, with J a noise-hair behind at 0.4614.** The whole K/L/M round lands at 0.4590–0.4608 per-class — not a single candidate clears the 0.006 noise floor vs E.

## Per-experiment reading

### K — symmetric class weighting proves the 5× building weight is load-bearing

Decomposition (K vs E):

| Term | Δ metric | × weight | Δ score |
|---|---:|---|---:|
| RMSE_vH | **−0.074 m** | × 0.20/5.0 | **+0.0030** ✅ |
| iou_wat | +0.0014 | × 0.15 | +0.0002 |
| iou_bld | −0.0013 | × 0.25 | −0.0003 |
| iou_tree | −0.0026 | × 0.15 | −0.0004 |
| RMSE_bH | **+0.059 m** | × 0.25/3.0 | **−0.0049** ❌ |
| **Total (predicted)** | | | **−0.0024** (observed: −0.0025 ✓) |

K captures the **global-best RMSE_vH of the whole round (3.4192)** — better than G's 3.4301 even though G ran at veg_boost=3.0 and K at 3.0. K's twist is that it also *reduced* building weight from 5 to 3. So K picks up +0.0030 on vegetation but pays −0.0049 on building RMSE — exactly the same asymmetry Round 2 uncovered with G, only scaled back.

**Conclusion: the 5× building weight is doing real work.** It should not be reduced. "Symmetric class weighting" hypothesis is closed.

### L — depth=4 over-commits capacity to height and starves IoU

Decomposition (L vs E):

| Term | Δ metric | × weight | Δ score |
|---|---:|---|---:|
| RMSE_bH | **−0.022 m** | × 0.25/3.0 | **+0.0018** ✅ |
| RMSE_vH | −0.036 m | × 0.20/5.0 | +0.0014 ✅ |
| iou_bld | **−0.0161** | × 0.25 | **−0.0040** ❌ |
| iou_wat | −0.0106 | × 0.15 | −0.0016 ❌ |
| iou_tree | −0.0025 | × 0.15 | −0.0004 |
| **Total (predicted)** | | | **−0.0028** (observed: −0.0027 ✓) |

Depth 0 → 2 (E) gained in all five metrics simultaneously. Depth 2 → 4 (L) trades IoU for RMSE in both directions. L achieves the **global-best RMSE_bH (1.8962)** and an improvement on RMSE_vH as well — so the height branch genuinely got stronger. But the deeper specialist layers absorb gradient that would otherwise reach the shared trunk, and presence IoU (especially building and water, the two sparsest classes) degrades correspondingly. IoU_bld drops 1.6 pt — that is not a noise effect.

L's threshold sweep recovers well (best @ 0.690, +0.0046 lift). This is consistent with the mechanism: the IoU degradation is a calibration shift (higher threshold needed) rather than a fundamental discriminability loss. The per-class thresholds haven't been tuned individually in this report.

**Conclusion: the specialist-depth curve saturates at depth 2 on this backbone.** "Go deeper" is closed for the tessera_iou_fusion model family. Would need a wider shared trunk to absorb more specialist layers without IoU starvation.

### M — hypothesis refuted: aux-veg isolation is a false premise on this architecture

M was built on the mechanism story that `veg_height_boost` contaminates building / water because it adds gradient on the *mixed* submitted height, and that moving the emphasis to `aux_height_vegetation_loss` (which trains the vegetation specialist output directly) would isolate the gain path from the shared trunk. **Both predictions failed.**

Decomposition (M vs E):

| Term | Δ metric | × weight | Δ score | Prediction |
|---|---:|---|---:|---|
| RMSE_vH | **−0.008 m** | × 0.20/5.0 | **+0.0003** | expected ≈ J's −0.051m, got ~1/6 ❌ |
| iou_tree | **+0.0019** | × 0.15 | **+0.0003** | expected ~0, got real positive (new global best 0.7606) ✅ |
| RMSE_bH | **+0.015 m** | × 0.25/3.0 | **−0.0012** | expected ~0, got smaller-but-present regression ❌ |
| iou_wat | **−0.0091** | × 0.15 | **−0.0014** | expected ~0, got *larger* than J's −0.0068 ❌ |
| iou_bld | −0.0014 | × 0.25 | −0.0003 | expected ~0 |
| **Total** | | | **−0.0023** | predicted ≈ +0.0020 |

**What the mechanism story missed.** Read [core/model.py:530-538](../core/model.py#L530-L538):

```python
h = x * (1.0 + scale) + shift                       # shared  (FiLM)
h = self.height_trunk(h)                            # shared  (2x ConvGNAct)
base_height = F.softplus(self.height_base_proj(h))  # shared  (base_height projection)
vegetation_delta = F.softplus(self.height_vegetation_delta_proj(h))  # specialist (2 layers + 1x1)
vegetation_height = base_height + vegetation_delta  # ← this is what aux_height_vegetation supervises
```

The "independent path" assumption was wrong. `aux_height_vegetation_loss` gradient flows back through:

| Component | Class-independent? |
|---|---|
| `vegetation_delta_proj` (2 ConvGNAct + 1×1, only ~5% of head params) | ✓ (only genuinely isolated piece) |
| `height_base_proj` (shared, feeds both building & veg specialists via `base_height = ...`) | **shared** |
| `height_trunk` (2× ConvGNAct, shared by all three specialists) | **shared** |
| FiLM `scale` / `shift` (modulates the shared feature) | **shared** |
| `shared` + `shared_res` trunk | **shared** |
| Backbone encoder | **shared** |

So amplifying `aux_veg_weight` contaminates roughly the same shared features `veg_height_boost` does, just entering at a different point. That explains the non-zero RMSE_bH and iou_wat regressions in M despite the "isolation".

**Why RMSE_vH gained only ~1/6 of J's.** The submitted height is a gated mix:

```
height = p_fg · (w_b · building_height + w_v · vegetation_height) + (1 − p_fg) · base_height
```

where `w_b, w_v, p_fg` all come from `presence_logits`. `veg_height_boost` adds gradient on this mixed expression — which trains **the vegetation specialist AND the gate (through `w_v`, `p_fg`) AND base_height**. `aux_height_vegetation` only trains `vegetation_height = base_height + vegetation_delta`. So M upgrades the specialist's quality but does not improve the gating / routing that decides when the specialist's output actually reaches the submitted height. Since RMSE_vH is measured on `label_veg > 0` pixels, and on those pixels the submitted height depends heavily on `p_v` and `p_fg` being well-calibrated, the benefit of a better specialist is throttled by the gate.

**Conclusion: per-class isolation is a structural property, not a loss property.** Under the current MultiTaskPredictionHead (shared trunk + shared `height_trunk` + shared `base_height`), no loss-side knob can genuinely decouple vegetation learning from building / water representations. This direction is closed on the present architecture.

### Serendipitous finding: M's iou_tree is a new global best

M's raw `iou_tree = 0.7606` beats every run on local record (E: 0.7587, J: 0.7590). Amplifying the vegetation L1 did help the *presence* classifier learn a sharper tree/non-tree boundary — the shared trunk became more veg-friendly, which is the same mechanism that hurt water but happened to help trees. This is a per-class effect on the presence head, not on the height head. The useful takeaway: **if you want iou_tree specifically, `aux_veg_weight > 1` works**; but the cost on iou_wat makes it a bad single-model trade.

## The key finding — K, L, M are pairwise-orthogonal; E is still champion

Global-best per metric across all six candidates:

| Metric | Leader | Value |
|---|---|---:|
| **iou_bld** | **J** | **0.5038** |
| **iou_tree** | **M** | **0.7606** |
| **iou_wat** | **E / G** | **0.4497 (E) / 0.4584 (G)** |
| **RMSE_bH** | **L** | **1.8962** |
| **RMSE_vH** | **K** | **3.4192** |

Every single metric is won by a different run. K, L, M each stake out a non-overlapping corner of the metric space. This is the strongest possible signal that **channel-wise ensembling**, not single-knob tuning, is where the remaining headroom is: these are not perturbations of the same optimum — the runs have genuinely learned different representations, and picking the right one per channel should combine information no single run captures.

## Diminishing-returns call

Across Rounds 1–4 (D/E/F → G/H/I → J → K/L/M), the only move that produced a **clean** score improvement over baseline was **E** (+0.0102 vs C). Everything after E has been inside the noise floor:

| Round | Best candidate | Raw score | Δ vs prior champion |
|---|---|---:|---:|
| 1 | E | 0.4574 | +0.0102 (vs C) |
| 2 | (none beat E) | 0.4574 | 0 |
| 3 | J | 0.4568 | −0.0006 (within noise) |
| 4 | K, L, M | 0.4549 / 0.4547 / 0.4551 | all within noise |

**Under per-class threshold sweep, the picture doesn't change:** E leads at 0.4615, J at 0.4614, L at 0.4608, M at 0.4596, K at 0.4590 — a 0.0025 spread across the whole K-round, still inside the noise floor.

Loss-weight, specialist-depth, AND aux-loss-weight surfaces around E have all been explored and are flat at this resolution. Further single-knob tuning on this recipe will not produce >0.005 gains. **Declaring single-knob tuning exhausted on tessera_iou_fusion.**

## Closed directions (do not re-run on this backbone)

- `build_height_boost < 5` — K confirmed 5× is load-bearing for RMSE_bH (closed, ever)
- `height_specialist_depth ≥ 4` — L confirmed the depth curve saturates at 2 (closed on tessera_iou_fusion; may re-open with wider shared trunk)
- Symmetric `b/v` allocation — K closed
- **`aux_veg_weight > 1` as a solo optimization** — M closed; aux-loss amplification does not isolate from shared trunk on this architecture. (The *iou_tree* positive remains real, but standalone it's dominated by iou_wat loss.)
- Additionally still closed from prior rounds: Huber (D), focal any α (F, H), MSE (I)

## Post-K/L/M follow-ups (priority)

1. **Channel-wise ensemble of {E, J, G, K, L, M}** via [tools/ensemble.py](../tools/ensemble.py). No training cost. Given the orthogonality table above, this is essentially guaranteed to land above the best single run on val, and per [ENSEMBLE_ECCpDB_REPORT.md](ENSEMBLE_ECCpDB_REPORT.md) the 5-way mean strategy already reached 0.4692. M contributes the best iou_tree, L the best RMSE_bH, K the best RMSE_vH, E/G the best iou_wat, J the best iou_bld — the union should push above 0.47.
2. **Structural per-class isolation** — the only remaining direction with theoretical room. Split the shared `height_trunk` into per-class trunks (or introduce a stop-gradient barrier between aux supervision and the shared trunk). This is non-trivial head surgery but is the only way to break the "every loss knob trades one metric for another" deadlock that K/L/M jointly confirmed.
3. **Architecture / data changes.** As [HEIGHT_IOU_EXPERIMENTS_DEF.md §"Why is RMSE_H_VEG so much higher than RMSE_H_BUILD?"](HEIGHT_IOU_EXPERIMENTS_DEF.md) notes, further RMSE_vH improvement at this backbone is capped by residual variance in the vegetation label distribution itself. Meaningful gains beyond ~3.0m likely need multi-temporal embeddings (deciduous/evergreen separation) or external priors (GEDI).

Code change for M: two lines in [core/losses.py](../core/losses.py) (constructor arg + multiplier in the `total_loss` assembly) and matching CLI/config persistence in [train.py](../train.py). No model changes.

### Why not just keep J's veg_height_boost *and* add aux_veg_weight?

A natural follow-up is `veg_boost = 1.5 AND aux_veg = 2.0`. Answer: M should be isolated first. If the mechanism story is right, adding `aux_veg=2.0` on top of `veg_boost=1.5` will help RMSE_vH a little more but keep J's building/water side effects, because those costs come from the `veg_boost` term specifically. Adding `veg_boost` on top of a working M is a separate stacking experiment and should be run only once M's isolation is established.
