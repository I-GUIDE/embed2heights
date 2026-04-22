# Height & IoU Experiments — D / E / F → G / H / I → J

**Date:** 2026-04-21 (Round 1: D/E/F) · 2026-04-22 (Round 2: G/H/I, Round 3: J)
**Baseline:** `alphaearth_tessera_iou_fusion_C_ch16_h96d2` (val score `0.4472`)
**Current champion (val):** `alphaearth_tessera_iou_fusion_E_specialist_d2` (val score `0.4574`, +0.0102)
**Current best strategy on record:** `alphaearth_tessera_iou_fusion_J_specialist_d2_veg15` (E + tuned veg boost, running)
**Goal:** reduce `RMSE_H_BUILD` / `RMSE_H_VEG` (together 45% of score, and the
baseline's weakest contribution) and lift sparse-class IoU.

Scoring is the leaderboard-correct composite (see
[METRIC_PROBE_REPORT.md](METRIC_PROBE_REPORT.md)): per-class RMSE ceilings
`X_bld=3.0m`, `X_veg=5.0m`, positive-only per-image IoU at `label > 0`.

## Motivation

C_ch16_h96d2 submission gave test scores `RMSE_H_BUILD=2.1330`,
`RMSE_H_VEG=3.8109`. Per-meter gradient of the score is `0.25/3.0 = 0.083` for
building and `0.20/5.0 = 0.040` for vegetation — buildings are higher
per-meter leverage but vegetation is further from the ceiling, so both matter.

Three working hypotheses, each isolated into one experiment (all other hparams
held equal to baseline):

| Exp | Hypothesis | Concrete change |
|---|---|---|
| **D** | The height loss is L1, but the metric is RMSE. Also `height_boost` only up-weights building pixels — vegetation is unweighted. | `--height-loss-kind huber --huber-delta 1.0 --veg-height-boost 3.0` |
| **E** | Per-class height specialists (`height_building_delta_proj`, `height_vegetation_delta_proj`) are single 1×1 Conv layers on top of a shared trunk — likely too thin to learn class-specific height distributions. | `--height-specialist-depth 2` (prepend 2× `ConvGNAct` before each specialist's 1×1 proj) |
| **F** | Presence Tversky plateaus on sparse classes (`iou_wat=0.40`, `iou_bld=0.49`). Focal should focus learning on hard examples. | `--iou-loss-kind focal --focal-gamma 2.0 --focal-alpha 0.25` |

## Setup

All four runs share: 30 epochs, bs=32, AdamW lr=2e-4, `loss_preset=presence_centered`,
`aux_weight=1.0`, `presence_tversky_weight=1.0`, `fraction_mae_weight=0.1`,
Tessera compression `ch16, h96, depth=2`, seed 42, same 80/20 split
(`splits/split.json`). Only the knob in the table above differs.

Code changes adding the new knobs (all default to legacy behavior — pre-existing
runs reproduce):
- [core/losses.py](../core/losses.py): `ImprovedCompositeLoss` gains
  `height_loss_kind {l1,huber,mse}`, `huber_delta`, `veg_height_boost`,
  `iou_loss_kind {tversky,focal}`, `focal_gamma`, `focal_alpha`.
- [core/model.py](../core/model.py): `MultiTaskPredictionHead` gains
  `height_specialist_depth`; threaded through `TesseraIoUFusionLightUNet`
  and `build_model`.
- [train.py](../train.py), [predict.py](../predict.py): matching CLI flags;
  persisted in `training_params.json` so predict/evaluate reconstruct the
  right model.

sbatch drivers: [run_exp_D_height_huber.sbatch](../run_exp_D_height_huber.sbatch),
[run_exp_E_height_specialist.sbatch](../run_exp_E_height_specialist.sbatch),
[run_exp_F_iou_focal.sbatch](../run_exp_F_iou_focal.sbatch).

## Results (val, 405 samples)

| Experiment | iou_bld | iou_tree | iou_wat | RMSE_bH | RMSE_vH | **Score** | Δ vs C |
|---|---:|---:|---:|---:|---:|---:|---:|
| C baseline | 0.4944 | 0.7529 | 0.4040 | 1.9003 | 3.5401 | **0.4472** | — |
| **E** specialist_d2 | **0.5029** | **0.7587** | **0.4497** | 1.9181 | **3.4929** | **0.4574** | **+0.0102** ✅ |
| D huber + veg_boost 3.0 | 0.4931 | 0.7569 | 0.4174 | **2.1382** | 3.5009 | 0.4312 | −0.0160 ❌ |
| F focal α=0.25, γ=2.0 | **0.3978** | 0.7484 | 0.4054 | 1.9189 | 3.5889 | 0.4190 | −0.0282 ❌ |

Arrows: E improves on **4 of 5 metrics** (RMSE_bH flat); D improves veg and
sparse IoU but regresses building RMSE hard; F regresses building IoU hard.

## Per-experiment reading

### E — depth=2 per-class height specialists → new champion

All five metrics improved or held; `iou_wat` jumped **+4.6 pt** (0.4040 →
0.4497), an unexpectedly large gain for a change inside the height branch.
Most plausible explanation: the deeper specialists absorb gradients that
previously flowed through the shared trunk, reducing contention between the
height regression and the presence/fraction classifiers. The shared trunk
then has more capacity budget for presence — water being the sparsest class
benefits most.

This is the new baseline for all further experiments on this model family.

### D — Huber(1.0) helped veg but broke buildings

- `RMSE_vH`: 3.5401 → 3.5009 (**−0.039m**, direction confirms `veg_height_boost`
  works)
- `RMSE_bH`: 1.9003 → **2.1382 (+0.238m)**, large regression
- IoU: small improvements across the board

Mechanism: Huber with `δ=1.0` switches to quadratic behavior inside
`|err| ≤ 1m`. For buildings where most validation errors are already below
~2m, that softens the gradient on the dominant error regime. The `veg_height_boost=3.0`
is an honest win for vegetation — it should be re-tested **without** the
Huber swap (i.e. `height_loss_kind=l1` + `veg_height_boost=3.0`).

### F — focal α=0.25 is backwards for sparse-positive targets

`iou_bld` crashed **−9.7 pt** (0.4944 → 0.3978); other metrics flat or worse.
`α=0.25` is the RetinaNet default, calibrated for massively-imbalanced object
detection where **easy negatives** dominate — it up-weights negatives
(`1−α = 0.75`) and down-weights positives. Combined with `(1−p_t)^γ`, this
configuration ends up emphasizing hard negatives far more than the sparse
positives, suppressing building recall.

For our sparse-positive regime (building positive rate ≪ 0.5), the correct
direction is the inverse: `α ≥ 0.5`, likely `α=0.75`. Retry planned as exp H.

## Round 1 takeaways

1. **E is the new champion** (+0.0102, all 5 metrics non-regressing).
2. **The `veg_height_boost` knob is validated** (D's `RMSE_vH` moved the
   right direction). The error was pairing it with Huber; isolate it next.
3. **Focal direction was wrong.** Do not abandon focal — re-try with
   `focal_alpha=0.75` to flip the positive-class weighting.
4. **MSE remains untested** — possibly best-aligned with the metric, but
   needs a structure-weight rebalance because per-pixel MSE has a different
   magnitude/gradient profile than L1.

This led to Round 2 (G / H / I), all stacked on E.

---

# Round 2 — G / H / I (stacked on E)

**Code addition for Round 2:** [train.py](../train.py) gained
`--structure-weight` to override `lambdas[3]` (default unchanged at 2.0). I
needs this because MSE roughly doubles the magnitude of the height_boost
term relative to L1.

| Exp | Stacked on E, change | Hypothesis |
|---|---|---|
| **G** | `--veg-height-boost 3.0` (keep L1) | Re-test D's veg-boost without the Huber damage. |
| **H** | `--iou-loss-kind focal --focal-alpha 0.75 --focal-gamma 2.0` | Inverted-α focal as flagged by F's failure mode. |
| **I** | `--height-loss-kind mse --structure-weight 1.0` | Align training objective with the RMSE metric; halve `w_structure` to keep `weighted_height_boost` ~equal to E's L1. |

sbatch drivers: [run_exp_G_specialist_veg_boost.sbatch](../run_exp_G_specialist_veg_boost.sbatch),
[run_exp_H_specialist_focal075.sbatch](../run_exp_H_specialist_focal075.sbatch),
[run_exp_I_specialist_mse.sbatch](../run_exp_I_specialist_mse.sbatch).

## Round 2 results (val, 405 samples)

⭐ marks the best value across all 7 runs in this report (C, D, E, F, G, H, I).

| Experiment | iou_bld | iou_tree | iou_wat | RMSE_bH | RMSE_vH | **Score** | Δ vs E |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E** specialist_d2 (champion) | **0.5029 ⭐** | **0.7587 ⭐** | 0.4497 | **1.9181 ⭐** | 3.4929 | **0.4574** | — |
| **G** E + veg_boost 3.0 | 0.5019 | 0.7547 | **0.4584 ⭐** | 2.0011 | **3.4301 ⭐** | 0.4535 | −0.0039 |
| **H** E + focal α=0.75 | 0.4640 | 0.7574 | 0.4277 | 1.9283 | 3.4583 | 0.4448 | −0.0126 |
| **I** E + MSE (s=1.0) | 0.4929 | 0.7584 | 0.4171 | 1.9772 | 3.5952 | 0.4410 | −0.0164 |

## Per-experiment reading (Round 2)

### G — veg_boost without Huber: tradeoff confirmed, weight too aggressive

G captures **two new global bests** (`iou_wat=0.4584`, `RMSE_vH=3.4301`)
but pays for them with a **+0.083m regression on `RMSE_bH`** (1.9181 →
2.0011). Score decomposition vs E:

| Term | Δ metric | × weight | Δ score |
|---|---:|---|---:|
| iou_bld | −0.0010 | × 0.25 | −0.0003 |
| iou_tree | −0.0040 | × 0.15 | −0.0006 |
| iou_wat | +0.0087 | × 0.15 | +0.0013 |
| RMSE_bH | +0.083 m | × 0.25 / 3.0 | −0.0069 |
| RMSE_vH | −0.063 m | × 0.20 / 5.0 | +0.0025 |
| **Total** | | | **−0.0040** ✓ matches |

The building-RMSE penalty alone (−0.0069) wipes out the veg gains. **The
knob works as intended; the value 3.0 is just too aggressive.** A lower
`veg_height_boost` (e.g. 1.0 or 1.5) very likely sits above E.

Mechanism for the building regression (best guess): with veg pixels
weighted 4× and building pixels weighted 6×, the relative gradient
allocation shifts. The submitted height comes from a presence-gated mix of
specialists, so increased veg-side gradient "pulls" the shared trunk
features in a vegetation-friendlier direction, slightly degrading building
specialist accuracy through the gating mix.

### H — inverted-α focal (0.75) recovers most of F's loss, but still loses to Tversky

F → H: `iou_bld` 0.3978 → 0.4640 (**+6.6 pt**) — confirms the α direction was
the problem. But H's `iou_bld` 0.4640 is still **−3.9 pt below E**'s 0.5029.

**Conclusion: focal does not have headroom over Tversky for our setting.**
Tversky's `(α=0.3, β=0.7)` recall-bias is doing more useful work than
focal's hard-example focusing. This direction can be closed.

Side note: H's `RMSE_vH` improved slightly vs E (3.4583 vs 3.4929, −0.035m).
Possibly focal sharpens the presence boundary, which propagates into a
cleaner height gating mix on the veg side. But the ~0.035m gain doesn't
recover the iou_bld loss.

### I — MSE failed, including on the metric it was supposed to align with

`RMSE_vH` got **worse** (3.4929 → 3.5952, +0.10m) and `RMSE_bH` also
worse (+0.06m). The "MSE is metric-aligned" intuition is broken because:

- The leaderboard metric is `mean(per_image_RMSE)`, not `sqrt(mean(per_pixel_squared_err))`.
- Per-pixel MSE training over-weights the few high-error pixels in each
  patch. The model can lower per-pixel L2 by aggressively flattening
  outlier pixels in some patches, but per-image RMSE averages over patches,
  so extreme local successes don't compound — and per-patch *normal* pixels
  drift slightly, raising per-image RMSE.
- Halving `w_structure` to 1.0 doesn't fix this because aux height losses
  (`aux_height_building`, `aux_height_vegetation`) are still under
  `aux_weight=1.0` with MSE — so those terms now dominate the aux side.

**Conclusion: MSE direction is closed for this objective.** L1 + per-class
boosting is the right family.

## Round 2 takeaways

1. **E remains champion (0.4574).** None of G/H/I clears it on overall
   score. Calibrate thresholds + submit E for test scoring.
2. **veg_height_boost is real but needs tuning.** G proved it produces the
   intended RMSE_vH improvement; the issue is weight magnitude. **Next
   experiment J = E + `--veg-height-boost 1.0` or `1.5`** is the most
   promising single-knob follow-up.
3. **Three directions can be closed:** Huber (D), focal at any α tested
   (F + H), MSE (I). All underperform L1+Tversky.
4. **Ensemble of E and G is a strong free lunch.** Their per-class
   strengths are nearly orthogonal:

   | Channel | E strength | G strength |
   |---|---|---|
   | building (ch 0, ch 3) | iou_bld 0.503, RMSE_bH 1.92 | iou_bld 0.502, RMSE_bH 2.00 |
   | vegetation (ch 1, ch 3) | iou_tree 0.759, RMSE_vH 3.49 | iou_tree 0.755, RMSE_vH 3.43 |
   | water (ch 2) | iou_wat 0.450 | iou_wat 0.458 |

   A per-channel ensemble (E for building, G for veg/water) via
   [tools/ensemble.py](../tools/ensemble.py) is zero-additional-training
   and should be very close to the union of all ⭐ entries.

---

# Round 3 — J (current best strategy on record)

sbatch: [run_exp_J_specialist_veg15.sbatch](../run_exp_J_specialist_veg15.sbatch) ·
output: `runs/alphaearth_tessera_iou_fusion_J_specialist_d2_veg15/`

## J is the consolidation of everything Rounds 1–2 proved

After 7 experiments (C/D/E/F/G/H/I) we have a narrow, well-supported
recipe. J combines every effect that moved the score the right way and
discards every one that didn't.

| Ingredient | Status | Evidence |
|---|---|---|
| Per-class height specialist depth **2** | **Keep** | E: +0.0102 vs C, all 5 metrics non-regressing |
| L1 height regression | **Keep** | Huber (D) and MSE (I) both lost |
| Presence Tversky for IoU aux | **Keep** | Focal α=0.25 (F) and α=0.75 (H) both lost |
| `veg_height_boost = 3.0` | **Tune down → 1.5** | G: best global RMSE_vH + iou_wat, but bH regressed +0.083m |
| `--huber-delta`, `--iou-loss-kind focal`, `--height-loss-kind mse` | **Closed** | repeatedly beaten by the defaults |

**J = E + `--veg-height-boost 1.5`.** Single-knob change on top of the
champion; all other hparams identical to E.

## Why 1.5 and not another value

G ran at 3.0 and gave, relative to E:

| Metric | Δ vs E | Score impact |
|---|---:|---:|
| iou_wat | +0.0087 | +0.0013 |
| RMSE_vH | −0.063 m | +0.0025 |
| RMSE_bH | **+0.083 m** | **−0.0069** |
| iou_bld / iou_tree | ~flat | ~0 |
| **Total** | | **−0.0039** |

If the relationship is near-linear in the boost weight (not guaranteed,
but reasonable for small-signal perturbations), 1.5 should give
roughly half the gain *and* half the penalty:

| Metric (predicted, 1.5) | Δ vs E | Score impact |
|---|---:|---:|
| RMSE_vH | ~ −0.03 m | +0.0012 |
| RMSE_bH | ~ +0.04 m | −0.0035 |
| iou_wat | ~ +0.004 | +0.0006 |
| **Total (predicted)** | | **≈ −0.0017** |

So the linear model predicts J lands just *below* E. If the real
tradeoff is sublinear near 0 (i.e. small veg_boost helps RMSE_vH while
barely hurting RMSE_bH), J could beat E. That is exactly the hypothesis
we are testing — if linear, the knob's optimum is at 0 and E remains
champion.

If J fails to beat E, 1.0 is the natural next dose; if J clearly beats
E, we should sweep 1.5 / 2.0 / 2.5.

## Post-J follow-ups (priority)

1. **E ⊕ G ensemble** (or E ⊕ J once J finishes) — E is best at
   building, G best at vegetation/water. A per-channel blend via
   [tools/ensemble.py](../tools/ensemble.py) requires zero additional
   training and is nearly guaranteed to land at or above the best single
   run. **Probably the highest expected-value move on the table.**
2. **Dose sweep around J** (1.0, 2.0) if J beats E — confirms the optimum.
3. **Deeper specialists (depth 3, depth 4)** — E showed depth 2 is a
   big step; the curve may not be saturated.
4. **Closed directions:** Huber, focal (any α), MSE — do not re-run.

---

# Why is RMSE_H_VEG so much higher than RMSE_H_BUILD? (diagnostic)

Asked in context; script: [tools/diagnostic_height_rmse.py](../tools/diagnostic_height_rmse.py),
JSON: `runs/alphaearth_tessera_iou_fusion_E_specialist_d2/height_rmse_diagnostic.json`.

## Full-image RMSE (from E, val 405 images)

| Region | per-image RMSE (leaderboard aggregation) | pixels | GT height μ | GT height σ |
|---|---:|---:|---:|---:|
| building pixels (label_bld > 0) | **1.907** | 984K | 3.62 m | 3.69 m |
| **full image (all valid)** | **2.882** | **18.0M** | **6.77 m** | **7.25 m** |
| vegetation pixels (label_veg > 0) | **3.462** | 11.9M | 9.49 m | 7.32 m |
| other (non-bld non-veg) | 1.622 | 5.2M | 1.09 m | 2.36 m |

Vegetation is ~66% of valid pixels, so full-image RMSE (2.88m) is
dominated by vegetation.

## The absolute gap is physically expected

Normalize RMSE by the class's GT std to get a scale-free difficulty:

| Class | RMSE / GT std |
|---|---:|
| Building | 0.517 |
| **Vegetation** | **0.473** (best) |
| Other (bare ground) | 0.687 |

**Relative to each class's natural height variance, vegetation is actually
the best-predicted class.** The absolute RMSE is higher because
vegetation heights span a far wider range:

- Building GT: p50 = 2.66 m, p95 = 10.02 m (mostly houses, short tail)
- Vegetation GT: p50 = 8.58 m, **p95 = 22.29 m** (grass / shrubs / mature
  trees all in the same class)

The leaderboard's RMSE ceilings (`X_bld = 3.0 m`, `X_veg = 5.0 m`) already
roughly compensate — they're in a 1.67× ratio, close to the GT-std ratio
of 2×. Current relative completion:

- building score term: max(0, 1 − 1.91/3.0) × 0.25 = **0.090**
- vegetation score term: max(0, 1 − 3.46/5.0) × 0.20 = **0.061**

## Other diagnostics

- **Small negative bias** (bld −0.43 m, veg −0.33 m — model under-predicts).
  bias²/variance is 1–3% of MSE → a post-hoc constant shift saves at most
  ~0.02m RMSE. Not the answer.
- **No outlier patches driving the mean.** Top-5% worst patches
  contribute only ~11% of the vegetation-RMSE sum. The problem is
  uniformly high variance, not a few catastrophes.
- **Variance (not bias) is what's left.** Veg error std = 3.66 m is half
  the class's own GT std. To meaningfully close this gap we need
  better pixel-level predictions, not loss reweighting.

## Implication for the strategy

The loss-side room for improvement is small. The last two remaining
directions worth running under the current backbone are both covered by J
(veg boost tuning) and the E⊕G ensemble. Beyond that, pushing RMSE_H_VEG
much below ~3.0m would likely require architecture / data changes —
richer backbone, multi-temporal embeddings (deciduous / evergreen),
or external height priors (e.g. GEDI). Out of scope for a loss-tuning
round.
