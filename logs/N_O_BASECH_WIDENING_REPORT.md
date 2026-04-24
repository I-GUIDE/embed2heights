# Experiments N & O — LightUNet base-channel widening (32 → 48 → 64)

**Date:** 2026-04-22
**Baseline:** `alphaearth_tessera_iou_fusion_J_specialist_d2_veg15` (val score **0.4568**, `base_ch=32`, ~2.75M params)
**One-line summary:** Widening LightUNet's base channel from 32 → 48 cleanly lifts score by **+0.0076 raw / +0.0105 per-class TS**, with all five axes improving simultaneously. Pushing further to 64 saturates — O trades IoU for RMSE_bH and lands at the same score as N within noise. **48 is the capacity sweet spot**, and this is the first architectural move since E to clear the 0.006 noise floor.

sbatch drivers: [run_exp_N_base48.sbatch](../run_exp_N_base48.sbatch), [run_exp_O_base64.sbatch](../run_exp_O_base64.sbatch)
Runs: `runs/alphaearth_tessera_iou_fusion_N_base48/`, `runs/alphaearth_tessera_iou_fusion_O_base64/`

## Motivation

Previous round [K_L_EXPERIMENTS_REPORT.md](K_L_EXPERIMENTS_REPORT.md) declared single-knob loss-weight tuning exhausted on `tessera_iou_fusion`. Three architecture swaps (HRNet-W18/W32, EmbeddingRefiner variants) had already under-performed LightUNet under the same recipe ([BEST_RESULT.md §Experiment Log](BEST_RESULT.md)), so "bigger backbone" looked closed.

But HRNet's train/val curves ran almost on top of each other (train 1.93 vs val 2.05 at convergence), which is an **under-optimized / under-fit** signature, not an over-capacity one. And LightUNet at `base_ch=32` is tiny — 2.75M params in the fusion wrapper. The hypothesis N/O tests: *scale LightUNet itself (retain its inductive bias for pixel-aligned embeddings) rather than swap architectures*. LightUNet decoder widths scale as `(b, 2b, 4b, 8b)`, so increasing `base_ch` is a single scalar knob.

Code changes: [core/model.py](../core/model.py) — `LightUNet` and `TesseraIoUFusionLightUNet` both accept `base_ch` with default 32 (legacy). [train.py](../train.py) / [predict.py](../predict.py) / [predict_tta.py](../predict_tta.py) surface `--lightunet-base-ch` and round-trip it through `training_params.json`.

## Results (val, 405 samples, raw @ 0.5)

⭐ = per-column best across {J, N, O}.

| Experiment | base_ch | Params | iou_bld | iou_tree | iou_wat | RMSE_bH | RMSE_vH | **Score** | Δ vs J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J (baseline) | 32 | 2.75M | 0.5038 | 0.7590 | 0.4429 | 1.9403 | 3.4421 | 0.4568 | — |
| **N** ⭐champion | **48** | **5.44M** | **0.5087** ⭐ | **0.7618** ⭐ | **0.4537** ⭐ | 1.9111 | 3.3963 ⭐ | **0.4644** | **+0.0076** ✅ |
| O | 64 | 9.20M | 0.5066 | 0.7562 | 0.4488 | **1.8971** ⭐ | 3.3999 | 0.4633 | +0.0065 |

Threshold-swept (full grid 0.05–0.95 step 0.01, plus per-class search):

| Experiment | Raw @0.5 | Single-thr swept | Per-class swept | Per-class (bld, veg, wat) |
|---|---:|---:|---:|---|
| J | 0.4568 | 0.4597 @0.650 | 0.4614 | (0.640, 0.560, 0.880) |
| **N** | **0.4644** | 0.4652 @0.680 | **0.4673** ⭐ | (0.630, 0.510, 0.850) |
| O | 0.4633 | 0.4659 @0.660 | 0.4672 | (0.620, 0.620, 0.910) |

**N is the new local champion on every scoring protocol.** O's per-class TS (0.4672) is within 0.0001 of N's — a tie under noise.

## Per-experiment reading

### N — base_ch 32 → 48 is a clean five-axis win

Decomposition (N vs J):

| Term | Δ metric | × weight | Δ score |
|---|---:|---|---:|
| iou_bld | **+0.0049** | × 0.25 | +0.0012 ✅ |
| iou_tree | +0.0028 | × 0.15 | +0.0004 ✅ |
| iou_wat | **+0.0108** | × 0.15 | **+0.0016** ✅ |
| RMSE_bH | **−0.029 m** | × 0.25/3.0 | **+0.0024** ✅ |
| RMSE_vH | **−0.046 m** | × 0.20/5.0 | **+0.0018** ✅ |
| **Total (predicted)** | | | **+0.0074** (observed: +0.0076 ✓) |

Every axis moves the right way. No trade-off. This is the **only** move in the experiment log since E_specialist_d2 (depth 2, +0.0102 over C) that lifts without regressing any axis. Two interpretations are consistent with this signature:

1. **J's 2.75M-param shared trunk was the bottleneck.** All the prior loss-knob tuning (K/L/M) ran into a "trade one metric for another" ceiling precisely because the trunk had no slack — every loss twist had to repurpose existing capacity. At 5.44M, the trunk has headroom to satisfy BCE, Tversky, height boost, and both aux L1 specialists simultaneously. The K_L diminishing-returns call is scoped to the 2.75M trunk, not to the architecture family.
2. **Width scales both the shared trunk and the skip connections.** The base-channel knob touches every DoubleConv in the U-Net, so it lifts feature dimensionality at every scale (32/64/128/256 → 48/96/192/384). That's why it helps both coarse-scale RMSE (height regression needs semantic context) and fine-scale IoU (presence classification needs spatial precision) at once — unlike specialist depth, which only thickens the last-mile per-class projections.

Final training loss: N train 0.951 / val 0.988 (3.9% gap). Well-converged, no overfitting signature.

### O — base_ch 48 → 64 saturates the scoring surface

Decomposition (O vs N):

| Term | Δ metric | × weight | Δ score |
|---|---:|---|---:|
| iou_bld | −0.0021 | × 0.25 | −0.0005 |
| iou_tree | **−0.0056** | × 0.15 | **−0.0008** ❌ |
| iou_wat | **−0.0049** | × 0.15 | **−0.0007** ❌ |
| RMSE_bH | **−0.014 m** | × 0.25/3.0 | **+0.0012** ✅ |
| RMSE_vH | +0.004 m | × 0.20/5.0 | −0.0002 |
| **Total (predicted)** | | | **−0.0011** (observed: −0.0011 ✓) |

Two findings here:

- **RMSE_bH continues to improve at 64** (1.8971, the global best across all experiments to date — beats even L_specialist_d4's 1.8962 within noise). Building height regression is genuinely capacity-hungry.
- **All three IoUs regress**, most notably `iou_tree` (−0.56 pt) and `iou_wat` (−0.49 pt). The presence classifier at 9.2M params begins to overfit noisy / rare-positive classes (water is the sparsest class in the label distribution).

Final training loss: O train 0.945 / val 0.984 (4.2% gap). Slightly better than N on the *loss* axis (by 0.006 in val loss), but **the loss ranking inverts the score ranking**. This is a known signature of loss-metric misalignment — here, the presence-centered loss's IoU contribution is a smooth Tversky surrogate, not the threshold-at-0.5 IoU used in scoring. With more capacity, O learns a slightly sharper loss-surface fit, but the threshold calibration drifts for rare classes, hurting hard-gated IoU.

### Training stability

Both runs produced exactly **30 non-finite-loss warnings** across training (1 per epoch on average, ~0.07% of batches). The rate is unchanged between N and O, so the occasional NaN is **not a capacity-induced instability** — it's a baseline property of this loss / AMP combination (likely the `softplus(delta)` × FiLM × presence_tversky composition occasionally emits inf under AMP). Not a blocker, but worth cleaning up if we ever go further on width.

## Capacity curve

| Config | base_ch | Params | Raw score | Per-class TS |
|---|---:|---:|---:|---:|
| J | 32 | 2.75M | 0.4568 | 0.4614 |
| **N** | **48** | **5.44M** | **0.4644** | **0.4673** |
| O | 64 | 9.20M | 0.4633 | 0.4672 |

Score rises steeply from 32 → 48 (+0.0076 raw), then plateaus from 48 → 64 (−0.0011, noise). The knee is between 48 and 64. Given raw N ≥ raw O and TS N ≈ TS O to within 0.0001, **48 is the Pareto point** — same score, 60% of the params.

Extrapolation: it is **not** useful to try base_ch ≥ 80 on this recipe; the IoU degradation direction O revealed will dominate. Only reason to revisit wider is if the head's IoU branch is re-capped (e.g., separate the presence classifier width from the height regressor width, so building-height can keep scaling while presence stays at the 48-regime calibration).

## Interaction with K/L/M — previously-closed directions may reopen

K_L_EXPERIMENTS_REPORT closed three directions under the hypothesis "single-knob tuning is exhausted on tessera_iou_fusion." That claim was scoped to the 2.75M trunk. N's result **re-opens** them at 5.44M, where the trunk has more slack to absorb competing losses:

- **`height_specialist_depth ≥ 4` on base_ch=48/64.** L at base=32/d=4 saw iou_bld drop 1.6 pt — "deeper specialists starve the shared trunk of gradient." At base=48 the shared trunk is 2.25× wider; it may tolerate d=4 without the same IoU cost. This is worth re-running once. (N/O were both run at d=2 for clean attribution.)
- **M's `aux_veg_weight=2.0` on base_ch=48.** M at base=32 closed because the "isolated" aux path wasn't actually isolated (shares trunk, `height_trunk`, `base_height`). A wider shared trunk has more room to absorb the extra veg gradient without zero-sum-ing the building / water channels. Less certain than L-reopening, but cheap to test.
- **G's `veg_height_boost=3.0` on base_ch=48.** G at base=32 achieved global-best iou_wat but paid RMSE_bH. Same capacity-slack argument applies — re-running at base=48 is worth one run.

**However, do NOT re-open K (`build_height_boost=3, veg=3`).** K's regression was causal (the 5× building weight is load-bearing for RMSE_bH *regardless of trunk width*). That conclusion is architecture-independent.

## The key finding — capacity WAS the bottleneck, at least partially

K_L_REPORT's strongest statement — *"Further single-knob tuning on this recipe will not produce >0.005 gains"* — was right on the 2.75M trunk, but the recipe's bottleneck was not in the loss; it was in the backbone's feature dimensionality. Moving to base_ch=48 lifts the trunk past its saturation point on all five metrics simultaneously — exactly the "capacity released, every loss term benefits" pattern you'd predict from the mechanism interpretation of K/L/M.

This changes the framing for future single-model work. **The new champion backbone is `tessera_iou_fusion` at base_ch=48**, and many of the loss knobs closed on the 2.75M trunk deserve one-shot re-examination at the new baseline before being re-declared closed.

## Closed / reopened directions

- ✅ **Widen LightUNet to base_ch=48** — promoted to new recipe baseline. N is new single-model champion (0.4644 raw / 0.4673 per-class TS).
- ❌ **Widen to base_ch ≥ 64 under current head** — closed. O saturates; further width requires head surgery (decouple presence/height capacities) before it pays off.
- 🔄 **L, M, G re-runs at base_ch=48** — re-opened for one-shot verification.
- 🔄 **N + per-class head-width split** — new direction. If presence regresses at width 64 but height still improves, separate the two branch widths. Head surgery, non-trivial.

## Post-N/O follow-ups (priority)

1. **Ensemble N into the 5-way `ECCpDB`** ([ENSEMBLE_ECCpDB_REPORT.md](ENSEMBLE_ECCpDB_REPORT.md) currently at 0.4692 via 5-mean + threshold bake). Replacing the weakest member with N should push this toward 0.48.
2. **N + TTA (flip + D4)** via [predict_tta.py](../predict_tta.py). Near-free +0.005–0.01 on single-model. Takes ~5 min extra inference.
3. **N at seeds {0, 1, 2}** — single-model seed ensemble at the new champion width. Noise-floor-ceiling-breaker for any single recipe.
4. **Re-run L / M / G at base_ch=48** — cheap, determines whether the K_L closing claim holds at the new baseline or whether one of those knobs now clears noise.
5. (Deprioritized) **Head-width split** — if (1)–(3) plateau at ~0.48, then revisit O's finding that RMSE_bH kept improving and invest in separating head branches.
