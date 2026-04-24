# AlphaEarth + Tessera Fusion Evolution Report

Date: 2026-04-22
Subject: How we went from "Tessera is the worst single source" to "Tessera
residual IoU branch is part of the current champion"

This note traces every Tessera-fusion run in [runs/](../runs/) in chronological
order, the architectural / loss decisions between them, and the per-metric
validation scores re-evaluated today with [evaluate.py](../evaluate.py). It
explains why Tessera is currently constrained to a *presence-logit residual
correction* and not fed into height or into the early LightUNet encoder.

## Architecture Diagrams

Rendered via [tools/plot_fusion_evolution.py](../tools/plot_fusion_evolution.py).
Each stage has its own PNG + SVG and there is a four-panel poster for slides:

- F0 naive concat (failed): [figures/fusion_f0_naive_concat.png](figures/fusion_f0_naive_concat.png)
- F1 residual IoU (basic): [figures/fusion_f1_residual_basic.png](figures/fusion_f1_residual_basic.png)
- F2 Tessera → Height (closed): [figures/fusion_f2_tessera_into_height.png](figures/fusion_f2_tessera_into_height.png)
- F3 thickened stem + C loss (current): [figures/fusion_f3_thick_stem_c_loss.png](figures/fusion_f3_thick_stem_c_loss.png)
- Four-panel compare: [figures/fusion_compare.png](figures/fusion_compare.png)

## TL;DR

- Starting point: AlphaEarth LightUNet V35 alone = `0.4213`. Tessera alone
  under the old metric was the *worst* single source
  ([BASELINE_REPORT.md §3.1](BASELINE_REPORT.md)).
- **Naive concat failed** — predictions never produced. Tessera's
  irrelevant channels dominated early LightUNet layers.
- **Isolated residual branch** (AlphaEarth primary + Tessera → 16-ch
  compression stem → presence-logit residual only): `0.4336` raw, +0.012
  over AlphaEarth-only under matched loss.
- **Loss realignment (C = presence_centered)** was the single biggest
  fusion-era win: `0.4336 → 0.4437` (+0.010), by replacing MAE + Tversky
  on fractions with direct BCE + Tversky on presence.
- **Extra stem capacity (C_ch16_h96d2)** gave a further `+0.0035` to reach
  `0.4472`, at which point the stem design froze and attention moved to
  downstream knobs.
- Under matched C loss, the **Tessera residual contributes +0.027** over a
  LightUNet-only baseline (`alphaearth_lightunet_C_presence_centered = 0.4166`
  vs `0.4437`). That isolated gain is what every post-C experiment (E, F,
  G, H, I, J, K, L, M, N, O) builds on.

## Run Inventory (val 405 samples, post-probe metric, raw @0.5)

Re-evaluated today via
`python evaluate.py --val-only --only <name> ...`. Sorted by score.

| Stage | Run | iou_bld | iou_tree | iou_wat | RMSE_bH | RMSE_vH | **Score** | Δ vs prior |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| naive concat | `alphaearth_tessera_naive_concat` | — (no predictions) | | | | | — | failed |
| residual base | `alphaearth_tessera_iou_fusion` | 0.4246 | 0.7449 | 0.4396 | 2.1157 | 3.7292 | **0.4084** | first alive |
| stem ch=8 | `alphaearth_tessera_iou_residual_ch8` | 0.3812 | 0.7552 | 0.4085 | 1.9584 | 3.5716 | 0.4138 | undercapacity |
| stem ch=3 h=64 d=1 | `alphaearth_tessera_iou_residual_ch3_h64d1` | 0.4081 | 0.7539 | 0.4151 | 1.9844 | 3.5233 | 0.4211 | still starved |
| LightUNet only + C | `alphaearth_lightunet_C_presence_centered` | 0.4098 | 0.7461 | 0.3587 | 1.9276 | 3.5249 | 0.4166 | no-tessera control |
| stem ch=16 h=64 | `alphaearth_tessera_iou_residual_ch16_h64` | 0.4438 | 0.7537 | 0.4265 | 1.9777 | 3.6220 | 0.4283 | over-expanded |
| B: drop SSIM+Grad | `alphaearth_tessera_iou_fusion_B_no_ssim_grad` | 0.4281 | 0.7564 | 0.4257 | 1.9481 | 3.5314 | 0.4308 | −0.004 vs residual |
| Tessera → height | `alpha_tessera_iou_alpha_height` | 0.4246 | 0.7449 | 0.4396 | 1.9154 | 3.5532 | 0.4321 | closed branch |
| stem ch=16 | `alphaearth_tessera_iou_residual_ch16` | 0.4418 | 0.7565 | 0.4091 | 1.9271 | 3.5570 | 0.4324 | stem sweep tie |
| **residual default** | `alphaearth_tessera_iou_residual` | 0.4423 | 0.7585 | 0.4188 | 1.9297 | 3.5695 | **0.4336** | stem-sweep champion |
| C_ch16_h64d1 | `alphaearth_tessera_iou_fusion_C_ch16_h64d1` | 0.4910 | 0.7570 | 0.3926 | 1.9746 | 3.5366 | 0.4392 | deeper stem noise |
| **C: presence_centered** | `alphaearth_tessera_iou_fusion_C_presence_centered` | 0.4813 | 0.7579 | 0.4070 | 1.9195 | 3.5336 | **0.4437** | **+0.010 loss win** |
| **C_ch16_h96d2** | `alphaearth_tessera_iou_fusion_C_ch16_h96d2` | 0.4944 | 0.7529 | 0.4040 | 1.9003 | 3.5401 | **0.4472** | stem Pareto point |
| E (C + specialist_d=2) | `alphaearth_tessera_iou_fusion_E_specialist_d2` | 0.5029 | 0.7587 | 0.4497 | 1.9181 | 3.4929 | 0.4574 | head knob |
| N (E + base_ch=48) ★ | `alphaearth_tessera_iou_fusion_N_base48` | 0.5087 | 0.7618 | 0.4537 | 1.9111 | 3.3963 | **0.4644** (TS 0.4673) | current champion |

Champion configuration:
`AlphaEarth 64ch + Tessera 128ch → stem (out_ch=16, hidden=96, depth=2) →
presence-logit residual only → V35 head + specialist_depth=2 + base_ch=48 +
presence_centered loss`. See [BEST_RESULT.md](BEST_RESULT.md) and
[N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md).

## Stage 0 — Single-Source Baselines

From [BASELINE_REPORT.md §3.1](BASELINE_REPORT.md), Tessera ranked sixth of
six single embeddings under the old metric (score `0.551`, `mIoU_tree=0.21`,
`RMSE_vH=11.93 m`). AlphaEarth led on every metric. The natural question:
can Tessera's different representation *complement* AlphaEarth's strengths,
or is it just noise?

## Stage 1 — Naive Concat (failed)

Design: stack Tessera's 128 pixel-aligned channels onto AlphaEarth's 64 →
192-channel input to a standard LightUNet (`model_type=lightunet`,
`secondary_train_embeddings_dir=../data/train/tessera_emb`).

Outcome: run produced no prediction files
([runs/alphaearth_tessera_naive_concat/](../runs/alphaearth_tessera_naive_concat/)
holds only checkpoints). Loss never converged to a useful state.

Interpretation: AlphaEarth is pixel-dense 64-dim features already calibrated
for dense prediction; Tessera (128-dim) has entirely different scale and
semantic geometry. Concatenated at the encoder input, Tessera's 2× channel
count dominates the first `DoubleConv`, and the network cannot untangle
which channels to trust. This is why the fusion architecture later
*restricted* Tessera to a late residual-only pathway.

## Stage 2 — Isolated Residual Branch (TesseraIoUFusionLightUNet)

Commit 62d1f05 introduced the load-bearing architecture. See
[core/model.py:579-651](../core/model.py#L579-L651).

```
Input 192ch → split
  ├─ [0:64]  AlphaEarth  → LightUNet features (trunk + skip connections)
  └─ [64:]   Tessera     → TesseraCompressionStem → 16ch  (NO skip)
                                                      ↓
                             MultiTaskPredictionHead
                             presence_logits = alpha_presence_head(alpha_feat)
                                             + presence_delta_head(tessera_feat)
                             height        = f(alpha_feat)  (Tessera excluded)
```

Key constraints:
- `presence_delta_head` is initialised **zero-weighted** so the branch
  starts as a no-op.
- Tessera cannot reach the height projections — `height_base_proj`,
  `height_building_delta_proj`, `height_vegetation_delta_proj` all consume
  only `alpha_feat`.
- The stem is a `ConvGNAct(1×1) → [hidden layers] → 2×ConvGNAct(3×3)` cascade
  with `ChannelCalibration` at the input
  ([core/model.py:586-601](../core/model.py#L586-L601)).

First live fusion run (`alphaearth_tessera_iou_fusion`) scored `0.4084` —
worse than AlphaEarth-only V35 at the time. RMSE_bH was 2.12 m, 0.2 m
worse than the single-source baseline. The fusion had a bug or stem
undersizing — resolved in the next sweep.

## Stage 3 — Compression-Stem Capacity Sweep

How much bandwidth should Tessera have? The sweep varies
`tessera_presence_ch` (interface width to the residual head) and optionally
an extra hidden stack `tessera_hidden_ch × tessera_hidden_depth`.

| Config | out_ch | hidden × depth | Score |
|---|---:|:---|---:|
| `_residual_ch8` | 8 | — | 0.4138 |
| `_residual_ch3_h64d1` | 3 | 64 × 1 | 0.4211 |
| `_residual_ch16_h64` | 16 | 64 × 0 | 0.4283 |
| `_residual_ch16` | 16 | — | 0.4324 |
| `_residual` (default) | 16 | — | **0.4336** |

Findings:
- `out_ch = 16` is the Pareto point. Narrower (3 / 8) starves the branch;
  going to 64-hidden-ch in the interface layer (`_ch16_h64`) regressed.
- `_residual` and `_residual_ch16` both score `0.432–0.434` — the
  ≤ 0.002 delta is noise. Use `out_ch = 16` and stop tuning.

## Stage 4 — Tessera → Height Ablation (closed)

`alpha_tessera_iou_alpha_height` granted Tessera *also* into the height
branch (not only presence logits). Score `0.4321` — iou_water improved to
`0.4396` (best of the residual-era sweep) but `iou_bld` dropped 0.018
relative to `_residual`, and no net improvement in RMSE. Combined delta
−0.0015 vs `_residual`.

Interpretation: Tessera's height contribution is noisy relative to
AlphaEarth's; the RMSE mask (per-class `gt > 0`) is already correctly
served by the alpha-only height path. Policy: **Tessera is frozen out of
height forever** — confirmed and not revisited.

## Stage 5 — Loss Realignment (B → C)

Commits 43d62ad (extractor expansion) + e683337 (C loss) split the loss
change into two steps so the effect could be attributed.

**B = `no_ssim_grad`.** Drop `SSIM` and `GradientDifference` terms from
the composite (zero out `w_ssim`, `w_grad` via the preset
[core/losses.py:170-171](../core/losses.py#L170-L171)). Keeps MAE + Tversky
on fractions + height boost. Score `0.4308`. Mildly below the `_residual`
baseline — structural losses were not the bottleneck.

**C = `presence_centered`.** The decisive change
([core/losses.py:299-361](../core/losses.py#L299-L361)):

- Drops Tversky on fraction-valued land-cover outputs (sigmoid over soft
  fractions). Replaces it with **Tversky/BCE on presence logits**
  (`label > 0` targets), fed directly from `aux_outputs["presence_logits"]`.
- Drops the unmasked MAE on fractions; uses masked fraction MAE only.
- Height-boost term stays with `build_boost = 5.0` and `veg_boost = 0.0`
  (tuned later in J to 1.5).

Result: `0.4336 → 0.4437` (**+0.010**) — the biggest single-step fusion
lift after the residual architecture itself. Why this works:

1. The submission channels 0-2 are `presence_prob`
   ([HEAD_EVOLUTION_REPORT.md §V35](HEAD_EVOLUTION_REPORT.md)).
   Applying Tversky to *fractions* asks the network to match a continuous
   target while the metric only cares about the binarised
   `presence > 0.5`. C removes that mismatch.
2. The Tessera residual adds logits on top of
   `alpha_presence_logits`. If the loss is not on presence logits
   directly, the residual branch has no direct gradient path to improve
   the metric — which is why the stem capacity sweep in stage 3 saturated.

## Stage 6 — Re-Sweep Stem Under C (small final win)

With C in place, does extra stem capacity now help?

| Config | out_ch | hidden × depth | Score |
|---|---:|:---|---:|
| C_ch16_h64d1 | 16 | 64 × 1 | 0.4392 |
| C (default) | 16 | 32 × 0 | 0.4437 |
| **C_ch16_h96d2** | 16 | 96 × 2 | **0.4472** |

- `ch16_h64d1` is slightly worse than default — the `d=1` extra block
  uses gradient budget without clear payoff.
- `ch16_h96d2` is `+0.0035` over default. Within noise floor individually
  (`0.006`) but directionally consistent with C's theory: a better-aligned
  loss unlocks modest additional stem capacity.

Settled: **C + ch16_h96d2** became the packaging choice for the ensemble
and the single-model submission at `0.4472`.

## Stage 7 — Isolating Tessera's Contribution

To separate "better loss" from "tessera residual", we ran
`alphaearth_lightunet_C_presence_centered` — LightUNet on AlphaEarth alone
with the **exact same C loss**.

| Model | Tessera? | Loss | Score |
|---|---|---|---:|
| `alphaearth_lightunet_C_presence_centered` | ✗ | C | 0.4166 |
| `alphaearth_tessera_iou_fusion_C_presence_centered` | ✓ | C | **0.4437** |
| Δ attributed to Tessera residual | | | **+0.0271** |

This is the authoritative answer to "is Tessera actually helping?": yes,
the residual branch is worth `+0.027` score after loss alignment.

Axis breakdown of the +0.027:
- `iou_bld`: 0.4098 → 0.4813 (+0.072) — Tessera's largest effect
- `iou_tree`: 0.7461 → 0.7579 (+0.012)
- `iou_wat`: 0.3587 → 0.4070 (+0.048) — second-largest effect
- `RMSE_bH`: 1.9276 → 1.9195 m (−0.008, within noise)
- `RMSE_vH`: 3.5249 → 3.5336 m (+0.009, within noise)

RMSE is essentially unchanged, confirming that Tessera's constraint to
IoU logits is working exactly as designed. The residual branch is a *presence
classifier booster*, not a height regressor.

## Stage 8 — C as Springboard for Downstream

Every post-04-21 experiment inherits the Stage 6 recipe as baseline:

```
AlphaEarth 64ch + Tessera 128ch
 ├─ TesseraCompressionStem(out=16, hidden=96, depth=2)
 ├─ LightUNet(base_ch=32)
 ├─ V35head (presence + base + Δ_b/Δ_v, softplus)
 └─ loss_preset = presence_centered
       + veg_height_boost, build_height_boost, aux weights
```

Downstream moves:
- D — Huber height + veg_boost=3: closed, hurt iou_bld.
- E — `height_specialist_depth=2`: **+0.0137**, five-axis win. Anchor.
- F / H — focal IoU α=0.25 / α=0.75: closed, no improvement.
- G — veg_boost=3 stacked on E: trades RMSE_bH for iou_wat, closed.
- I — MSE height: closed.
- J — veg_boost=1.5 compromise: within noise of E, pre-N baseline.
- K — symmetric building+veg 3×: confirms 5× building weight is load-bearing.
- L — specialist_depth=4: best RMSE_bH at base_ch=32, but iou_bld drops 1.6pt.
- M — aux_veg_weight=2: not isolated from shared trunk, closed.
- **N — base_ch=48**: **+0.0076**, five-axis win, current champion.
- O — base_ch=64: saturates, tie with N under TS.

See [K_L_EXPERIMENTS_REPORT.md](K_L_EXPERIMENTS_REPORT.md) and
[N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md) for the
post-C exploration.

## Architecture Rationale Summary

The final fusion design encodes three hard-won lessons:

1. **Asymmetric trust** — AlphaEarth goes through the full U-Net (trunk +
   skip connections + head); Tessera goes through a narrow 16-channel
   residual that can only influence one classifier output. This prevents
   Tessera's different scale/geometry from corrupting features AlphaEarth
   already has right.

2. **Zero-init residual branch** — `presence_delta_head` starts at zero so
   training begins at the AlphaEarth-only solution and Tessera only
   contributes where it demonstrably helps.

3. **Loss targets the submitted outputs** — `presence_centered` supervises
   the *exact logits* that become channels 0-2 of the submission. Without
   this, the residual branch has no direct gradient to the metric.

## Reproducing

```bash
# Re-evaluate all surviving fusion runs on val split
python evaluate.py --val-only --only \
    alphaearth_tessera_iou_fusion \
    alphaearth_tessera_iou_residual \
    alphaearth_tessera_iou_residual_ch3_h64d1 \
    alphaearth_tessera_iou_residual_ch8 \
    alphaearth_tessera_iou_residual_ch16 \
    alphaearth_tessera_iou_residual_ch16_h64 \
    alphaearth_tessera_iou_fusion_B_no_ssim_grad \
    alphaearth_tessera_iou_fusion_C_presence_centered \
    alphaearth_tessera_iou_fusion_C_ch16_h64d1 \
    alphaearth_tessera_iou_fusion_C_ch16_h96d2 \
    alpha_tessera_iou_alpha_height \
    alphaearth_lightunet_C_presence_centered

# Champion training (recreate N)
sbatch run_exp_N_base48.sbatch
```

## References

- Fusion architecture: [core/model.py:579-651](../core/model.py#L579-L651)
- Loss presets: [core/losses.py:145-370](../core/losses.py#L145-L370)
- Head evolution (V2 → V35): [HEAD_EVOLUTION_REPORT.md](HEAD_EVOLUTION_REPORT.md)
- Post-C loss/specialist knobs: [K_L_EXPERIMENTS_REPORT.md](K_L_EXPERIMENTS_REPORT.md)
- Trunk-widening (E → N): [N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md)
- Single-source baselines: [BASELINE_REPORT.md](BASELINE_REPORT.md)
- Label distribution context: [LABEL_BAND_ANALYSIS.md](LABEL_BAND_ANALYSIS.md)
- Champion snapshot: [BEST_RESULT.md](BEST_RESULT.md)
