# AlphaEarth Backbone Experiment Report

**Date**: 2026-04-15
**Scope**: Single-modality backbone comparison on AlphaEarth embeddings, prior to multimodal fusion.

---

## Executive Summary

| Milestone | Score |
|---|---:|
| Old baseline (LightUNet, old head) | 0.7410 |
| Current LightUNet (multi-head / softplus) | 0.7559 (+0.0149) |
| Best backbone — HRNet-W18 | 0.7587 (+0.0028 over current LightUNet) |
| Best ensemble (weighted, no retraining) | 0.7641 (+0.0054 over W18) |
| Ensemble + threshold calibration (proxy) | ~0.7703 |

**Key insight**: the multi-head prediction recipe accounts for ~84% of the gain over the old baseline; backbone choice contributes the remaining ~16%. Post-hoc ensembling yields a larger gain than any single backbone swap.

---

## 1. Experimental Setup

**Input**: AlphaEarth foundation-model embeddings (`64 x 256 x 256`), the strongest single embedding source from the initial baseline study.

**Targets**: `building_fraction`, `vegetation_fraction`, `water_fraction`, `height_m` — all dense, pixel-aligned.

**Data split**: `splits/split.json` — 1619 train / 405 val.

**Training**: 30 epochs, AdamW, AMP, ReduceLROnPlateau (factor 0.5, patience 2), grad clip 1.0.

| Setting | HRNet / Refiner runs | LightUNet ablation |
|---|---|---|
| Batch size | 16 | 32 |
| Learning rate | 1e-4 | 2e-4 |
| Aux weight | 0.05 | 0.25 |

> **Caveat**: the LightUNet ablation shares the current multi-head code path but differs in hyperparameters. It controls for the prediction head factor, not purely for backbone architecture.

## 2. Backbones Tested

| Model | Params | Design Philosophy |
|---|---:|---|
| LightUNet | 2.38M | Lightweight U-Net encoder-decoder |
| EmbeddingRefiner | 1.20M | Full-resolution ConvNeXt + ASPP, no downsampling |
| HRNet-W18 | 3.65M | Multi-resolution parallel branches, width 18 |
| HRNet-W32 | 10.83M | Wider HRNet, width 32 |

**Common design principle**: AlphaEarth is already a dense semantic embedding, not raw imagery. The backbone should preserve pixel-level detail while adding task-specific refinement and context — not re-encode from scratch.

This matters because:
- Land-cover fractions are sub-pixel regression, not hard segmentation.
- Buildings and water are spatially sparse — small structures and boundaries matter.
- Height RMSE is evaluated only on GT building/vegetation pixels, benefiting from both local footprint detail and broader context.

### 2.1 HRNet-W18 / W32

Maintains a high-resolution branch throughout the network, with parallel lower-resolution branches for context. Information is repeatedly exchanged across resolutions.

```
AlphaEarth 64ch → ChannelCalibration → full-res stem
  → Stage 2: 256² + 128²
  → Stage 3: 256² + 128² + 64²
  → Stage 4: 256² + 128² + 64² + 32²
  → upsample all → concatenate → fuse → MultiTaskPredictionHead
```

- High-res branch protects building/water boundaries and sub-pixel fraction detail.
- Low-res branches provide receptive field for height and region-level context.
- W18 (`[18, 36, 72, 144]`) is the current best capacity/regularization tradeoff.
- W32 (`[32, 64, 128, 256]`) has more capacity but appears under-tuned.

### 2.2 EmbeddingRefiner

Takes the opposite approach: assumes AlphaEarth already encodes useful semantics, so performs full-resolution refinement without any downsampling.

```
AlphaEarth 64ch → ChannelCalibration → 1×1 stem
  → full-res ConvNeXt blocks → ASPP (dilated + global pooling)
  → feature fusion → full-res ConvNeXt blocks → MultiTaskPredictionHead
```

- Avoids discarding spatial detail needed for fraction regression.
- Parameter-efficient (1.20M) yet nearly matches LightUNet and HRNet-W18.
- Achieves the best vegetation height RMSE among all tested backbones.

## 3. Prediction Head Design

All backbones use `MultiTaskPredictionHead` producing `[building, vegetation, water, height_norm]`. Height is rescaled by `HEIGHT_NORM_CONSTANT = 30` at inference.

| Output | Activation | Rationale |
|---|---|---|
| Fractions | `sigmoid` → [0, 1] | Valid range for fraction regression |
| Height | `softplus` (nonneg, unbounded) | Replaces old `sigmoid×1.5` that capped at 45 m |
| Presence (aux) | Auxiliary supervision + height gating | **Not** fused into segmentation (avoids prior contamination) |

**Height label stats**: p99 = 24.5 m, p99.9 = 30.3 m, max ~209 m; only 0.0018% of pixels exceed 45 m.

The separated multi-head avoids the old direct 4-channel regression where fractions and height competed in one shallow head. Bounded fractions + nonnegative uncapped height better match the scoring metric (fractions thresholded at 0.5; RMSE only on GT-positive pixels).

## 4. Results

### 4.1 Single-Model Validation

| Rank | Experiment | mIoU bldg | mIoU veg | mIoU water | RMSE bldg ht | RMSE veg ht | **Score** |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | HRNet-W18 | 0.6063 | **0.7427** | 0.6803 | **3.596 m** | 3.948 m | **0.7587** |
| 2 | LightUNet (current) | **0.6110** | 0.7204 | **0.6885** | 3.762 m | 4.035 m | **0.7559** |
| 3 | EmbeddingRefiner | 0.6077 | 0.7297 | 0.6756 | 3.802 m | **3.846 m** | **0.7554** |
| 4 | HRNet-W32 | 0.5998 | 0.7269 | 0.6658 | 4.771 m | 4.051 m | 0.7421 |
| 5 | Old baseline | 0.5981 | 0.7244 | 0.6804 | 4.830 m | 4.330 m | 0.7410 |

**Key observations**:

1. **Most of the gain is from the prediction head**, not the backbone. Current LightUNet alone gains +0.0149 over the old baseline; switching to HRNet-W18 adds only +0.0028 more.
2. **Each backbone has a per-metric strength**: LightUNet leads in building/water mIoU; HRNet-W18 leads in vegetation mIoU and building height RMSE; EmbeddingRefiner leads in vegetation height RMSE.
3. **EmbeddingRefiner is effectively tied with LightUNet** (-0.0005) at half the parameters.
4. **HRNet-W32 is under-tuned** — above the old baseline but clearly below the top three, likely needing lower LR, warmup, lower aux weight, and/or stochastic depth.

### 4.2 Ensemble & Threshold Sweep (No Retraining)

A post-processing test using `tools/sweep_thresholds_and_ensemble.py` on the same 405 val samples. Per-channel weighted blend:

| Channel | LightUNet | Refiner | W18 |
|---|---:|---:|---:|
| Building frac | 0.45 | 0.30 | 0.25 |
| Vegetation frac | 0.20 | 0.35 | 0.45 |
| Water frac | 0.50 | 0.20 | 0.30 |
| Height | 0.15 | 0.35 | 0.50 |

| Ensemble | Score | Delta vs W18 |
|---|---:|---:|
| **weighted_metric_v1** | **0.7641** | **+0.0054** |
| avg (W18 + LightUNet + Refiner) | 0.7630 | +0.0043 |
| avg (W18 + Refiner) | 0.7626 | +0.0039 |

**Threshold calibration** (prediction thresholds varied, GT fixed at 0.5):

```
weighted_metric_v1 + building=0.575, vegetation=0.900, water=0.900
proxy score ≈ 0.7703
```

**Interpretation**:
- Raw ensemble gives a larger gain (+0.0054) than any backbone swap, at zero training cost.
- Per-metric strengths of different backbones are complementary — ensembling captures this.
- High vegetation/water thresholds suggest the models over-predict low-confidence fractions near the default 0.5 cutoff.
- Threshold calibration results are proxy estimates; they must be materialized as actual predictions before being treated as final scores.

## 5. Risks & Caveats

| Risk | Impact |
|---|---|
| Val/leaderboard gap | Geographic/temporal shift may change rankings |
| Impure LightUNet ablation | Different hyperparameters (bs, lr, aux weight) confound the backbone comparison |
| Fixed threshold 0.5 | Per-class calibration likely helps buildings/water |
| Height tail (> 45 m) | Extremely rare (0.0018%); softplus allows it but model may still underpredict without tail-weighted loss |
| No spatial augmentation | Flips and 90-degree rotations not yet applied |
| Incomplete loss logging | Height-boost, presence, and aux-height components not separately reported |

## 6. Next Steps

**Immediate (highest expected impact)**:
1. Materialize `weighted_metric_v1` ensemble predictions through the standard `evaluate.py` path.
2. Convert threshold sweep into real post-processing (calibrate or hard-threshold fraction channels, then re-evaluate at fixed threshold 0.5).
3. Add spatial augmentation: flips + 90-degree rotations (no spectral aug — channels are abstract embeddings).

**Short-term tuning**:
4. Run a strict LightUNet ablation with bs=16, lr=1e-4, aux weight=0.05 to isolate the backbone factor.
5. Tune W32: LR 3e-5, 40-50 epochs, warmup, lower aux weight, stochastic depth.
6. Address height tail: weighted loss > 30 m, oversample tall patches, `log1p` auxiliary target.
7. Add loss diagnostics: log MAE / SSIM / grad / Tversky / height-boost / presence / aux-height separately.

**Downstream (after single-modality stabilizes)**:
8. Multimodal fusion: AlphaEarth + TerraMind S2, AlphaEarth + SAR, or late ensemble of W18 + LightUNet/Refiner.

## 7. Conclusion

Best single-model validation improved from **0.7410 → 0.7587** (+0.0177). Decomposing this gain:

- **+0.0149** from the multi-head / softplus prediction recipe (LightUNet ablation).
- **+0.0028** from switching to HRNet-W18 backbone (with hyperparameter caveat).

The prediction head is the dominant factor. Among backbones, HRNet-W18 is the champion but all three current backbones are within 0.0033 of each other — their per-metric complementarity makes **ensembling the highest-leverage next move**.

The raw weighted ensemble already reaches **0.7641**, and threshold calibration suggests a post-processing ceiling of **~0.7703** on this validation split. Priority before deeper fusion: materialize the ensemble, calibrate thresholds, and run one strict LightUNet rerun plus one more W32 tuning pass.
