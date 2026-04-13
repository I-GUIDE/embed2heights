# Emb2Heights Baseline Evaluation Report

**Date**: 2026-04-12
**Task**: Predict sub-pixel land cover (Building, Vegetation, Water) and surface height (nDSM) from GFM embeddings.
**Dataset**: 2024 patches, 256x256 @ 10m resolution (France), 4-band labels `[building%, vegetation%, water%, height_m]`.

---

## 1. Baseline Overview

Four baselines were evaluated, each using the **same training pipeline** but with embeddings from a different Geospatial Foundation Model (GFM). The only differences are the input embedding and the model architecture matched to its tensor shape.

### 1.1 Embedding Characteristics

| Baseline | GFM | Channels | Spatial Size | Tensor Type |
|---|---|---|---|---|
| AlphaEarth | AlphaEarth | 64 | 256 x 256 | Pixel-aligned |
| Tessera | Tessera | 128 | 256 x 256 | Pixel-aligned |
| TerraMind S2 | TerraMind (Sentinel-2) | 768 | 16 x 16 | ViT tokens |
| THOR S2 | THOR (Sentinel-2) | 768 | 16 x 16 | ViT tokens |

### 1.2 Model Architecture

The model is selected based on the embedding spatial structure:

- **Pixel-aligned embeddings** (AlphaEarth, Tessera): `LightUNet` — a lightweight encoder-decoder with skip connections. Input and output are at the same 256x256 resolution. Data is loaded via `PixelEmbeddingDataset`.

- **ViT token embeddings** (TerraMind, THOR): `EfficientDecoder256` — a bottleneck + 4-stage transposed-convolution decoder with residual blocks that upsamples 16x16 tokens to 256x256 output. Data is loaded via `LatentTokenDataset` (scale_factor=16).

### 1.3 Training Configuration (shared across all baselines)

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=2e-4, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Gradient Clipping | max_norm=1.0 |
| Loss Function | ImprovedCompositeLoss (4-term weighted sum) |
| Epochs | 30 |
| Batch Size | 32 |
| Patch Size | 256 |
| Train/Val Split | 80% / 20% (random, seed=42) |

### 1.4 Loss Function

`ImprovedCompositeLoss` combines four terms with weights `[1.0, 0.5, 0.5, 2.0]`:

1. **MAE** (weight 1.0): Pixel-level regression with foreground/background split.
2. **SSIM Loss** (weight 0.5): Enforces structural similarity and sharp boundaries on land-cover channels.
3. **Gradient Loss** (weight 0.5): Penalizes blurred edges in predictions.
4. **Tversky Loss** (weight 2.0): Asymmetric segmentation loss (alpha=0.3, beta=0.7) that penalizes false negatives heavily, designed to capture sparse building footprints. Includes a structure-boosted height component where building-pixel height errors are penalized 2x more.

---

## 2. Evaluation Metrics

The leaderboard score is a weighted mean of five metrics:

| Metric | Weight | Measures |
|---|---|---|
| mIoU_buildings | 25% | Binary segmentation quality for buildings (threshold=0.5) |
| mIoU_trees | 15% | Binary segmentation quality for vegetation |
| mIoU_water | 15% | Binary segmentation quality for water |
| RMSE_building_height | 25% | Height prediction accuracy on building pixels (meters) |
| RMSE_vegetation_height | 20% | Height prediction accuracy on vegetation pixels (meters) |

- **mIoU**: Mean of IoU_positive and IoU_negative for binary segmentation (higher is better).
- **RMSE**: Root mean squared error of height predictions conditioned on GT class presence (lower is better).
- **Composite Score**: `sum(mIoU_i * w_i) + sum((1 - RMSE_i / 30) * w_i)` — all terms normalized to [0, 1], higher is better.

---

## 3. Results

### 3.1 Validation Set (20%, 405 samples, seed=42)

Evaluated on the held-out validation split only — the same split used during training, reproduced deterministically. These scores reflect generalization performance without data leakage.

| # | Baseline | mIoU_bld | mIoU_tree | mIoU_wat | RMSE_bH (m) | RMSE_vH (m) | Score |
|---|---|---|---|---|---|---|---|
| 1 | **AlphaEarth** | **0.5981** | **0.7244** | **0.6804** | **4.83** | **4.33** | **0.741** |
| 2 | TerraMind S2 | 0.5309 | 0.4777 | 0.5448 | 5.83 | 6.95 | 0.641 |
| 3 | THOR S2 | 0.5171 | 0.3647 | 0.4622 | 5.99 | 7.65 | 0.602 |
| 4 | Tessera | 0.5065 | 0.2056 | 0.5528 | 7.21 | 11.93 | 0.551 |

### 3.2 Full Training Set (2024 samples, for reference)

Evaluated on all samples including the 80% training split. Scores are slightly inflated due to data leakage.

| # | Baseline | mIoU_bld | mIoU_tree | mIoU_wat | RMSE_bH (m) | RMSE_vH (m) | Score |
|---|---|---|---|---|---|---|---|
| 1 | **AlphaEarth** | **0.6042** | **0.7164** | **0.6786** | **4.57** | **4.20** | **0.744** |
| 2 | TerraMind S2 | 0.5351 | 0.4837 | 0.5367 | 5.51 | 6.55 | 0.647 |
| 3 | THOR S2 | 0.5177 | 0.3688 | 0.4529 | 5.73 | 7.22 | 0.607 |
| 4 | Tessera | 0.5143 | 0.2088 | 0.5461 | 6.85 | 11.50 | 0.558 |

> **Overfitting is mild**: val-only scores are only 0.003–0.007 lower than full-set scores. Rankings are identical across both evaluations.

---

## 4. Analysis

### 4.1 AlphaEarth leads across all metrics

AlphaEarth achieves the best score on every single metric. Its pixel-aligned 64-channel embedding at native 256x256 resolution preserves fine spatial detail that the LightUNet decoder can directly exploit via skip connections. No upsampling is needed, so there is no information loss from spatial compression.

### 4.2 ViT token embeddings lose spatial detail

TerraMind S2 and THOR S2 compress the scene into 768-channel tokens on a 16x16 grid (16x spatial downsampling). The decoder must reconstruct 256x256 predictions from this coarse representation, which fundamentally limits boundary sharpness and small-object detection. This is reflected in their lower mIoU scores, particularly for sparse classes (buildings, water).

### 4.3 Tessera performs worst despite moderate channel count

Tessera's 128-channel pixel-aligned embedding underperforms AlphaEarth's 64-channel embedding across all metrics. The vegetation metrics are particularly poor (mIoU_tree=0.21, RMSE_vH=11.5m), suggesting that Tessera's pre-training objective does not encode vegetation-relevant spectral/structural information as effectively.

### 4.4 Height prediction remains challenging

Even the best baseline (AlphaEarth) has a building height RMSE of 4.57m — roughly one story of error. This suggests that:
- The baseline decoder architectures are too simple to capture the height regression task well.
- The loss function, while composite, may benefit from further tuning of the height-specific terms.
- More sophisticated architectures (attention mechanisms, multi-scale fusion, ensemble of embeddings) could improve results.

### 4.5 Vegetation height is harder than building height

Across all baselines, RMSE_vegetation_height is consistently higher than RMSE_building_height. Vegetation height is inherently more variable and diffuse compared to building height (which tends to be uniform within a footprint), making it a harder regression target.

---

## 5. Possible Improvements

1. **Ensemble multiple embeddings** — Fuse AlphaEarth + TerraMind S2 (or all four) as multi-source input to leverage complementary information.
2. **Deeper/wider decoders** — Add attention layers, FPN-style multi-scale fusion, or transformer decoder heads.
3. **Height-specific loss tuning** — Increase the weight on height regression or use separate loss heads for land cover vs. height.
4. **Data augmentation** — The current pipeline uses random crops only. Since the inputs are pre-computed embeddings (not raw imagery), only spatial transforms (flips, 90-degree rotations) are valid — they must be applied identically to the embedding and label tensors. Spectral/color augmentations (jitter, channel dropout) are **not applicable** because embedding channels are abstract latent features, not spectral bands.
5. **Proper validation** — Use the saved train/val split file to evaluate only on the 20% held-out validation set for honest performance estimates.
6. **Post-processing** — Apply CRF or morphological operations to sharpen building boundaries in predictions.
