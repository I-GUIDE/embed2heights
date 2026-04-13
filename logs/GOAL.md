# ESA Embed2Heights Challenge — Background Context

## Challenge Objective

Develop a **multi-task model** that takes pre-computed **Geospatial Foundation Model (GFM) embeddings** as input and jointly predicts:

1. **Land cover segmentation** — sub-pixel fractions of buildings, vegetation, and water
2. **Surface height regression** — normalized Digital Surface Model (nDSM), i.e. height above ground in meters

Participants do **not** run the foundation models themselves. Instead, they receive an AI-ready package of pre-computed embeddings from four GFMs and focus entirely on **feature fusion and decoder design**.

---

## Dataset

### Overview

- **Training**: 2,024 patches, 256x256 pixels at 10 m resolution, sampled from major French cities and rural areas
- **Test**: 946 patches (unlabeled, used for competition submission), from different regions and years
- **Labels**: derived from IGN (French National Geographic Institute) airborne LiDAR at 1 m source resolution, aggregated into 10x10 m cells

### Label Format (4-band TIFF per patch)

| Channel | Content | Range |
|---|---|---|
| 0 | Building fraction | 0–100% |
| 1 | Vegetation fraction | 0–100% |
| 2 | Water fraction | 0–100% |
| 3 | nDSM (relative height above ground) | meters |

Labels are **continuous**, not discrete categories — each pixel stores the percentage contribution of each class within the 10x10 m cell.

### Pre-computed Embeddings

| GFM | Type | Channels | Shape | Size |
|---|---|---|---|---|
| **AlphaEarth** | Pixel-aligned | 64 | 256x256 | 33.93 GB |
| **TESSERA** | Pixel-aligned | 128 | 256x256 | 67.82 GB |
| **TerraMind S1** | ViT tokens | 768 | 16x16 | 1.60 GB |
| **TerraMind S2** | ViT tokens | 768 | 16x16 | 1.60 GB |
| **THOR S1** | ViT tokens | 768 | 16x16 | 1.96 GB |
| **THOR S2** | ViT tokens | 768 | 16x16 | 1.95 GB |

- Pixel-aligned embeddings preserve full 256x256 spatial resolution — decoders can use skip connections directly.
- ViT token embeddings are 16x spatially downsampled — decoders must upsample from 16x16 to 256x256.
- Test set contains only S2 variants (no S1).

---

## Scoring Criteria

### Five Metrics with Weights

| Metric | Weight | Task | Better |
|---|---|---|---|
| **mIoU_buildings** | 25% | Segmentation | Higher |
| **mIoU_trees** | 15% | Segmentation | Higher |
| **mIoU_water** | 15% | Segmentation | Higher |
| **RMSE_building_height** | 25% | Height regression | Lower |
| **RMSE_vegetation_height** | 20% | Height regression | Lower |

### Metric Definitions

- **mIoU** = mean(IoU_positive, IoU_negative) for binary segmentation at threshold 0.5. Predictions and labels are both binarized before computing IoU.
- **RMSE** = root mean squared error of predicted height vs. ground truth, computed **only on pixels where the GT class is present** (building pixels for building height, vegetation pixels for vegetation height).

### Composite Score Formula

```
Score = sum(mIoU_i * w_i) + sum((1 - RMSE_i / 30) * w_i)
```

- All terms are normalized to [0, 1]. **Higher is better.**
- 30 meters is the normalization ceiling for RMSE — predictions with RMSE >= 30m contribute 0 to the score.
- Building segmentation (25%) and building height (25%) together account for **50%** of the total score — buildings are the most important class.

### Submission Format

- 946 `.npy` files, each shaped `(4, 256, 256)`: channels are [building, vegetation, water, height]
- **Public score**: computed on an undisclosed subset of test patches, for iterative feedback
- **Private score**: computed on the full test set, used for final ranking
