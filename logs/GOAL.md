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
| **iou_buildings** | 25% | Segmentation | Higher |
| **iou_trees** | 15% | Segmentation | Higher |
| **iou_water** | 15% | Segmentation | Higher |
| **RMSE_building_height** | 25% | Height regression | Lower |
| **RMSE_vegetation_height** | 20% | Height regression | Lower |

### Metric Definitions (verified 2026-04-17 via dummy-probe submission)

Official definitions were not disclosed in the rules / Data page / starter
notebook. They were reverse-engineered from an all-zero submission on
2026-04-17 — see [METRIC_PROBE_REPORT.md](METRIC_PROBE_REPORT.md) for the
full derivation.

- **IoU_class (positive-only, per-image)**
  For each test patch, binarize the label channel with `label > 0` (ANY
  nonzero fraction counts as positive) and the prediction channel with
  `pred > pred_threshold` (submitter-controlled; 0.5 is conventional).
  Compute `|P ∩ T| / |P ∪ T|` for the positive class only. When **both**
  P and T are empty, IoU = 1.0 (sklearn `zero_division=1.0` convention).
  When exactly one is empty, IoU = 0.0. Final class metric = mean of
  per-image IoU over all test samples.

- **RMSE_class (per-image macro, GT-masked)**
  For each test patch, compute `sqrt(mean((pred_height - gt_height)² on
  pixels where gt_class > 0))`. Skip patches where no gt pixel passes.
  Final metric = mean of per-image RMSE over all remaining samples.
  **Per-image** averaging, NOT global pixel-accumulated.

### Composite Score Formula

```
Score = sum(iou_class * w_class) + sum(max(0, 1 - RMSE_class / X_class) * w_class)
```

- IoU portion is confirmed exactly (matched leaderboard to 4 decimals).
- RMSE portion uses `max(0, 1 - RMSE/X)` per class. `X_class` (normalization
  ceiling) is UNKNOWN but small: X_building < 4m and X_vegetation < 10.9m
  (our dummy's RMSE values clamped both contributions to 0). A follow-up
  probe with nonzero height predictions is needed to pin X down.
- **Common misconception (what we had before)**: it is NOT `mean(IoU_pos,
  IoU_neg)` at threshold 0.5, and RMSE is NOT `/30` global pixel-level.
  Both were plausible defaults that diverged from the actual leaderboard
  metric, which is why pre-2026-04-17 local val scores ran ~0.25-0.35
  higher than the leaderboard.

### Submission Format

- 946 `.npy` files, each shaped `(4, 256, 256)`: channels are [building, vegetation, water, height]
- **Public score**: computed on an undisclosed subset of test patches, for iterative feedback
- **Private score**: computed on the full test set, used for final ranking
