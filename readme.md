# Emb2Heights Baselines

Baseline models for the **ESA Embed2Heights** competition. Predicts 4-channel output per pixel from pre-computed Geospatial Foundation Model (GFM) embeddings:
`[% Building, % Vegetation, % Water, Height (m)]`.

## Data Layout

```
esa/
├── data/
│   ├── train/                          # 2,024 labeled patches (256x256 @ 10m)
│   │   ├── alphaearth_emb/             # 64ch pixel-aligned
│   │   ├── tessera_emb/                # 128ch pixel-aligned
│   │   ├── terramind_s1_emb/           # 768ch ViT tokens (16x16)
│   │   ├── terramind_s2_emb/           # 768ch ViT tokens (16x16)
│   │   ├── thor_s1_emb/                # 768ch ViT tokens (16x16)
│   │   ├── thor_s2_emb/                # 768ch ViT tokens (16x16)
│   │   └── labels/                     # 4ch ground truth
│   └── test/                           # 946 unlabeled patches
│       ├── alphaearth_test_emb/
│       ├── tessera_test_emb/
│       ├── terramind_test_s2_emb/
│       └── thor_test_s2_emb/
└── emb2heights-baselines/              # this repo
    ├── core/
    │   ├── dataset.py                  # PixelEmbeddingDataset, LatentTokenDataset
    │   ├── model.py                    # LightUNet, EfficientDecoder256
    │   └── losses.py                   # ImprovedCompositeLoss
    ├── tools/
    │   ├── generate_split.py           # create reproducible train/val splits
    │   └── download_data.py            # download dataset from EOTDL
    ├── train.py                        # single-baseline training
    ├── predict.py                      # inference (val or test)
    ├── evaluate.py                     # compute 5 leaderboard metrics
    ├── run_all_baselines.py            # batch train/predict all 4 baselines
    ├── splits/                         # saved train/val split JSONs
    ├── runs/                           # experiment checkpoints & outputs
    ├── submission/                     # competition submission files
    └── logs/                           # evaluation reports
```

## Setup

```bash
conda env create -f environment.yml
conda activate emb2heights
```

## Quick Start

### 1. Download data

Requires EOTDL authentication (`eotdl auth login`):

```bash
python tools/download_data.py --path ../data
```

### 2. Generate train/val split

Split labeled data into reproducible train and validation sets (default 80/20, seed=42):

```bash
python tools/generate_split.py
# Creates splits/train.json, splits/val.json, splits/split.json
```

### 3. Train all baselines

```bash
python run_all_baselines.py --skip-predict
```

Or train a single baseline:

```bash
python train.py \
    --model-type lightunet \
    --train-embeddings-dir ../data/train/alphaearth_emb \
    --train-targets-dir ../data/train/labels \
    --experiment-name alphaearth_run01 \
    --split-file splits/split.json \
    --epochs 30
```

### 4. Evaluate on validation set

```bash
# Predict on training data (val split)
python run_all_baselines.py --skip-train

# Evaluate
python evaluate.py --val-only
```

### 5. Generate competition submission

Predict on the test set (no labels required):

```bash
python run_all_baselines.py --skip-train --predict-test
```

Or for a single baseline:

```bash
python predict.py \
    --experiment-name alphaearth_baseline \
    --model-type lightunet \
    --test-embeddings-dir ../data/test/alphaearth_test_emb \
    --predictions-dir submission/alphaearth
```

Predictions are saved as `pred_<core_id>.npy` files with shape `(4, 256, 256)`:
- Channel 0: Building coverage (0-1)
- Channel 1: Vegetation coverage (0-1)
- Channel 2: Water coverage (0-1)
- Channel 3: Height in meters

## Model Architectures

| `--model-type` | Architecture | Input Type | Used for |
|---|---|---|---|
| `lightunet` | LightUNet (encoder-decoder with skip connections) | Pixel-aligned (256x256) | AlphaEarth, Tessera |
| `embedding_refiner` | Full-resolution ConvNeXt/ASPP refiner with multi-head prediction | Pixel-aligned (256x256) | AlphaEarth, Tessera |
| `hrnet_w18` | HRNet-style high-resolution backbone, width 18, with multi-head prediction | Pixel-aligned (256x256) | AlphaEarth, Tessera |
| `hrnet_w32` | HRNet-style high-resolution backbone, width 32, with multi-head prediction | Pixel-aligned (256x256) | AlphaEarth, Tessera |
| `decoder_residual` | EfficientDecoder256 (progressive upsampling) | ViT tokens (16x16) | TerraMind, THOR |
| `auto` | Auto-select based on input spatial size | Any | Default |

AlphaEarth single-modality backbone comparison:

```bash
python train.py --model-type embedding_refiner --experiment-name alphaearth_refiner --split-file splits/split.json --batch-size 8 --grad-accum-steps 4 --lr 1e-4 --aux-weight 0.05 --num-workers 0
python train.py --model-type hrnet_w18 --experiment-name alphaearth_hrnet_w18 --split-file splits/split.json --batch-size 4 --grad-accum-steps 4 --lr 1e-4 --aux-weight 0.05 --num-workers 0
python train.py --model-type hrnet_w32 --experiment-name alphaearth_hrnet_w32 --split-file splits/split.json --batch-size 2 --grad-accum-steps 8 --lr 5e-5 --aux-weight 0.05 --num-workers 0
```

## Loss Function

`ImprovedCompositeLoss` with 4 weighted terms:
- **MAE** (w=1.0): pixel-level regression with foreground/background split
- **SSIM** (w=0.5): structural similarity on land-cover channels
- **Gradient** (w=0.5): edge sharpness penalty on land-cover channels
- **Tversky** (w=2.0): asymmetric segmentation loss (alpha=0.3, beta=0.7) + 2x height boosting on building pixels
- **Auxiliary multi-head loss**: for compatible models, lightly supervises class-presence logits and building/vegetation height heads before they are fused into the final 4-channel prediction

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=2e-4, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Gradient Clipping | max_norm=1.0 |
| Batch Size | 32 |
| Epochs | 30 |
| Train/Val Split | 80/20 (seed=42) |

## Evaluation Metrics

5 leaderboard metrics with weights:

| Metric | Weight | Task |
|---|---|---|
| mIoU Buildings | 25% | Segmentation |
| mIoU Trees | 15% | Segmentation |
| mIoU Water | 15% | Segmentation |
| RMSE Building Height | 25% | Height regression |
| RMSE Vegetation Height | 20% | Height regression |

Composite score: `sum(mIoU_i * w_i) + sum((1 - RMSE_i / 30) * w_i)` — higher is better.

## Baseline Results (Validation, 405 samples)

| # | Baseline | mIoU_bld | mIoU_tree | mIoU_wat | RMSE_bH | RMSE_vH | Score |
|---|---|---|---|---|---|---|---|
| 1 | **AlphaEarth** | 0.598 | 0.724 | 0.680 | 4.83m | 4.33m | **0.741** |
| 2 | TerraMind S2 | 0.531 | 0.478 | 0.545 | 5.83m | 6.95m | 0.641 |
| 3 | TerraMind S1 | 0.524 | 0.436 | 0.452 | 5.13m | 7.36m | 0.622 |
| 4 | THOR S1 | 0.519 | 0.411 | 0.497 | 5.66m | 7.53m | 0.619 |
| 5 | THOR S2 | 0.517 | 0.365 | 0.462 | 5.99m | 7.65m | 0.602 |
| 6 | Tessera | 0.507 | 0.206 | 0.553 | 7.21m | 11.93m | 0.551 |

See [logs/BASELINE_REPORT.md](logs/BASELINE_REPORT.md) for detailed analysis.
