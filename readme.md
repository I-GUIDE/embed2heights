# Emb2Heights Baselines

Baseline models for the **Emb2Heights** competition. Predicts 4-channel output per pixel from GFM embeddings:
`[% Building, % Vegetation, % Water, Height (m)]`.

## Data Layout

```
esa/
├── data/
│   ├── train/                          # 2,024 labeled patches (256x256 @ 10m)
│   │   ├── alphaearth_emb/             # 64ch pixel-level
│   │   ├── tessera_emb/                # 128ch pixel-level
│   │   ├── terramind_s2_emb/           # 768ch patch-level (16x16)
│   │   ├── thor_s2_emb/                # 768ch patch-level (16x16)
│   │   └── labels/                     # 4ch ground truth
│   └── test/                           # 946 unlabeled patches (competition submission)
│       ├── alphaearth_test_emb/
│       ├── tessera_test_emb/
│       ├── terramind_test_s2_emb/
│       └── thor_test_s2_emb/
└── emb2heights-baselines/              # this repo
    ├── core/
    │   ├── dataset.py
    │   ├── model.py
    │   └── losses.py
    ├── train.py
    ├── predict.py
    ├── evaluate.py
    ├── generate_split.py
    ├── run_all_baselines.py
    ├── splits/                         # generated train/val split files
    └── runs/                           # experiment outputs
```

## Setup

```bash
conda env create -f environment.yml
conda activate emb2heights
```

## Quick Start

### 1. Generate train/val split

Split labeled data into independent train and validation sets (default 80/20):

```bash
python generate_split.py
# Creates splits/train.json, splits/val.json, splits/split.json
```

### 2. Train all baselines

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

### 3. Evaluate on validation set

```bash
# Predict on training data (val split)
python run_all_baselines.py --skip-train

# Evaluate
python evaluate.py --val-only
```

### 4. Generate competition submission

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

| `--model-type` | Architecture | Used for |
|---|---|---|
| `lightunet` | U-Net with skip connections | Pixel-level embeddings (AlphaEarth, Tessera) |
| `decoder_residual` | Progressive upsampling decoder | Patch-level embeddings (TerraMind, THOR) |
| `auto` | Auto-select by input channels | Default |

## Loss Function

`ImprovedCompositeLoss` with 4 weighted terms:
- **MAE** (w=1.0): foreground/background split regression
- **SSIM** (w=0.5): structural similarity on land-cover channels
- **Gradient** (w=0.5): edge sharpness on land-cover channels
- **Tversky** (w=2.0): asymmetric loss (alpha=0.3, beta=0.7) + height boosting on building pixels

## Evaluation Metrics

5 leaderboard metrics with weights:
- mIoU Buildings (25%)
- mIoU Trees (15%)
- mIoU Water (15%)
- RMSE Building Height (25%)
- RMSE Vegetation Height (20%)
