# Emb2Heights Baselines

Baselines for the **ESA Embed2Heights** competition. Each model predicts a 4-channel output per pixel from pre-computed Geospatial Foundation Model (GFM) embeddings: `[building_fraction, vegetation_fraction, water_fraction, height_m]`.

## Repository Layout

```
emb2heights-backbone/
├── core/
│   ├── dataset.py     # PixelEmbeddingDataset, LatentTokenDataset, id/path helpers
│   ├── model.py       # LightUNet, EmbeddingRefiner, HRNet-W{18,32}, EfficientDecoder256Fast
│   ├── losses.py      # ImprovedCompositeLoss (MAE + SSIM + Gradient + Tversky + aux)
│   └── metrics.py     # Leaderboard metric helpers (WEIGHTS, binary_iou, compute_weighted_score)
├── tools/
│   ├── download_data.py                   # pull the EOTDL dataset
│   ├── generate_split.py                  # write splits/split.json
│   ├── calibrate_thresholds.py            # per-class threshold sweep on a val prediction dir
│   ├── sweep_thresholds_and_ensemble.py   # in-memory ensemble + threshold sweep using evaluate.py's metric
│   ├── create_test_ensemble_submission.py # materialize the weighted_metric_v1 ensemble on the test set
│   ├── make_dummy_submission.py           # build an all-constant .zip submission (metric probing)
│   ├── predict_dummy_metrics.py           # predict leaderboard values under each candidate formula
│   └── *.ipynb                            # ad-hoc analysis notebooks
├── splits/               # saved train/val split JSONs (reproducible seed=42 split)
├── runs/                 # experiment outputs (gitignored): model_best.pth, predictions/, etc.
├── submission/           # zipped test-set predictions ready to upload (gitignored)
├── logs/                 # evaluation / experiment reports
├── train.py              # single-experiment training
├── predict.py            # inference (val or test)
└── evaluate.py           # compute the 5 leaderboard metrics on runs/*/predictions
```

Data is expected one level up from this repo (`../data/train/...`, `../data/test/...`); override with `--train-embeddings-dir`, `--train-targets-dir`, `--test-embeddings-dir`.

## Setup

```bash
conda env create -f environment.yml
conda activate emb2heights
```

## Workflow

### 1. Download data

Requires EOTDL authentication (`eotdl auth login`):

```bash
python tools/download_data.py --path ../data
```

### 2. Generate a reproducible train/val split

The split is keyed by normalized core id so the same patch lives in the same split regardless of embedding source:

```bash
python tools/generate_split.py            # writes splits/{train,val,split}.json, 80/20, seed=42
```

Pass `splits/split.json` to `train.py` via `--split-file` to reuse it.

### 3. Train a single baseline

```bash
python train.py \
    --model-type hrnet_w18 \
    --train-embeddings-dir ../data/train/alphaearth_emb \
    --train-targets-dir    ../data/train/labels \
    --experiment-name      alphaearth_hrnet_w18 \
    --split-file           splits/split.json \
    --epochs 30
```

Artifacts go to `runs/<experiment_name>/`: `model_best.pth`, `model_last.pth`, `loss_curve.png`, `training_params.json`.

There is **no multi-baseline driver script** — launch a shell loop or slurm array to sweep over multiple backbones/sources.

### 4. Predict

Validation (paired with labels, for offline scoring):

```bash
python predict.py \
    --experiment-name      alphaearth_hrnet_w18 \
    --model-type           hrnet_w18 \
    --test-embeddings-dir  ../data/train/alphaearth_emb \
    --test-targets-dir     ../data/train/labels
```

Competition test set (label-free, submission-ready filenames):

```bash
python predict.py \
    --experiment-name      alphaearth_hrnet_w18 \
    --model-type           hrnet_w18 \
    --test-embeddings-dir  ../data/test/alphaearth_test_emb \
    --predictions-dir      submission/alphaearth_hrnet_w18
```

Predictions are `(4, 256, 256)` float32 arrays — channels: `[building%, veg%, water%, height_m]`.

### 5. Evaluate

```bash
python evaluate.py                                # every runs/*/predictions
python evaluate.py --only alphaearth_hrnet_w18    # one experiment
python evaluate.py --val-only                     # restrict to each experiment's own val split
python evaluate.py --pred-threshold 0.3           # non-default prediction binarization
```

The metric formulas were reverse-engineered by the 2026-04-17 dummy-probe submission (see [logs/METRIC_PROBE_REPORT.md](logs/METRIC_PROBE_REPORT.md)) and are shared between `evaluate.py` and the `tools/` scripts via `core/metrics.py`.

### 6. Ensembles & threshold tuning

- `tools/sweep_thresholds_and_ensemble.py` — load every AlphaEarth experiment, evaluate a grid of single-model / weighted / averaged ensembles at several prediction thresholds, print a leaderboard-style table (in-memory only, no files written).
- `tools/calibrate_thresholds.py --pred-dir runs/<exp>/predictions` — pick per-class thresholds on the val split, optionally materialize hard-thresholded predictions.
- `tools/create_test_ensemble_submission.py` — materialize the `weighted_metric_v1` ensemble on the test set (raw + calibrated hard-mask variants), and write an `ensemble_manifest.json`.

## Models

| `--model-type`       | Architecture                                           | Expected input              |
|----------------------|--------------------------------------------------------|-----------------------------|
| `lightunet`          | Light U-Net (32/64/128/256), multi-head output         | Pixel-aligned 256x256       |
| `embedding_refiner`  | Full-resolution ConvNeXt blocks + ASPP, multi-head     | Pixel-aligned 256x256       |
| `hrnet_w18`          | HRNet-style multi-resolution (width 18), multi-head    | Pixel-aligned 256x256       |
| `hrnet_w32`          | Same as above, width 32                                | Pixel-aligned 256x256       |
| `decoder_residual`   | `EfficientDecoder256Fast` — bottleneck + 4× upsample  | ViT tokens 16x16 (768ch)    |
| `auto`               | Pick by input channels (<512 → pixel, else token)     | Any                         |

`decoder` is accepted as an alias for `decoder_residual`. Pixel-aligned backbones (all the AlphaEarth / Tessera options) share a common `MultiTaskPredictionHead` with a fraction head, a fraction-derived presence head, and a fraction-gated softplus height head.

## Loss

`ImprovedCompositeLoss` sums five weighted terms, with `valid_mask` on every term to exclude nodata pixels (all four bands zero) and nDSM holes (nDSM=0 but land cover present).

| Term                 | Default weight | Applied to       |
|----------------------|----------------|------------------|
| MAE (fg/bg split)    | 1.0            | All 4 channels   |
| SSIM                 | 0.5            | Land cover (0–2) |
| Gradient difference  | 0.5            | Land cover (0–2) |
| Tversky (α=0.3, β=0.7) + 5× building-pixel height boost | 2.0 | Land cover + height |
| Auxiliary multi-head supervision | 0.25          | Presence logits + per-class height heads (if backbone supports it) |

## Training configuration (defaults in `train.py`)

| Parameter           | Value                                    |
|---------------------|------------------------------------------|
| Optimizer           | AdamW, lr=2e-4, weight_decay=1e-4        |
| Scheduler           | ReduceLROnPlateau (factor=0.5, patience=2)|
| Gradient clipping   | max_norm=1.0                             |
| Batch size          | 32                                       |
| Epochs              | 30                                       |
| Patch size          | 256                                      |
| Train/val split     | 80/20 (seed=42)                          |
| AMP                 | Enabled on CUDA                          |
| Aux weight          | 0.25                                     |

Most of these are overridable from the CLI.

## Leaderboard metric (verified via 2026-04-17 probe)

| Metric                    | Weight | Definition                                                                                  |
|---------------------------|--------|---------------------------------------------------------------------------------------------|
| `iou_buildings`           | 25%    | Per-image positive-only IoU (`pred > 0.5`, `label > 0`); empty/empty → 1.0; sample-averaged |
| `iou_trees`               | 15%    | same                                                                                         |
| `iou_water`               | 15%    | same                                                                                         |
| `RMSE_building_height`    | 25%    | Per-image RMSE on pixels where building label > 0; sample-averaged                          |
| `RMSE_vegetation_height`  | 20%    | same for vegetation                                                                          |

Composite score: `Σ iou_i × w_i + Σ max(0, 1 − RMSE_i / X_i) × w_i`. `X_class` is bounded by the probe (`X_building < 4m`, `X_vegetation < 10.9m`) but its exact values are still unknown; `compute_weighted_score` uses `30` as a placeholder, which over-estimates the RMSE contribution.

See [logs/METRIC_PROBE_REPORT.md](logs/METRIC_PROBE_REPORT.md) for the full derivation.

## Experiment history

| Report                                                      | Summary                                                                |
|-------------------------------------------------------------|------------------------------------------------------------------------|
| [logs/BASELINE_REPORT.md](logs/BASELINE_REPORT.md)          | Original 6-embedding comparison (**uses pre-probe metric — see banner**)|
| [logs/ALPHAEARTH_BACKBONE_REPORT.md](logs/ALPHAEARTH_BACKBONE_REPORT.md) | LightUNet vs EmbeddingRefiner vs HRNet-W{18,32} on AlphaEarth  |
| [logs/BEST_RESULT.md](logs/BEST_RESULT.md)                  | Current champion (weighted ensemble of W18 + LightUNet + Refiner)      |
| [logs/METRIC_PROBE_REPORT.md](logs/METRIC_PROBE_REPORT.md)  | Metric formula reverse-engineering                                     |
| [logs/LABEL_BAND_ANALYSIS.md](logs/LABEL_BAND_ANALYSIS.md)  | Label distribution, nodata, class imbalance                            |
| [logs/GOAL.md](logs/GOAL.md)                                | Challenge-level background                                             |
