# Emb2Heights Baselines

Baselines for the **ESA Embed2Heights** competition. Each model predicts a 4-channel output per pixel from pre-computed Geospatial Foundation Model (GFM) embeddings: `[building_fraction, vegetation_fraction, water_fraction, height_m]`.

## Repository Layout

```
emb2heights-backbone/
├── core/
│   ├── data/          # data discovery, datasets, and training loader assembly
│   ├── config.py      # YAML recipe and CLI config helpers
│   ├── engine/        # device, seed, checkpoint, and train-loop helpers
│   ├── inference/     # prediction/TTA, ensemble, calibration, transforms, submission helpers
│   ├── metrics.py     # leaderboard metric implementation
│   ├── models/        # model package: blocks, backbones, heads, fusion, registry, factory
│   ├── losses/        # supervised loss primitives and ImprovedCompositeLoss
│   └── pretrain/      # self-supervised pretraining data, masks, model, losses
├── tools/
│   ├── download_data.py                   # download/resume the EOTDL dataset
│   ├── generate_split.py                  # write splits/split.json
│   ├── ensemble.py                        # materialize mean or weighted prediction ensembles
│   ├── sweep_thresholds.py                # per-class threshold sweep on a val prediction dir
│   ├── make_submission.py                 # integrated test ensemble + threshold + zip pipeline
│   ├── make_dummy_submission.py           # build an all-constant .zip submission (metric probing)
│   └── *.ipynb                            # ad-hoc analysis notebooks
├── splits/               # saved train/val split JSONs (reproducible seed=42 split)
├── runs/                 # experiment outputs (gitignored): active/key runs plus history/ archive
├── submission/           # zipped test-set predictions ready to upload (gitignored)
├── logs/                 # evaluation / experiment reports; start with logs/EXPERIMENT_INVENTORY.md, logs/EXPERIMENT_EVOLUTION.md, logs/HISTORY_MOVE_MANIFEST.md, logs/JOB_ARCHIVE_MANIFEST.md, and logs/MODEL_INTERFACE_CLEANUP_REVIEW.md
├── pretrain.py           # CLI entrypoint for self-supervised AlphaEarth+Tessera pretraining
├── train.py              # single-experiment training
├── predict.py            # inference (val or test)
└── evaluate.py           # compute the 5 leaderboard metrics on runs/*/predictions
```

Data is expected one level up from this repo (`../data/train/...`, `../data/test/...`); override with config or CLI paths when needed.
`configs/defaults.yml` holds the global training defaults. New
competition experiments should start from one of the three active recipes in
`configs/active/`.

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

Resume an interrupted download from the local catalog:

```bash
python tools/download_data.py resume --path ../data --workers 8
```

### 2. Generate a reproducible train/val split

The split is keyed by normalized core id so the same patch lives in the same split regardless of embedding source:

```bash
python tools/generate_split.py            # writes splits/{train,val,split}.json, 80/20, seed=42
```

Pass `splits/split.json` to `train.py` via `--split-file` to reuse it.

### 3. Train from an active strategy

New experiments are constrained to three strategy families. Fork one recipe and
change only the field you are actively testing.

```bash
python train.py --config configs/active/ae_tessera_gated.yml
python train.py --config configs/active/xfusion_crosslevel.yml
python train.py --config configs/active/ae_only_baseline.yml
```

Each run writes a merged `resolved_config.yml`, execution metadata, and a
compact metrics summary under `runs/<experiment_name>/`.

Artifacts go to `runs/<experiment_name>/`: `model_best.pth`, `model_last.pth`,
`loss_curve.png`, `loss_history.jsonl`, `resolved_config.yml`,
`run_metadata.json`, `metrics_summary.json`.

There is **no multi-baseline driver script** — launch a shell loop or slurm array to sweep over multiple backbones/sources.

### 4. Predict

Validation for an active AE+Tessera run:

```bash
python predict.py \
    --experiment-name ae_tessera_gated_v001 \
    --model-type ae_tessera_gated \
    --test-embeddings-dir ../data/train/alphaearth_emb \
    --secondary-test-embeddings-dir ../data/train/tessera_emb \
    --test-targets-dir ../data/train/labels
```

Competition test set for the same strategy:

```bash
python predict.py \
    --experiment-name ae_tessera_gated_v001 \
    --model-type ae_tessera_gated \
    --test-embeddings-dir ../data/test/alphaearth_test_emb \
    --secondary-test-embeddings-dir ../data/test/tessera_test_emb \
    --predictions-dir submission/ae_tessera_gated_v001
```

Three-modal XFusion additionally needs TerraMind-S2 token embeddings:

```bash
python predict.py \
    --experiment-name xfusion_crosslevel_v001 \
    --model-type xfusion_crosslevel \
    --test-embeddings-dir ../data/train/alphaearth_emb \
    --secondary-test-embeddings-dir ../data/train/tessera_emb \
    --token-test-embeddings-dir ../data/train/terramind_s2_emb \
    --test-targets-dir ../data/train/labels
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

- `tools/ensemble.py` — materialize a mean or weighted ensemble prediction directory.
- `tools/sweep_thresholds.py --pred-dir runs/<exp>/predictions` — pick per-class thresholds on a labeled validation prediction directory; add `--water-k-grid 0,4,8,12,16,24,32` to also tune the water connected-component filter used by `make_submission.py --water-cc-min-size`.
- `tools/make_submission.py` — integrated test pipeline: optionally ensemble test predictions, sweep thresholds from validation predictions, save binarized test predictions, and create the required zip.

Example integrated submission:

```bash
python tools/make_submission.py \
    --ensemble-inputs runs/a/test_predictions runs/b/test_predictions \
    --ensemble-output-dir runs/ens/test_predictions \
    --sweep-ensemble-inputs runs/a/predictions runs/b/predictions \
    --split-file splits/split.json \
    --binarized-output-dir runs/ens/test_predictions_binary \
    --threshold-report runs/ens/thresholds.json \
    --output runs/ens/submission.zip
```

## Active Strategies

| Strategy recipe | Purpose | Internal model type |
|---|---|---|
| `active/ae_tessera_gated.yml` | Main two-modal AlphaEarth + Tessera line | `ae_tessera_gated` |
| `active/xfusion_crosslevel.yml` | Main three-modal TerraMind-S2 + AlphaEarth + Tessera line | `xfusion_crosslevel` |
| `active/ae_only_baseline.yml` | Fallback and sanity-check line | `ae_only` |

Live model code is intentionally limited to these active families. Very old
runs are preserved through `logs/`, run metadata, and archived artifacts rather
than live architecture branches.

Rule for new work: every run should be a small edit to one of the three active
YAML recipes. If a change does not clearly belong to one of those families,
document why before launching it.

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

Composite score: `Σ iou_i × w_i + Σ max(0, 1 − RMSE_i / X_i) × w_i`, with `X_building = 3.0m` and `X_vegetation = 5.0m`.

See [logs/METRIC_PROBE_REPORT.md](logs/METRIC_PROBE_REPORT.md) for the full derivation.
