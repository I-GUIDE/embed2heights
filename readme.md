# Emb2Heights Baselines

Baselines for the **ESA Embed2Heights** competition. Each model predicts a 4-channel output `[building_fraction, veg_fraction, water_fraction, height_m]` from pre-computed GFM embeddings.

## Repo Layout

```
├── core/
│   ├── data/          # datasets and data loading
│   ├── models/        # blocks, backbones, heads, fusion, registry, factory
│   ├── losses/        # ImprovedCompositeLoss and primitives
│   ├── inference/     # prediction, TTA, ensemble, calibration, submission helpers
│   ├── engine/        # device, seed, checkpoint, train-loop helpers
│   ├── metrics.py     # leaderboard metric implementation
│   └── pretrain/      # self-supervised pretraining
├── tools/
│   ├── ensemble.py              # mean or weighted prediction ensemble
│   ├── sweep_thresholds.py      # per-class threshold sweep on val predictions
│   ├── binarize_ensemble.py     # binarize continuous predictions at swept thresholds
│   └── make_dummy_submission.py # all-constant .zip for metric probing
├── configs/active/   # canonical YAML recipes for all active strategies
├── splits/           # reproducible train/val split JSONs (seed=42)
├── runs/             # experiment outputs (gitignored)
├── train.py          # training entrypoint
├── predict.py        # inference (val or test)
└── evaluate.py       # compute leaderboard metrics on runs/*/predictions
```

Data lives at `../data/train/` and `../data/test/` relative to this repo.

> **Note:** `--config` YAML loading is not yet wired into `train.py`. Use CLI flags directly (or the SLURM scripts). The YAMLs in `configs/active/` are the canonical parameter reference for each recipe.

## Setup

```bash
conda env create -f environment.yml
conda activate emb2heights
```

## Workflow

### 1. Download data

```bash
python tools/download_data.py --path ../data          # requires: eotdl auth login
python tools/download_data.py resume --path ../data   # resume interrupted download
```

### 2. Generate a reproducible split

```bash
python tools/generate_split.py   # writes splits/{train,val,split}.json  (80/20, seed=42)
```

5-fold group-stratified splits already exist at `splits/group_code_5fold_seed42/fold_{0..4}/split.json`.

### 3. Train

All new experiments fork one of the three active recipes and change only the field under test.

```bash
# Champion recipe — 5-fold cross-validation
sbatch run_uw_gated_F_5fold.bash      # trains folds 0-4 in parallel (SLURM array)
```

Artifacts go to `runs/<experiment_name>/`: `model_best.pth`, `model_last.pth`, `loss_curve.png`, `loss_history.jsonl`, `training_params.json`.

### 4. Predict

Validation (paired mode, produces labels for metric computation):

```bash
python predict.py \
    --experiment-name ae_tessera_gated_v001 \
    --model-type ae_tessera_gated \
    --test-embeddings-dir ../data/train/alphaearth_emb \
    --secondary-test-embeddings-dir ../data/train/tessera_emb \
    --test-targets-dir ../data/train/labels
```

Test set (label-free, submission filenames):

```bash
python predict.py \
    --experiment-name ae_tessera_gated_v001 \
    --model-type ae_tessera_gated \
    --test-embeddings-dir ../data/test/alphaearth_test_emb \
    --secondary-test-embeddings-dir ../data/test/tessera_test_emb \
    --predictions-dir submission/ae_tessera_gated_v001
```

Three-modal XFusion additionally needs `--token-test-embeddings-dir ../data/train/terramind_s2_emb`.

Predictions are `(4, 256, 256)` float32 arrays.

### 5. Evaluate

```bash
python evaluate.py                                   # all runs/*/predictions
python evaluate.py --only ae_tessera_gated_v001      # single experiment
python evaluate.py --val-only                        # restrict to each run's own val split
```

### 6. Build a submission

The leaderboard requires **binarized** predictions (ch0-2 as `{0, 1}`, ch3 continuous height).
Binarization at the right thresholds is what compresses the zip from ~879 MB → ~221 MB.

```bash
# Full 5-fold submission pipeline (predict test + sweep thresholds + binarize + zip)
sbatch run_uw_gated_F_submit.bash
```

Or manually:

```bash
# 1. Ensemble test predictions
python tools/ensemble.py mean \
    --inputs runs/fold0/test_preds runs/fold1/test_preds ... \
    --output-dir runs/ens/test_preds

# 2. Sweep thresholds on OOF val predictions
python tools/sweep_thresholds.py \
    --pred-dir runs/fold0/predictions \
    --labels-dir ../data/train/labels \
    --split-file splits/group_code_5fold_seed42/fold_0/split.json

# 3. Binarize at swept thresholds
python tools/binarize_ensemble.py \
    --input-dir  runs/ens/test_preds \
    --output-dir runs/ens/test_preds_bin \
    --thresholds 0.620 0.575 0.875

# 4. Zip
cd runs && zip -r -q submission.zip ens/test_preds_bin/
```

## Active Strategies

| Recipe | Purpose | Model type |
|---|---|---|
| `active/uw_gated_F.yml` | **Champion** — simple GMU, 5-fold mean 0.4999, LB ~0.48 | `ae_tessera_gated` |
| `active/ae_tessera_gated.yml` | Two-modal AlphaEarth + Tessera base config | `ae_tessera_gated` |
| `active/xfusion_crosslevel.yml` | Three-modal TerraMind-S2 + AlphaEarth + Tessera | `xfusion_crosslevel` |
| `active/ae_only_baseline.yml` | Fallback / sanity check | `ae_only` |

Rule: every run is a small edit to one of these recipes.

## Loss (`ImprovedCompositeLoss`)

All terms use `valid_mask` to exclude nodata pixels (all-zero bands) and nDSM holes.

| Term | Applied to |
|---|---|
| MAE (fg/bg split) | All 4 channels |
| Tversky (α=0.3, β=0.7) + building/veg pixel height boost | Land cover + height |
| Auxiliary multi-head supervision | Presence logits + per-class height heads |

The `presence_centered` loss preset (champion) emphasizes presence/IoU terms and down-weights fraction MAE.

## Leaderboard Metric

| Metric | Weight |
|---|---|
| `iou_buildings` | 25% |
| `iou_trees` | 15% |
| `iou_water` | 15% |
| `RMSE_building_height` | 25% |
| `RMSE_vegetation_height` | 20% |

Composite: `Σ iou_i × w_i + Σ max(0, 1 − RMSE_i / X_i) × w_i`, with `X_building = 3.0 m`, `X_vegetation = 5.0 m`.

**Calibration note:** the public leaderboard tracks your *worst-fold* CV score, not the 5-fold mean. Expect LB ≈ worst-fold val.
