<div align="center">

# 🏔️ Embed2Heights: Reaching new heights with GeoFM

### Multi-Embedding Fusion for Presence Segmentation & Height Regression

**Team `Attention_Plzzz`** — Dingqi Ye · Daniel Kiv · Wen Zhou · Wei Hu · Ayush Khot

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900?logo=nvidia&logoColor=white)
[![Solution Deck](https://img.shields.io/badge/Solution_Deck-PDF-B31B1B?logo=latex&logoColor=white)](docs/framework_overview.pdf)

</div>

---

> 📑 **Start with the [Solution Deck](docs/framework_overview.pdf)** — the full method in slides.
> This README focuses on **reproducing** the submission.

---

## 📌 Table of Contents

- [🚀 TL;DR — Quick Start](#-tldr--quick-start)
- [🏗️ Architecture](#️-architecture)
- [🎯 What the Final Submission Is](#-what-the-final-submission-is)
- [🔁 Reproduce: Step by Step](#-reproduce-step-by-step)
  - [1️⃣ Environment](#1️⃣-environment)
  - [2️⃣ Data](#2️⃣-data)
  - [3️⃣ Generate the delmask masks](#3️⃣-generate-the-delmask-masks-before-training)
  - [4️⃣ Train + Predict + Assemble](#4️⃣-train--predict--assemble)
  - [5️⃣ Evaluate the OOF folds](#5️⃣-evaluate-the-oof-folds-optional)
- [🗂️ Repository Layout](#️-repository-layout)

---

## 🚀 TL;DR — Quick Start

```bash
conda env create -f environment.yml && conda activate emb2heights   # 1. env
# 2. put the challenge data under data/ (or export DATA_ROOT=...)
python tools/generate_missing_masks.py                              # 3. delmask masks
scripts/run_all.sh                                                  # 4. everything -> submission/FINAL_*.zip
```

> 💡 `run_all.sh` is **resumable** and **subsettable** — smoke-test a single
> member-fold with `MEMBERS="0" FOLDS="0" scripts/run_all.sh`.

---

## 🏗️ Architecture

![Architecture](docs/architecture.png)

Four blocks (vector version in [`docs/architecture.pdf`](docs/architecture.pdf)):

| | Block | What it does |
|---|---|---|
| 🧱 | **Dense pixel backbone** | AlphaEarth (`64×256²`) and Tessera (`128×256²`) each go through a **U-Net++** (the primary backbone; the ensemble also swaps in UNet 3+ / TransUNet variants); a learned **spatial gate** fuses them into `F_pixel`. |
| 🎛️ | **Coarse token conditioning** | The 4 token sources (TerraMind / Thor, S1/S2, each `768×16²`) are projected to ctx 96 and exchange information via zero-init **cross-source self-attention**. |
| 🔀 | **Fusion** | Each conditioned token source modulates `F_pixel` via zero-init **FiLM + additive + gate** (×4, one per source, upsampled `16²→256²`): `F_out = F_pixel + Σ δᵢ` — tokens *condition* the pixel body, never replace it. |
| 🔱 | **Split-trunk multi-task head** | Separate seg / height trunks feed the presence heads (ch 0–2) and the presence-gated height head (ch 3). |

Training is **two-stage** (coupled → *dual purify*, the dashed box in the figure); details below.

---

## 🎯 What the Final Submission Is

An ensemble of **5 model variants × 5 leave-region-out folds**, each trained through
the **two stages** (coupled → *dual purify*), combined into the 4-channel prediction:

| `MEMBER` | Ensemble variant | `pixel_backbone_kind` | Seed |
|:---:|---|---|:---:|
| **0, 1, 2** | 🥇 U-Net++ (nested decoder) | `unetpp` | 0, 1, 2 |
| **3** | 🥈 UNet 3+ (full-scale skip) | `unet3plus` | 0 |
| **4** | 🥉 TransUNet (attn bottleneck) | `unetpp_trans` | 0 |

The `MEMBER 0-4` argument to the scripts below indexes exactly these five rows
`(config, seed)` — three U-Net++ seeds plus one UNet 3+ and one TransUNet. `FOLD 0-4`
selects the leave-region-out split. So the 25 member-folds = 5 `MEMBER` × 5 `FOLD`.

Per member-fold, the two stages produce four checkpoints off one stage-1 model
(stage 2 = *dual purify*, i.e. the two frozen-trunk purify branches):

```
stage 1  coupled     coupled seg+height, 80 ep                           -> <exp>
stage 2  dual purify
  ├─ height-purify  20 ep, freeze seg    (presence-trunk-grad-scale 0)   -> <exp>_purify              (ch 3)
  └─ seg-purify     20 ep, freeze height (height-trunk-grad-scale 0)
                    + 20 ep clDice on top                                -> <exp>_segpurify, _cldice  (ch 0-2)
```

> ⏱️ **Note:** stage 1 runs 80 ep to match the final submission, but it keeps its
> *best-val* checkpoint and converges by ~50 ep — running `--epochs 50` reproduces the
> result within noise at half the stage-1 cost (verified: 5-fold seed-0 at 50 ep matched
> the 80-ep submission to ΔScore −0.0006).

**Final channels** (`assemble_final.py`):

- 🗺️ **seg (ch 0–2)** = mean of **50** test predictions (25 `_cldice` + 25 `_segpurify`),
  binarised at OOF-tuned per-class thresholds + a water connected-component filter.
- 📏 **height (ch 3)** = mean of **25** `_purify` test predictions, then a per-class height
  calibration: **building `h → 1.05·h + 0.116`**, **veg `h → h + 0.12`** (derived from the
  model's range-compression / region-shift bias; no public-board tuning).

---

## 🔁 Reproduce: Step by Step

### 1️⃣ Environment

```bash
conda env create -f environment.yml     # creates env "emb2heights"
conda activate emb2heights
```

Key deps: PyTorch (CUDA), numpy, rasterio, scipy, tqdm, pyyaml. One GPU per training job.

### 2️⃣ Data

Place the challenge embeddings/labels under `data/`, or point `DATA_ROOT` elsewhere
(`export DATA_ROOT=/path/to/data`):

```
data/
  train/
    alphaearth_emb/  tessera_emb/                 # dense pixel embeddings (.tif)
    terramind_s1_emb/ terramind_s2_emb/           # coarse token embeddings
    thor_s1_emb/ thor_s2_emb/
    labels/                                       # label_<core>_*.tif  (4-channel GT)
  test/
    alphaearth_test_emb/ tessera_test_emb/
    terramind_test_s1_emb/ terramind_test_s2_emb/
    thor_test_s1_emb/ thor_test_s2_emb/
```

Filenames share a `<core>` id (e.g. `0041_FQ`) that ties an embedding to its label.
`tools/download_data.py` documents where each embedding comes from.

### 3️⃣ Generate the delmask masks *(before training)*

![Deleted building footprints — label empty where the nDSM shows buildings](docs/fig_building_holes.png)

~100 training tiles have building footprints that were **missing from the GT**:
the label is empty where the nDSM + embeddings clearly show buildings (red = our
detector's recovered region above). Training drops the **presence/seg** loss on those
pixels (height is kept), so the model is never punished for correctly predicting a
building. Generate the masks **first**, into `runs/missing_masks/`:

```bash
python tools/generate_missing_masks.py       # -> runs/missing_masks/<core>.npy   (add --report for a ranked summary)
```

They ship precomputed but are git-ignored (binary, fully regenerable); the training
config reads them via `missing_building_mask_dir: ${REPO_DIR}/runs/missing_masks`.

### 4️⃣ Train + Predict + Assemble

**One command runs everything** — trains all 25 member-folds (two stages each), predicts
out-of-fold val + the 946 test tiles, and assembles the submission zip:

```bash
scripts/run_all.sh                 #  -> submission/FINAL_*.zip
```

It is ♻️ **resumable** (finished stages / prediction dirs are skipped) and 🎚️ **subsettable**
via env vars, e.g. a single member-fold smoke test:

```bash
MEMBERS="0" FOLDS="0" scripts/run_all.sh
```

<details>
<summary>🔧 <b>Run the three steps individually</b> (e.g. one per cluster job)</summary>
<br/>

Per `(member, fold)`, `run_all.sh` calls three self-contained steps plus the final assembly:

```bash
scripts/train_member_fold.sh        <MEMBER 0-4> <FOLD 0-4>   # two-stage training (coupled -> dual purify)
scripts/predict_val_member_fold.sh  <MEMBER 0-4> <FOLD 0-4>   # OOF val predictions
scripts/predict_test_member_fold.sh <MEMBER 0-4> <FOLD 0-4>   # 946 test tiles
python assemble_final.py                                       # tune thr + ensemble -> zip
```

</details>

Outputs land in `runs/<exp>/…` (checkpoints, `predictions/` = OOF val, `test_predictions/`
= the 946 test tiles). The submission zip holds 946 `[4,256,256]` float32 `.npy` tiles
under `predictions/`.

### 5️⃣ Evaluate the OOF folds *(optional)*

Score any checkpoint on its held-out fold under the official GT (presence = coverage > 0.10):

```bash
python evaluate.py xfusion_095_unetpp_s0_f0_segpurify 0    # seg IoU  (per-class thr sweep)
python evaluate.py xfusion_095_unetpp_s0_f0_purify    0    # height RMSE
```

---

## 🗂️ Repository Layout

```
core/                model / loss / data / engine / inference / metrics
train.py             training entry point (all stages via CLI flags)
predict.py           inference entry point (val or test)
evaluate.py          official-GT evaluation (presence = coverage > 0.10) per fold
configs/active/      the 3 member configs (+ defaults.yml)
splits/…5fold_seed42 the leave-region-out fold splits (grouped by region code)
scripts/             train / predict / run_all drivers
assemble_final.py    OOF threshold tuning + 50-seg ensemble + height calib -> zip
runs/missing_masks/  delmask masks (git-ignored; generate in Step 3)
tools/               data download, fold generation, missing-mask generation
docs/                framework_overview.pdf (+ .tex), architecture.pdf (+ .png)
```
