# Embed2Heights — Final Submission (reproducible)

Team: Attention_Plzzz



Reproduction package for our final leaderboard submission.

**Public score 0.5067** — `IoU_build 0.5032 / IoU_veg 0.8211 / IoU_water 0.5270 / RMSE_H_build 1.782 m / RMSE_H_veg 3.072 m`.

The system predicts, per `256×256` tile, a `[building, vegetation, water, height(m)]`
tensor from the frozen challenge embeddings. Everyone gets the *same* embeddings, so
the whole design is about matching the metric and squeezing robust signal — no raw
imagery, no external data.

---

## 1. What the final submission is

An ensemble of **5 model variants × 5 leave-region-out folds**, each trained through
**4 stages**, combined into the 4-channel prediction:

| Ensemble member | `pixel_backbone_kind` | seeds |
|---|---|---|
| U-Net++ (nested decoder) | `unetpp` | 0, 1, 2 |
| UNet 3+ (full-scale skip) | `unet3plus` | 0 |
| TransUNet (attn bottleneck) | `unetpp_trans` | 0 |

Per member-fold, four checkpoints are produced off one stage-1 model:

```
stage 1        coupled seg+height, 50 ep                                 -> <exp>
  ├─ height-purify  20 ep, freeze seg    (presence-trunk-grad-scale 0)   -> <exp>_purify              (ch 3)
  └─ seg-purify     20 ep, freeze height (height-trunk-grad-scale 0)
                    + 20 ep clDice on top                                -> <exp>_segpurify, _cldice  (ch 0-2)
```

Final channels (`assemble_final.py`):
- **seg (ch 0-2)** = mean of **50** test predictions (25 `_cldice` + 25 `_segpurify`), binarised at OOF-tuned per-class thresholds + a water connected-component filter.
- **height (ch 3)** = mean of **25** `_purify` test predictions, then a per-class height calibration: **building `h → 1.05·h + 0.116`**, **veg `h → h + 0.12`** (both derived from the model's known range-compression / the region-shift bias; no public-board tuning).

Full method write-up: **`docs/framework_overview.pdf`**.

---

## 2. Environment

```bash
conda env create -f environment.yml     # creates env "emb2heights"
conda activate emb2heights
```
Key deps: PyTorch (CUDA), numpy, rasterio, scipy, tqdm, pyyaml. One GPU per training job.

---

## 3. Data layout

Place the challenge embeddings/labels under `data/` (or point `DATA_ROOT` elsewhere):

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
`tools/download_data.py` documents where each embedding comes from.
Filenames share a `<core>` id (e.g. `0041_FQ`) that ties an embedding to its label.

---

## 4. Reproduce

**One command runs everything** — train all 25 member-folds (4 stages each),
predict out-of-fold val + the 946 test tiles, and assemble the submission zip:

```bash
scripts/run_all.sh                 #  -> submission/FINAL_*.zip
```

It is **resumable** (finished train stages / prediction dirs are skipped) and
**subsettable** via env vars, e.g. a single member-fold as a smoke test:

```bash
MEMBERS="0" FOLDS="0" scripts/run_all.sh
```

Per `(member, fold)`, `run_all.sh` calls three self-contained steps (usable on
their own, e.g. one per cluster job):

```bash
scripts/train_member_fold.sh        <MEMBER 0-4> <FOLD 0-4>   # 4 training stages
scripts/predict_val_member_fold.sh  <MEMBER 0-4> <FOLD 0-4>   # OOF val predictions
scripts/predict_test_member_fold.sh <MEMBER 0-4> <FOLD 0-4>   # 946 test tiles
python assemble_final.py                                       # tune thr + ensemble -> zip
```

Outputs land in `runs/<exp>/…` (checkpoints, `predictions/` = OOF val used for
threshold tuning + `evaluate.py`, `test_predictions/` = the 946 test tiles).
The submission zip contains 946 `[4,256,256]` float32 `.npy` tiles under `predictions/`.

### Cluster note
Each `(member, fold)` is a self-contained job. Example SLURM array (25 tasks):
```bash
# in a submit script:  M=$((SLURM_ARRAY_TASK_ID/5)); F=$((SLURM_ARRAY_TASK_ID%5))
#   scripts/train_member_fold.sh $M $F \
#     && scripts/predict_val_member_fold.sh $M $F \
#     && scripts/predict_test_member_fold.sh $M $F
sbatch --array=0-24 <your_wrapper>.sbatch
# then once, after the array finishes:  python assemble_final.py
```
Staging the embeddings to node-local disk is recommended (shared-FS I/O contention).

---

## 5. The `delmask` dependency (provided)

Training drops the presence/seg loss on ~100 tiles whose building footprints were
human-deleted from the GT (`missing_building_mask_dir`). These masks live in
**`runs/missing_masks/`** as 100 `.npy` files.

They are **not tracked in git** (binary, ~6 MB, fully regenerable — the directory
ships with only a `.gitkeep` placeholder). Generate them before training with:

```
python tools/generate_missing_masks.py      # add --report for a ranked summary
```

---

## 6. Repository layout

```
core/                model / loss / data / engine / inference / metrics
train.py             training entry point (all stages via CLI flags)
predict.py           inference entry point (val or test)
evaluate.py          official-GT evaluation (presence = coverage > 0.10) per fold
configs/active/      the 3 member configs (+ defaults.yml)
splits/…5fold_seed42 the leave-region-out fold splits (grouped by region code)
scripts/             train / predict / run_all drivers
assemble_final.py    OOF threshold tuning + 50-seg ensemble + height calib -> zip
runs/missing_masks/  delmask masks (git-ignored; generate before training)
tools/               data download, fold generation, missing-mask generation
docs/                framework_overview.pdf (+ .tex), architecture.pdf
```
