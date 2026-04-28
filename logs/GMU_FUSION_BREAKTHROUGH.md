# GMU Fusion Breakthrough — Recipe Notes

The single architectural change that moved this codebase from `0.4644` to
`0.5072` (raw, random-split val) and `0.4999 ± 0.0147` (5-fold
group-stratified mean) is **gated multimodal feature fusion** between the
AlphaEarth and Tessera streams. This note distills how we got there.

## TL;DR

| Stage | Recipe | Score | Δ |
|---|---|---|---|
| Champion N | `tessera_iou_fusion`, residual_presence (legacy) | 0.4644 | — |
| **`uw_gated_F`** | **+ `--fusion-mode gated_feature`** | **0.5072** | **+0.0428** |
| `uw_gated_F` 3-seed bag | seeds {42, 0, 1} | 0.5048 ± 0.0042 | confirms breakthrough |
| 5-fold group-stratified | 5 disjoint-geography folds | **0.4999 ± 0.0147** | honest leaderboard estimate |

## The change

Old fusion ([commit `1aa5701` and prior](#)): Tessera was squeezed to a
16-channel side stream and added as a zero-init residual to the
**presence logits only**. The fraction head, the FiLM step, and the
entire height regression branch never saw any Tessera signal.

New fusion ([commit `1aa5701`](#)): Tessera is promoted to a peer feature
stream at base_ch=48, fused into the trunk via a learned spatial gate:

```
G       = σ(W_g · concat(F_AE, F_TES))
F_fused = G ⊙ F_AE + (1 − G) ⊙ F_TES
```

Mathematically this is a **Gated Multimodal Unit** (Arevalo et al.,
ICLR Workshop 2017). Citation belongs there. The implementation
contribution is the **zero-init weights + bias=+4** trick: at step 0,
sigmoid(4) ≈ 0.98 → fused ≈ AE-only. The model is bit-identical to
AE-only at init and learns Tessera contribution as a residual. Same
hygiene as `presence_delta_head` and the softplus height deltas already
in the codebase.

## Why it moved the needle

Per-axis deltas vs N (champion → uw_gated_F):

| axis | N | uw_gated_F | Δ |
|---|---|---|---|
| iou_bld | 0.5087 | 0.5085 | ≈0 |
| iou_tree | 0.7618 | 0.7668 | +0.005 |
| **iou_wat** | **0.4537** | **0.5018** | **+0.048** |
| **RMSE_bH** | **1.911 m** | **1.677 m** | **−0.234 m** |
| **RMSE_vH** | **3.396 m** | **3.011 m** | **−0.385 m** |

The big gains are precisely on the axes that **never received Tessera
signal under the old fusion**: water (Tessera SAR is great at water),
both heights (Tessera S1+S2 phenology + structural scattering). Building
IoU is unchanged because AE was already strong there — the gate likely
keeps G≈1 over building pixels and opens elsewhere.

## What also got tried in this round and didn't pan out

- **Homoscedastic uncertainty weighting** (Kendall/Gal): replaces hand-tuned
  loss scalars with learned log-variances. Single-arm score `0.4500`
  (regressed −0.014 vs N). Combined with gated_feature: `0.4948` (worse
  than gated_feature alone). Retired.
- **Rich gate** (2-layer MLP) **+ untied gates G_AE/G_TES + modality
  dropout 0.15**, all at once ([commit `8062df6`](#)). Score: `0.5051` —
  inside seed noise of the simple-gate champion. None of the three knobs
  added value on this dataset.
- **Multi-scale gating** at all 4 U-Net levels ([commit `0cc3783`](#)):
  score `0.5023`. Did push `iou_wat` to its highest value across runs
  (`0.5191`) — confirms the per-scale routing mechanism is working — but
  cost too much elsewhere to be a net win.

These are documented for the record so we don't re-litigate them.

## Compute that made this viable

The iteration cadence is what allowed the breakthrough to surface. Two
threads:

- **`--compile` switch** ([commits `0481008`, `88cd256`, `deb6d6c`](#)):
  one flag bundles `torch.compile` (max-autotune-no-cudagraphs),
  channels-last, bf16 AMP, fused AdamW. Persistent Inductor cache at
  `~/.cache/embed2heights_inductor` is shared across SLURM jobs, so
  warmup is paid once per (code, GPU) pair. Steady-state 30-epoch run
  on H100: ~55 min. The 5-fold sweep landed in <1 h wall-clock.
- **`--no-augment` is part of the champion recipe.** All augmentation
  variants explored ([commits `6a405ce`, `d8476fe`, `762a855`](#)) — D4
  rotations, flip_rot180, on-the-fly only — closed without producing a
  positive delta on this dataset. The most decisive measurement: A/B
  test ([`run_ab_modernize.bash`](#)) where the augmented arm's `iou_bld`
  fell `−0.037` and `iou_wat` fell `−0.041` vs the no-aug arm. The
  modernization commits keep the augment plumbing in place but the
  recipe pin is `--no-augment`.

## Honest leaderboard estimate

Random-split val (0.5072) overstates leaderboard performance because
patches in train and val share location codes (the 2-letter suffix on
sample IDs is geographic). Group-stratified 5-fold (Dingqi's
`splits/group_code_5fold_seed42`) is the right protocol:

```
fold 0: 0.4766  (hardest — high-building scenes)
fold 1: 0.5224
fold 2: 0.4979
fold 3: 0.4986
fold 4: 0.5041
mean ± std = 0.4999 ± 0.0147
```

Compare with the prior baseline's `~0.42` on fold 0: gated_feature adds
**+0.057 on the most-honest fold**. Submission expectation for the
5-fold ensemble: `~0.50–0.52` depending on test geography distribution.

## Submission

Generated by [`run_submission_ensemble.bash`](#): predict all 5 fold
checkpoints on the test embeddings → average → zip 946 `.npy` files
into `runs/submission/gated_F_5fold_ensemble.zip`. Each file is
`(4, 256, 256)` `[building%, veg%, water%, height_m]`.

## Cross-task spatial attention (`ctaskattn_v1`) — small positive signal

After `gated_feature`, the next architectural move that produced a
*positive* delta on fold 0 was **cross-task spatial attention from the
land-cover head onto the height_trunk** (commit `b689aff`,
`--cross-task-attention`). One additional piece inside
`MultiTaskPredictionHead`, downstream of the existing FiLM step:

```
fractions (B, 3, H, W) ──► Conv1x1(3 → 1) ──► sigmoid ──► attn ∈ (0, 1)
                            (zero-init weights)

FiLM-conditioned height features  h  ──►  h * (0.5 + attn)   ∈ [0.5·h, 1.5·h]
```

Zero-init weights → `attn = sigmoid(0) = 0.5` at t=0 → multiplier 1.0
→ no-op at init (same warm-start hygiene as gated_feature). Gain is
soft (range [0.5, 1.5]) so a confidently wrong segmentation cannot
zero out the height.

| run (fold 0) | iou_bld | iou_tree | iou_wat | RMSE_bH | RMSE_vH | Score | Δ |
|---|---|---|---|---|---|---|---|
| `gated_F_fold0` (champion) | 0.5003 | 0.7518 | 0.4692 | 1.8442 | 3.1997 | **0.4766** | — |
| **`ctaskattn_v1`** | **0.5015** | **0.7533** | **0.4950** | 1.8548 | **3.1771** | **0.4810** | **+0.0044** |

Gain concentrated on `iou_water` (+0.026) and a small lift on `iou_bld`,
with `RMSE_vH` slightly improved. Within the +0.006 noise margin on a
single fold but every per-axis number moved the right direction. Worth
a 5-fold confirmation before promoting.

## What also got tried in this round and didn't pan out

- **4-modality flat GMU** (`multi_gfm_v1`, commit `e1fefab`): k=4 untied
  gates over AE+Tessera+TerraMind+THOR (Arevalo Figure 2a). Anchor
  warm-start (AE open, others closed). Result on fold 0: pending re-eval.
- **Hierarchical bipartite GMU** (`hierarchical_v1`, commit `b689aff`):
  3-stage tree of bimodal GMUs grouped by temporal scope (annual ↔
  epoch). Fold 0: **0.4666 (−0.0100)**. The grouping prior over-
  constrained the model. Closed.
- **Dirichlet auxiliary loss** (`dirichlet_v1`, commit `b689aff`):
  4-class softplus head + Dirichlet NLL on the simplex
  {bld, veg, wat, background}. Fold 0: **0.4626 (−0.0140)**. The α
  values diverged (sum-of-α grew unbounded, val_loss hit −13) but the
  IoU/RMSE metrics didn't improve. Closed.

These are documented for the record so we don't re-litigate them.

## How to run these models

All commands assume the conda env is active and CWD is the project
root:

```bash
SCRIPT_DIR=/u/dkiv2/group_dkiv2/active/embed2heights
DATA_DIR=/projects/bcrm/emb2height/data/train
TEST_DIR=/projects/bcrm/emb2height/data/test
cd $SCRIPT_DIR
conda activate emb2heights
```

### Train `ctaskattn_v1` (champion + cross-task attention)

The single CLI delta vs `gated_feature` champion is
`--cross-task-attention`. Everything else mirrors `uw_gated_F`'s
recipe:

```bash
python train.py \
    --experiment-name ctaskattn_v1 \
    --model-type tessera_iou_fusion \
    --train-embeddings-dir $DATA_DIR/alphaearth_emb \
    --secondary-train-embeddings-dir $DATA_DIR/tessera_emb \
    --train-targets-dir $DATA_DIR/labels \
    --split-file $SCRIPT_DIR/splits/group_code_5fold_seed42/fold_0/split.json \
    --batch-size 32 --patch-size 256 --epochs 30 \
    --lr 2e-4 --weight-decay 1e-4 \
    --loss-preset presence_centered \
    --aux-weight 1.0 --presence-tversky-weight 1.0 --fraction-mae-weight 0.1 \
    --tessera-presence-ch 16 --tessera-hidden-ch 96 --tessera-hidden-depth 2 \
    --height-specialist-depth 2 --lightunet-base-ch 48 \
    --build-height-boost 5.0 --veg-height-boost 1.5 --aux-veg-weight 1.0 \
    --iou-loss-kind tversky --focal-gamma 2.0 --focal-alpha 0.25 \
    --structure-weight 2.0 --no-augment --scheduler plateau --compile \
    --fusion-mode gated_feature --cross-task-attention \
    --seed 42
```

Or via slurm:

```bash
sbatch run_tier1_2_fold0.bash    # array index 2 = ctaskattn_v1
```

### Predict on the test set (label-free, submission format)

```bash
python predict.py \
    --experiment-name ctaskattn_v1 \
    --model-type tessera_iou_fusion \
    --test-embeddings-dir $TEST_DIR/alphaearth_test_emb \
    --secondary-test-embeddings-dir $TEST_DIR/tessera_test_emb
```

`fusion_mode`, `cross_task_attention`, channel widths, etc. are
auto-loaded from `runs/<exp>/training_params.json` — no need to repeat
training-time flags. Outputs land in `runs/<exp>/predictions/` as
`<core>_<region>_<year>.npy`, shape `(4, 256, 256)`,
`[building%, veg%, water%, height_m]`.

### Predict on the train set for offline eval (paired mode)

```bash
python predict.py \
    --experiment-name ctaskattn_v1 \
    --model-type tessera_iou_fusion \
    --test-embeddings-dir $DATA_DIR/alphaearth_emb \
    --secondary-test-embeddings-dir $DATA_DIR/tessera_emb \
    --test-targets-dir $DATA_DIR/labels
```

### Evaluate on a chosen split

Group-fold (the honest leaderboard estimate):

```bash
python evaluate.py \
    --only ctaskattn_v1 \
    --val-only \
    --split-file $SCRIPT_DIR/splits/group_code_5fold_seed42/fold_0/split.json \
    --labels-dir $DATA_DIR/labels
```

Random val (reproduces the old 0.5072 baseline number):

```bash
python evaluate.py \
    --only ctaskattn_v1 \
    --val-only \
    --split-file $SCRIPT_DIR/splits/split.json \
    --labels-dir $DATA_DIR/labels
```

### Train a 5-fold bag for the new champion

Pattern is identical to [`run_gated_F_5fold.bash`](../run_gated_F_5fold.bash);
just add `--cross-task-attention` and rename. Once trained,
[`run_submission_ensemble.bash`](../run_submission_ensemble.bash)
generalizes — point it at the new fold experiment names.

### Re-train any other variant

| variant | model_type | extra flags |
|---|---|---|
| `gated_feature` champion | `tessera_iou_fusion` | `--fusion-mode gated_feature` |
| `ctaskattn` (this section) | `tessera_iou_fusion` | `--fusion-mode gated_feature --cross-task-attention` |
| 4-modality flat GMU | `multi_gfm` | `--fusion-mode gated_feature` (default for multi_gfm), all 4 emb dirs |
| Hierarchical bipartite | `multi_gfm` | `--fusion-mode hierarchical`, all 4 emb dirs |
| Dirichlet aux | `tessera_iou_fusion` | `--fusion-mode gated_feature --dirichlet-weight 0.5` |

For the 4-modality variants, supply
`--terramind-s1-train-emb-dir`, `--terramind-s2-train-emb-dir`,
`--thor-s1-train-emb-dir`, `--thor-s2-train-emb-dir`. Test-time uses
the analogous `--*-test-emb-dir` flags.

## Where to push next

- Confirm `ctaskattn_v1` across the 5 group folds before promoting.
- Resolve `multi_gfm_v1` (4-modality flat) once 80776 evaluates — that
  decides whether more modalities help in any form.
- If both confirm: stack `ctaskattn` on top of the chosen fusion, run
  the 5-fold bag with `--cross-task-attention`, ensemble for submission.
