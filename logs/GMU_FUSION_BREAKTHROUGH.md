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

## Where to push next

The fusion scaffold composes cleanly to 4 modalities. TerraMind and
THOR embeddings are already on disk at
`/projects/bcrm/emb2height/data/train/{terramind,thor}_s{1,2}_emb`.
Extending `gated_feature` to a 4-stream multimodal-GMU
([Arevalo §3.1, Figure 2a](https://arxiv.org/abs/1702.01992)) is the
next obvious swing — same zero-init bias trick, k=4 gates instead of
2, no other architectural change required.
