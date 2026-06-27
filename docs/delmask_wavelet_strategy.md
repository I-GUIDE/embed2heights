# Combined strategy: delmask + wavelet (covgt10 ensemble)

Two **orthogonal** building-IoU levers stacked on the `covgt10` two-stage lineage
(`xfusion_095_p3_2stage_softbin_covgt10`). Both were validated standalone on fold0
seed0 under the official `cov>0.10` GT; this branch combines them for the
3-seed × 5-fold ensemble submission.

Combined config: `configs/active/xfusion_095_p3_2stage_softbin_covgt10_delmask_wavelet.yml`

---

## Lever 1 — delmask (loss masking on deleted footprints)

**Problem.** The training labels have systematic *missing* building footprints —
regions where the building footprint was (we believe) human-deleted, even though the
nDSM height channel and the AlphaEarth embedding both say "building". Penalizing the
model for predicting building there teaches it to suppress real buildings.

**Fix.** Drop the presence/seg loss on those flagged pixels (set `valid_mask[0]=0`),
training split only. Height supervision is kept intact (`valid_mask[1]` untouched —
the nDSM there is real building height). Mirrors the existing `ndsm_hole` convention
but for the presence label.

- Detector: `scratchpad/missing_building_detector_v2.py` — density-gap, not per-pixel
  AND. `E = (height>3) & (p_build>0.6)` (label-independent); `gap = blur(E) - blur(bld>0.10)`;
  flag `gap>0.25 & no-class & height>2`, morphological-close, keep regions ≥80px. This
  fixes the v1 salt-and-pepper fragmentation that deleted genuine under-labeled regions.
- Masks: 100 tiles / 589k px → `runs/missing_masks/<core>.npy` (regenerate with
  `scratchpad/export_missing_masks.py`).
- Config knob: `data.missing_building_mask_dir` (CLI `--missing-building-mask-dir`,
  default off). Threaded train-only in `core/data/training.py`; applied in
  `core/data/datasets.py:_apply_missing_mask`.

**Result (fold0 seed0, `cov>0.10` building IoU):**

| eval | baseline | delmask | Δ |
|---|---|---|---|
| DIRTY (vs gappy GT) | 0.4795 | 0.4855 | +0.0061 |
| CLEAN (flagged excluded, fair A/B) | 0.4835 | 0.4937 | **+0.0102** |

The gappy-label eval *under-credits* delmask (it scores the model's freed predictions in
flagged regions as FP). The fair eval (`scratchpad/eval_clean_delmask.py`, flagged pixels
excluded for both models) ≈ doubles the edge to +0.0102 (~3σ vs ~0.003 seed spread).
Best threshold shifts 0.525 → 0.600 (the model predicts building more freely).

---

## Lever 2 — wavelet (edge-preserving downsampling)

**Problem.** The 3 `MaxPool2d(2)` downsamplers in the pixel LightUNet branches alias
away the building-edge high frequencies before the encoder bottleneck.

**Fix.** `pixel_backbone_kind=wavelet`: replace each MaxPool with Haar-DWT pooling that
keeps LL (local average) + LH/HL/HH (edge detail) subbands, concatenates (→4·C), and
projects back to C with a 1×1 conv warm-started to copy LL only (== average pooling).
Anti-aliased, shift-stable (WaveCNet motivation); channel-preserving so stage-1
checkpoints still warm-start stage-2 strict-clean. (`core/models/backbones.py`,
`core/models/token_fusion.py`; smoke test `tools/smoke_wavelet.py`.)

**Result (fold0 seed0, `cov>0.10`):** building IoU 0.4795 → **0.4869 (+0.0074)**,
water IoU 0.4938 → 0.5053, veg flat; building height RMSE 1.6754 → 1.6630 m (better),
veg height RMSE 2.9600 → 2.9930 m (slightly worse). Net submission-equivalent Score +0.0033.

---

## Why stack them

The mechanisms are independent: delmask cleans the *supervision target* (which pixels
contribute to the presence loss); wavelet changes the *encoder downsampling* (which
frequencies survive). They touch disjoint parts of the pipeline, so they *should* stack
— though additivity is not guaranteed and the 3×5 OOF is the arbiter.

## Final pipeline (per the submission requirement)

Each model is the **full two-stage** run (`array_delmask_wavelet.sbatch`, one job per
seed×fold, data staged once to node-local NVMe):

1. **stage 1** (80 ep, coupled) → `runs/<EXP>/model_best.pth` = **IoU** channels 0–2
2. **stage 2** (20 ep, purify; `--init-checkpoint` + `--presence-trunk-grad-scale 0`,
   `--epochs 20 --lr 0.00015`) → `runs/<EXP>_purify/model_best.pth` = **height** channel 3
3. predict + eval both checkpoints on the fold val split

**Submission** = ensemble over the 15 models:
- channels 0–2 (building/veg/water) = mean of the **stage-1** seg, binarized at the
  OOF-optimal `cov>0.10` thresholds
- channel 3 (height) = mean of the **stage-2** purify height

Assembly mirrors `assemble_covgt10_submission.py`; thresholds re-tuned on the combined OOF.

## Status

- Sweep: SLURM array job `116389`, 3 seed × 5 fold = 15 models, full two-stage per job.
- Baseline to beat: covgt10 stage-1 fold0 building `cov>0.10` IoU = 0.4795.
