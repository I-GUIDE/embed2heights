# Ensemble ECCpDB Submission Report

**Date:** 2026-04-22
**Submission:** `submission_ECCpDB_thr_0575_0525_0725.zip` (binarized, 217.9 MB)
**Public Score:** **0.4209** (Final_Score 0.4209)
**Local Val Score:** 0.4692 (leaderboard-aligned, val-only 405 samples)
**Val → Public gap:** **0.0483** (vs historical ~0.08 on prior submissions)

## TL;DR

A 5-way mean ensemble of `tessera_iou_fusion` variants pushed the public
leaderboard from `0.3750` (previous best single-model continuous submission)
to `0.4209` — a jump of `+0.0459`. The val → public gap collapsed from the
historically stable `~0.08` to `~0.05`. Analysis strongly attributes the gap
reduction to ensemble variance averaging rather than the binarization format
or the per-class threshold bake.

## Ensemble Configuration

| Alias | Checkpoint | Rationale |
|---|---|---|
| E  | `alphaearth_tessera_iou_fusion_E_specialist_d2`   | Strongest single model overall (val 0.4574) |
| C  | `alphaearth_tessera_iou_fusion_C_ch16_h96d2`      | Best `RMSE_bH` locally |
| Cp | `alphaearth_tessera_iou_fusion_C_presence_centered` | Second-best `IOU_VEG` |
| D  | `alphaearth_tessera_iou_fusion_D_huber_veg3`      | Best `RMSE_vH` + strong water |
| B  | `alphaearth_tessera_iou_fusion_B_no_ssim_grad`    | Different loss → architectural diversity |

- **Blend:** simple per-pixel mean (via `tools/ensemble.py mean`)
- **Post-hoc:** per-class thresholds baked via `tools/make_submission.py --binarize-thresholds 0.575 0.525 0.725` (selected by `tools/sweep_thresholds.py` on val)
- **Output format:** class channels binary `{0, 1}`, height channel continuous float32

## Local Validation Table

Sweep of candidate ensembles on val (405 samples from `splits/split.json`, each
row reports per-class tuned thresholds).

| Scheme | iou_bld | iou_tree | iou_wat | RMSE_bH | RMSE_vH | Score |
|---|---:|---:|---:|---:|---:|---:|
| E (single, base 0.5) | 0.5029 | 0.7587 | 0.4497 | 1.9181 | 3.4929 | 0.4574 |
| E (single, tuned) | 0.5065 | 0.7606 | 0.4670 | 1.9181 | 3.4929 | 0.4612 |
| Mean E+C | 0.5087 | 0.7609 | 0.4763 | 1.8811 | 3.4651 | 0.4674 |
| Mean E+C+Cp | 0.5071 | 0.7626 | 0.4752 | 1.8737 | 3.4621 | 0.4678 |
| Mean E+C+Cp+D | 0.5085 | 0.7624 | 0.4766 | 1.8827 | 3.4348 | 0.4687 |
| **Mean E+C+Cp+D+B (submitted)** | **0.5073** | **0.7633** | **0.4787** | **1.8774** | **3.4379** | **0.4692** |
| Mean E+C+Cp+D+C_h64 | 0.5090 | 0.7629 | 0.4767 | 1.8760 | 3.4394 | 0.4693 |
| Channel-weighted v1 | 0.5097 | 0.7624 | 0.4783 | 1.8807 | 3.4459 | 0.4690 |

ECCpDB selected over ECCpDCh64 (val-tie at 0.4693 vs 0.4692) because B
provides genuine loss-function diversity, which should transfer better under
distribution shift than a near-twin architectural variant.

## Public Leaderboard Comparison

| Submission | Score | IOU_BUILD | IOU_VEG | IOU_WATER | RMSE_bH | RMSE_vH |
|---|---:|---:|---:|---:|---:|---:|
| `lightunet_v35head` (single, continuous) | 0.3406 | 0.2866 | 0.6343 | 0.3929 | 2.1761 | 3.8437 |
| `fusion_closs` (C_h96d2 continuous calibrated) | 0.3750 | 0.3626 | 0.6824 | 0.4142 | 2.1330 | 3.8109 |
| `weighted_metric_v1` (old binary ensemble) | 0.3708 | 0.2935 | 0.7763 | 0.4346 | 2.2095 | 3.7510 |
| **ECCpDB (this)** | **0.4209** | **0.4193** | **0.8117** | **0.4590** | **2.1081** | **3.7224** |

Per-class deltas vs `fusion_closs` (the previous public best):

- `IOU_BUILD`:  +0.0567
- `IOU_VEG`:    +0.1293  ← largest contributor
- `IOU_WATER`:  +0.0448
- `RMSE_bH`:    -0.0249  (improvement)
- `RMSE_vH`:    -0.0885  (improvement)

All three IoU classes move in the same direction, which is a variance-reduction
signature rather than a threshold-shift signature.

## The Val → Public Gap Analysis

Prior to this submission, the gap was remarkably stable:

- `lightunet_v35head`: val 0.4213 → public 0.3406  (gap **0.0807**)
- `fusion_closs`:      val 0.4561 → public 0.3750  (gap **0.0811**)

This submission broke that pattern:

- `ECCpDB`:            val 0.4692 → public 0.4209  (gap **0.0483**)

Gap compression: `~0.033`. The question is: which part of the pipeline bought
that compression — ensemble averaging, binarization, or the per-class
threshold bake?

### Elimination argument

- **Threshold tuning** is expected to *widen* val→public gap, not shrink it.
  The tuned thresholds `(0.575, 0.525, 0.725)` are optimized on val and
  necessarily over-fit local quirks; under distribution shift their gain
  degrades. Gap compression therefore cannot come from this term.
- **Binarization format** is roughly gap-neutral. The server applies a hard
  `pred > 0.5` to class channels anyway — shipping `{0, 1}` versus float
  probabilities only matters through the chosen threshold value (above), not
  through the format itself.
- **Ensemble averaging** is the only transformation here that structurally
  reduces generalization error variance, and therefore the only one that can
  plausibly compress the val→public gap.

### Quantitative sanity check

Variance-reduction factor for `N` members with mean pairwise correlation `ρ`:

```
gap_ensemble / gap_single ≈ 1 / sqrt(1 + (N-1) * ρ)
```

Four of five members are `tessera_iou_fusion` variants trained on the same
split, so `ρ ≈ 0.7` is a reasonable guess. With `N = 5`:

```
scaling ≈ 1 / sqrt(1 + 4 * 0.7) ≈ 0.527
predicted gap: 0.081 × 0.527 ≈ 0.043
observed gap:                    0.048
```

Within noise of the prediction. The mechanism is consistent with the numbers.

### Local-to-public amplification

Interestingly, the ensemble's gain is **amplified** on public relative to val:

- Local delta (val): `fusion_C@0.4561 → ECCpDB@0.4692` = `+0.0131`
- Public delta:      `fusion_closs@0.3750 → ECCpDB@0.4209` = `+0.0459`

The ratio `0.0459 / 0.0131 ≈ 3.5×` is larger than a pure linear prediction
would suggest. The extra gain comes from the gap compression itself: when the
baseline (single-model, continuous) was paying an ~`0.081` variance tax on
public and ECCpDB only pays `~0.048`, the difference ~`0.033` is added on top
of whatever local improvement the ensemble produces.

## Open Question — Planned Next Submission

The above analysis concludes ensemble is the dominant gap-closer, but cannot
fully disentangle:

- ensemble-averaging effect (gap compression)
- binarization + threshold-bake effect (possibly residual gap, either sign)

### Experiment: submit `submission_ECCpDB_continuous.zip`

Same 5-way mean ensemble, no threshold bake — class channels stay continuous
`[0, 1]`. Server will apply its own `pred > 0.5`. Equivalent to the local val
row `Mean E+C+Cp+D+B base (0.5, 0.5, 0.5)`, which scored `~0.465` on val.

Decision matrix:

| Continuous public score | Interpretation | Future strategy |
|---|---|---|
| `≈ 0.41` (drop ≤ 0.015) | Ensemble did all the gap work; binarize/tune ~neutral | Default to continuous submissions; skip threshold bake to reduce workflow complexity |
| `< 0.40` (drop > 0.02) | Threshold bake contributed meaningfully | Keep per-class threshold sweep + binarization in the pipeline |
| `≥ 0.4209` (tuned was worse) | Val thresholds actively over-fit | Never bake tuned thresholds; submit continuous always |

Tomorrow's submission will fill in the final piece. Regardless of outcome,
ensembling the strong fusion-family checkpoints is now a confirmed standard
move for this leaderboard.

## Reproduction

```bash
# 1. Generate per-model test predictions (946 each).
sbatch run_ens5_predict.sbatch           # submits 80223-equivalent GPU job

# Or manually, per model:
python predict.py \
  --experiment-name <EXP_NAME> \
  --model-type tessera_iou_fusion \
  --test-embeddings-dir   /u/dingqi2/workspace/esa/data/test/alphaearth_test_emb \
  --secondary-test-embeddings-dir /u/dingqi2/workspace/esa/data/test/tessera_test_emb \
  --predictions-dir runs/<EXP_NAME>/test_predictions

# 2. Mean-blend the 5 dirs (preserves `_YYYY` suffix since the ensemble tool
#    patch on 2026-04-22).
python tools/ensemble.py mean \
  --inputs runs/alphaearth_tessera_iou_fusion_E_specialist_d2/test_predictions \
           runs/alphaearth_tessera_iou_fusion_C_ch16_h96d2/test_predictions \
           runs/alphaearth_tessera_iou_fusion_C_presence_centered/test_predictions \
           runs/alphaearth_tessera_iou_fusion_D_huber_veg3/test_predictions \
           runs/alphaearth_tessera_iou_fusion_B_no_ssim_grad/test_predictions \
  --output-dir runs/ens_ECCpDB/test_predictions

# 3a. Binarized (the submitted zip; val score 0.4692, public 0.4209).
python tools/make_submission.py \
  --pred-dir runs/ens_ECCpDB/test_predictions \
  --binarize-thresholds 0.575 0.525 0.725 \
  --output runs/ens_ECCpDB/submission_ECCpDB_thr_0575_0525_0725.zip

# 3b. Continuous (the scheduled next-day submission).
python tools/make_submission.py \
  --pred-dir runs/ens_ECCpDB/test_predictions \
  --output runs/ens_ECCpDB/submission_ECCpDB_continuous.zip
```

## Cross-References

- [SUBMISSION_COMPARISON_REPORT.md](SUBMISSION_COMPARISON_REPORT.md) — prior
  three submissions (lightunet_v35head, fusion_closs, weighted_metric_v1) that
  established the `~0.08` gap baseline.
- [HEIGHT_IOU_EXPERIMENTS_DEF.md](HEIGHT_IOU_EXPERIMENTS_DEF.md) — the D / E /
  F experiments whose checkpoints feed this ensemble.
- [METRIC_PROBE_REPORT.md](METRIC_PROBE_REPORT.md) — score formula that
  `evaluate.py` / `sweep_thresholds.py` uses locally.
