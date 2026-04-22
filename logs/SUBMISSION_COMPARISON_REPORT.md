# Submission Comparison Report

Date: 2026-04-21

This note compares three public leaderboard submissions and explains the main patterns behind their scores, especially why the old `weighted_metric_v1` submission achieved unusually high `IOU_VEG`.

## Scope

Compared submissions:

1. `lightunet_v35head` test submission from `runs/lightunet_v35head/test_predictions_alphaearth`
2. `fusion_closs` continuous-rescaled submission from `runs/alphaearth_tessera_iou_fusion_C_ch16_h96d2/submission_presence_calibrated_072_054_092.zip`
3. `alphaearth_weighted_metric_v1_calibrated_test_submit.zip`

The first two have leaderboard-aligned local validation metrics available. The third is an older ensemble submission whose local leaderboard-aligned validation record is no longer materialized in the repo, so analysis there relies on the submission artifact itself plus the returned leaderboard scores.

## Server Results

These server metrics were returned by the leaderboard.

| Submission | Score | IoU_build | IoU_veg | IoU_water | RMSE_H_build | RMSE_H_veg |
|---|---:|---:|---:|---:|---:|---:|
| `lightunet_v35head` | 0.3406 | 0.2866 | 0.6343 | 0.3929 | 2.1761 | 3.8437 |
| `fusion_closs` continuous calibrated | 0.3750 | 0.3626 | 0.6824 | 0.4142 | 2.1330 | 3.8109 |
| `weighted_metric_v1` calibrated test submit | 0.3708 | 0.2935 | 0.7763 | 0.4346 | 2.2095 | 3.7510 |

## Local Validation Anchors

### `lightunet_v35head`

Leaderboard-aligned local validation sweep:

- base `(0.5, 0.5, 0.5)`: score `0.4213`
- best per-class `(0.375, 0.400, 0.675)`: score `0.4276`

Source: local sweep on `runs/lightunet_v35head/predictions`.

### `alphaearth_tessera_iou_fusion_C_ch16_h96d2`

Leaderboard-aligned local validation metrics:

- base `(0.5, 0.5, 0.5)`: score `0.4472`
- best per-class `(0.72, 0.54, 0.92)`: score `0.4561`
- continuous calibration via `logit shift` reproduces the same `0.4561` at server-side `0.5`

Source: [presence_calibration_val.json](../runs/alphaearth_tessera_iou_fusion_C_ch16_h96d2/presence_calibration_val.json).

## Key Findings

### 1. The public leaderboard gap is stable across recent runs

For the two submissions with reliable local leaderboard-aligned validation:

- `lightunet_v35head`: `0.4213 -> 0.3406`, gap `-0.0807`
- `fusion_closs` calibrated: `0.4561 -> 0.3750`, gap `-0.0811`

This is strong evidence that:

- the local leaderboard-aligned evaluator is directionally correct
- the public leaderboard subset is consistently harder than the local validation split by about `0.08`

Practical rule of thumb at the moment:

- expected public score `≈ local leaderboard-val score - 0.08`

### 2. The continuous calibration did not obviously hurt the fusion submission

The calibrated fusion submission scored `0.3750`, which is almost exactly what the stable `-0.08` mapping predicts from the local calibrated validation score `0.4561`.

That makes the following interpretation more likely than not:

- the continuous calibration itself was not the main source of failure
- the fusion model simply experiences the same public-set drop as `v35head`

This does not prove the server threshold is exactly `0.5`, but it does support the broader hypothesis that the server is applying a fixed hard threshold to continuous presence scores.

### 3. The old ensemble is highly vegetation-specialized

The old `weighted_metric_v1` submission got:

- mediocre `IOU_BUILD`: `0.2935`
- strong `IOU_WATER`: `0.4346`
- extremely strong `IOU_VEG`: `0.7763`

This pattern is not balanced performance. It is a classic "veg-heavy ensemble" signature.

## Why `IOU_VEG` Was So High In `weighted_metric_v1`

The most likely explanation is a combination of dataset properties and ensemble behavior.

### A. Vegetation is much easier than building or water

From [LABEL_BAND_ANALYSIS.md](./LABEL_BAND_ANALYSIS.md):

- building nonzero pixels: `3.3%`
- vegetation nonzero pixels: `40.7%`
- water nonzero pixels: `1.7%`

Vegetation is:

- common rather than rare
- spatially contiguous
- less sensitive to single-pixel false positives than building/water

That alone makes high vegetation IoU much easier to achieve.

### B. Vegetation labels are near-binary

The label analysis already notes that vegetation is much closer to a hard 0/1 occupancy pattern than building or water.

Implication:

- hard-thresholded vegetation masks lose relatively little information
- binary post-processing can work surprisingly well on vegetation even when it would damage building/water

### C. The submission artifact confirms it was a hard-mask submission

Inspection of `submission/alphaearth_weighted_metric_v1_calibrated_test_submit.zip` shows:

- class channels are exactly binary `0/1`
- sample means over the first 50 test patches:
  - building: `0.0252`
  - vegetation: `0.2866`
  - water: `0.0407`

Artifact facts from the zip:

- `mins = [0, 0, 0, 0]`
- `maxs = [1, 1, 1, 25.456989]`
- `frac_non01 = [0, 0, 0]`

So the vegetation channel was not "saved" by server-side thresholding. It was already a hard vegetation mask before upload.

### D. Channel-wise ensemble blending likely helped vegetation most

The old `weighted_metric_v1` is documented as an in-memory ensemble over:

- HRNet-W18
- LightUNet
- EmbeddingRefiner

Even though the exact channel-weight JSON is no longer present in the repo, the ensemble design was channel-aware. Vegetation is the class where different AlphaEarth backbones are most likely to agree on large contiguous regions. That kind of agreement is exactly where ensembles help the most.

For sparse classes:

- building suffers more from calibration drift and small-object miss/false positive tradeoffs
- water suffers from rarity and threshold instability

For vegetation:

- agreement regions are large
- majority-like blending is robust
- hard thresholding after blending can produce very clean masks

### E. Public subset composition likely favored this behavior

Without public labels this cannot be proved directly, but the returned score suggests the public subset used for that leaderboard evaluation was especially favorable to vegetation-heavy predictions:

- `IOU_VEG = 0.7763` is far above the recent fusion run's `0.6824`
- but that advantage did not transfer to building or RMSE

So the ensemble likely matched the public vegetation distribution well while remaining weak on the building-heavy part of the score.

## Interpretation By Submission

### `lightunet_v35head`

Strengths:

- decent vegetation
- reasonable water

Weaknesses:

- building too weak for the metric weighting
- height also slips on public

Overall:

- a cleaner baseline than the old ensemble
- still not enough building power

### `fusion_closs` continuous calibrated

Strengths:

- best building IoU among the three submissions
- best building RMSE among the three submissions
- overall best public score so far

Weaknesses:

- vegetation lower than the old ensemble
- still suffers the same `~0.08` public gap as `v35head`

Overall:

- the best balanced submission so far
- most aligned with the actual weighted metric

### `weighted_metric_v1`

Strengths:

- exceptional vegetation IoU
- good water IoU

Weaknesses:

- weak building IoU
- weak building RMSE
- binary output format is now risky / unsupported under current submission assumptions

Overall:

- a vegetation-optimized ensemble, not a balanced leaderboard winner

## Actionable Conclusions

1. Do not over-interpret the old ensemble's `IOU_VEG = 0.7763` as evidence it is a better overall model. It is mostly a vegetation specialist.
2. For current experiments, public score should be forecast as:
   - `expected public ≈ local leaderboard-val - 0.08`
3. The fusion model family remains the strongest direction because it improves the building-related terms that dominate the weighted score.
4. If revisiting ensembles, the goal should not be "maximize vegetation IoU". The goal should be:
   - keep fusion-level building performance
   - borrow only the old ensemble's vegetation strength
5. Any future ensemble should be evaluated with score-part accounting, not just raw IoU, because the metric is building-heavy.

## Bottom Line

The old `weighted_metric_v1` submission got its standout `IOU_VEG` because vegetation is the easiest class, close to binary, and especially favorable to hard-mask channel-wise ensembles. That score did not translate into the best overall leaderboard result because the challenge is dominated by building IoU and building-height RMSE, where the old ensemble was weak.

The current fusion submission family is less spectacular on vegetation but more competitive where the weighted metric actually pays.
