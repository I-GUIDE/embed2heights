# Metric Probe Report — 2026-04-17

Official metric formulas are not documented (Rules page, Data page, Starter Pack notebook, and upstream baseline repo all list only the 5 metric names + weights). We reverse-engineered them by submitting a dummy.

## Test

Submit all-zero predictions for all 946 test patches (4 channels: building/veg/water/height).

Tools: [tools/make_dummy_submission.py](../tools/make_dummy_submission.py), [tools/predict_dummy_metrics.py](../tools/predict_dummy_metrics.py).

## Leaderboard returned

```
IOU_BUILD     0.0212
IOU_VEG       0
IOU_WATER     0.1788
RMSE_H_BUILD  4.0760
RMSE_H_VEG    10.9264
SCORE         0.032124
```

## Conclusions

1. **IoU = per-image positive-only Jaccard, empty/empty → 1.0.** Equivalent to `sklearn.metrics.jaccard_score(zero_division=1.0)`. For an all-zero prediction, the returned value equals the fraction of patches where GT is also empty — 2.12% for building, 0% for veg, 17.88% for water. No other formula reproduces these values.

2. **Label binarization threshold ≈ 0**, not 0.5. Threshold 0.5 would make building empty in ~80% of patches (iou_build ≈ 0.8); we got 0.02, only consistent with "any non-zero fraction counts as positive".

3. **RMSE = per-image macro, GT-masked.** Returned 4.08 / 10.93 match our R3/R4 predictions (per-image, 3.57–4.73 for building, 10.36–11.18 for veg). Rules out R1/R2 (global pixel-accumulated: 7.34–12.01).

4. **Score formula (IoU part is exact):**
   ```
   0.25 × 0.0212 + 0.15 × 0 + 0.15 × 0.1788 = 0.03212 ≈ returned 0.0321 ✓
   ```
   RMSE contribution was 0, confirming `max(0, 1 - RMSE/X)` with `X_building = 3.0m` and `X_vegetation = 5.0m` (values pinned down by a follow-up height probe and now codified in [core/metrics.py](../core/metrics.py) `RMSE_NORMALIZATION`). The 30m ceiling we assumed before is wrong.

5. **Our `evaluate.py` was measuring the wrong metric.** Used `mean(IoU_pos, IoU_neg)` at threshold 0.5, with global pixel-accumulated RMSE. Inflated sparse-class IoU by ~0.25–0.35.

## Impact on evaluate.py

Same model (`alphaearth_hrnet_w18_softplus_bs16_lr1e4_aux005`), same 405 val patches:

| Metric | Before | After |
|---|---:|---:|
| iou_bld | 0.598 | **0.267** |
| iou_tree | 0.724 | 0.729 |
| iou_wat | 0.680 | **0.389** |
| RMSE_bH | 4.83m | **2.04m** |
| RMSE_vH | 4.33m | **3.60m** |

Fixed in this commit:
- Positive-only IoU, empty/empty → 1.0
- `pred > 0.5` (--pred-threshold), `label > 0` (--label-threshold)
- Per-image RMSE averaging
- Per-class RMSE normalization `X_building = 3.0m`, `X_vegetation = 5.0m` (pinned by the follow-up height probe; matches `RMSE_NORMALIZATION` in [core/metrics.py](../core/metrics.py)).

## Still unknown

- **Server-side `pred_threshold`** — all-zero probe can't disambiguate. Assuming 0.5.
- **RMSE label threshold** — probably 0 (matches IoU), but 4.08 is closer to R3 (threshold 0.5 → 4.73) than R4 (threshold 0 → 3.57). Could be train/test distribution shift.

## Reproduce

```bash
python tools/make_dummy_submission.py                   # all-zero probe
python tools/predict_dummy_metrics.py                   # candidate-value table
```

The height probe (`--height-value 5`) has been run and confirmed `X_building = 3.0m`, `X_vegetation = 5.0m`.

## References

- Challenge: [platform.ai4eo.eu/geoai](https://platform.ai4eo.eu/geoai)
- Upstream baseline: [VMarsocci/emb2heights-baselines](https://github.com/VMarsocci/emb2heights-baselines)
- Corrected metric spec: [GOAL.md](GOAL.md)
- Label distribution: [LABEL_BAND_ANALYSIS.md](LABEL_BAND_ANALYSIS.md)
