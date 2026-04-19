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
   RMSE contribution was 0, meaning `max(0, 1 - RMSE/X)` with `X_building < 4m` and `X_vegetation < 10.9m`. The 30m ceiling we assumed before is wrong.

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
- `MAX_HEIGHT = 30` kept as placeholder with a clear comment — real X_class is unknown; absolute score over-estimates, but IoU portion and model ranking are correct.

## Still unknown

- **X_building and X_vegetation** in the RMSE normalization. Only bounded: `X_bld < 4m`, `X_veg < 10.9m`.
- **Server-side `pred_threshold`** — all-zero probe can't disambiguate. Assuming 0.5.
- **RMSE label threshold** — probably 0 (matches IoU), but 4.08 is closer to R3 (threshold 0.5 → 4.73) than R4 (threshold 0 → 3.57). Could be train/test distribution shift.

A second probe (`--height-value 5`) would pin X down.

## Reproduce

```bash
python tools/make_dummy_submission.py                   # all-zero probe
python tools/predict_dummy_metrics.py                   # candidate-value table

# follow-up probe to pin X_class
python tools/make_dummy_submission.py \
    --class-value 0 --height-value 5 \
    --output-dir runs/dummy_h5/predictions \
    --zip-path   runs/dummy_h5/submission.zip
```

## References

- Challenge: [platform.ai4eo.eu/geoai](https://platform.ai4eo.eu/geoai)
- Upstream baseline: [VMarsocci/emb2heights-baselines](https://github.com/VMarsocci/emb2heights-baselines)
- Corrected metric spec: [GOAL.md](GOAL.md)
- Label distribution: [LABEL_BAND_ANALYSIS.md](LABEL_BAND_ANALYSIS.md)
