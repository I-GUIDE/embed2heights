
# 🏆 Best Strategy Board

> Pinned tracker for the current best strategy combo on the ESA Embed2Heights leaderboard.
> **Last updated:** 2026-04-15 · **Current best score:** `0.7641` (val, raw ensemble) · **Submitted:** `—`
>
> **⚠ Scores here predate the 2026-04-17 metric probe** — they use
> `mean(IoU_pos, IoU_neg)` at threshold 0.5 and global pixel-accumulated RMSE.
> The leaderboard uses positive-only per-image IoU at `label > 0` and per-image
> RMSE on class-present pixels ([METRIC_PROBE_REPORT.md](METRIC_PROBE_REPORT.md)).
> Re-evaluate with `evaluate.py` or `tools/sweep_thresholds_and_ensemble.py`
> before promoting a new champion.

## 📊 Current Champion

| Dim | Choice | Rationale |
|---|---|---|
| **Data & Target** | AlphaEarth 64ch, no-data mask, fixed 80/20 split (`splits/split.json`) | Pixel-aligned, strongest single source |
| **Loss** | `ImprovedCompositeLoss` (MAE 1.0 + SSIM 0.5 + Grad 0.5 + Tversky 2.0 + height boost) + aux weight `0.05` | Tversky handles class imbalance; lower aux weight stabilizes multi-head training |
| **Model** | Weighted ensemble: W18 + LightUNet + Refiner | Best raw validation score so far; larger gain than further backbone swapping |
| **Height Head** | `softplus` nonnegative height, no 45m hard cap | Keeps stability while allowing rare tall targets above 45m |
| **Fusion** | Single source (no fusion yet) | AlphaEarth-only backbone is being squeezed before multimodal fusion |
| **Post & Ensemble** | Per-channel weighted prediction average, threshold=0.5 | Raw ensemble reaches 0.7641; threshold calibration proxy reaches about 0.7703 |

**Score breakdown:**

| Metric | Value(local) | Value(submission) |
|---|---:|---:|
| mIoU_building | `0.6145` | \ |
| mIoU_tree | `0.7379` | \ |
| mIoU_water | `0.7001` | \ |
| RMSE_building_height | `3.5587 m` | \ |
| RMSE_vegetation_height | `3.8355 m` | \ |
| **Weighted score** | **`0.7641`** | \ |

**Commit / Run:** uncommitted worktree · in-memory `weighted_metric_v1` from `tools/sweep_thresholds_and_ensemble.py`

---

## 🧪 Per-Dimension Leaderboard

Best known option for each axis (may not yet be combined into the champion).

### Data & Target
- [x] **AlphaEarth** — best raw val 0.7641 with W18/LightUNet/Refiner ensemble; best single model 0.7587 with HRNet-W18
- [ ] TerraMind S2 — val 0.641
- [ ] no-data mask (enabled) — +? vs disabled
- [ ] *idea:* height-tail weighting / oversampling for rare >45m pixels
- [ ] *idea:* stratified split by building density

### Loss
- [x] **Composite(MAE+SSIM+Grad+Tversky) + aux heads** — current
- [x] **Softplus height target/output** — removes 45m hard cap while keeping height nonnegative
- [ ] *idea:* focal-Tversky, scale-aware height loss, uncertainty weighting
- [ ] *idea:* log-height auxiliary loss for high-tail stability

### Model
- [x] **Weighted W18 + LightUNet + Refiner ensemble** — current raw champion, val 0.7641
- [x] **HRNet-W18 multi-head** — current champion, val 0.7587
- [x] **LightUNet multi-head ablation** — val 0.7559; shows most of the old-baseline gain comes from the current head/training code path. Not a perfectly isolated backbone ablation because logged bs/lr/aux differ from W18/Refiner.
- [x] **EmbeddingRefiner multi-head** — strong efficient candidate, val 0.7554
- [x] **HRNet-W32 multi-head** — val 0.7421; needs tuning
- [x] **Old LightUNet report** — val 0.7410
- [x] **EfficientDecoder256** (token inputs)
- [ ] *idea:* W32 with lower LR / warmup / lower aux weight
- [ ] *idea:* SegFormer head, ConvNeXt-UNet, dual-branch decoder

### Fusion
- [ ] *idea:* concat AlphaEarth + Tessera (early)
- [ ] *idea:* cross-attn between pixel and ViT-token streams
- [ ] *idea:* per-channel expert (building ← best model, height ← best model)
- [ ] *idea:* late fusion / ensemble HRNet-W18 + LightUNet/EmbeddingRefiner

### Post & Ensemble
- [x] **weighted avg of top-3 per channel** — raw val 0.7641
- [x] **per-class threshold sweep proxy** — best observed ~0.7703 with thresholds around bld=0.575, veg=0.900, water=0.900
- [ ] *todo:* materialize ensemble predictions and run standard `evaluate.py`
- [ ] *todo:* convert threshold sweep into real prediction calibration / hard-threshold post-processing
- [ ] *idea:* TTA (flip/rot)
- [ ] *idea:* building-mask-gated height regression

---

## 📜 Experiment Log

| Date | Run | Δ vs Champion | Score | Notes |
|---|---|---|---|---|
| 2026-04-15 | `weighted_metric_v1` in-memory ensemble | champion | **0.7641** | Per-channel W18/LightUNet/Refiner weighted average at threshold 0.5. |
| 2026-04-15 | `weighted_metric_v1` + threshold sweep proxy | +0.0062 vs raw champion | **~0.7703** | Prediction thresholds varied with GT fixed at 0.5; needs materialized post-processing before promotion. |
| 2026-04-15 | `alphaearth_hrnet_w18_softplus_bs16_lr1e4_aux005` | -0.0054 | **0.7587** | Best single model. Strongest RMSE: building 3.596m, vegetation 3.948m. |
| 2026-04-15 | `lightunet_alphaearth` | -0.0082 | **0.7559** | LightUNet with current multi-head code path. Best building mIoU 0.6110 and water mIoU 0.6885 among single models. |
| 2026-04-15 | `alphaearth_refiner_softplus_bs16_lr1e4_aux005` | -0.0087 | **0.7554** | Best efficiency/score tradeoff. Best vegetation RMSE 3.8458m. |
| 2026-04-15 | `alphaearth_hrnet_w32_softplus_bs16_lr5e5_aux005` | -0.0220 | **0.7421** | Slightly above old LightUNet, but under-tuned for its capacity. |
| 2026-04-14 | Old AlphaEarth LightUNet report | -0.0231 | **0.7410** | Original baseline number from `logs/BASELINE_REPORT.md`; not the current `lightunet_alphaearth` ablation. |

---

## 📝 Promotion Rules

A candidate becomes the new champion when:
- Val score ≥ champion + 0.005, **and**
- No single metric drops > 0.02 (avoid over-fitting one axis)
- Run is reproducible: split file + seed + commit SHA recorded

---

## 🔎 Notes

- Detailed backbone report: `logs/ALPHAEARTH_BACKBONE_REPORT.md`
- The LightUNet ablation changes the interpretation: the current head/training code path accounts for most of the gain over the old 0.7410 baseline; HRNet-W18 is +0.0028 over this ablation, but this should not be over-interpreted as pure architecture gain because LightUNet used different logged hyperparameters.
- Stricter next ablation: rerun LightUNet with bs=16, lr=1e-4, aux weight=0.05.
- `weighted_metric_v1` is currently computed in memory by `tools/sweep_thresholds_and_ensemble.py`; materialize predictions before treating it as a run artifact.
- Threshold sweep score is a calibration proxy: prediction thresholds vary while GT masks stay fixed at 0.5.
- Current best is local validation only, not public/private leaderboard.
- Evaluation uses threshold `0.5`; per-class threshold sweep is a high-priority next step.
- Use `model_best.pth` for prediction/evaluation.
