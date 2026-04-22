
# 🏆 Best Strategy Board

> Pinned tracker for the current best strategy combo on the ESA Embed2Heights leaderboard.
> **Last updated:** 2026-04-22 · **Current best val score:** `0.4692` (5-way mean ensemble `ECCpDB`, tuned thresholds) · **Current best public score:** `0.4209` (binarized ECCpDB zip, submitted 2026-04-22) · See [ENSEMBLE_ECCpDB_REPORT.md](ENSEMBLE_ECCpDB_REPORT.md)
> **Previous single-model champion:** `alphaearth_tessera_iou_fusion_C_presence_centered` at val `0.4437` raw / `0.4491` threshold-swept. Retained below as per-dimension reference.
>
> **Scoring:** Leaderboard composite with per-class RMSE ceilings `X_bld=3.0m`, `X_veg=5.0m`, positive-only per-image IoU at `label > 0`, per-image RMSE on GT-positive pixels. See [METRIC_PROBE_REPORT.md](METRIC_PROBE_REPORT.md). Earlier entries in the 0.74–0.77 range used the pre-probe metric and are **not comparable** to current numbers.

## 📊 Current Champion

| Dim | Choice | Rationale |
|---|---|---|
| **Data & Target** | AlphaEarth 64ch + Tessera 128ch, no-data mask, fixed 80/20 split (`splits/split.json`) | AlphaEarth remains primary; Tessera is constrained to presence-logit residual correction |
| **Model** | **AlphaEarth LightUNet + Tessera residual IoU branch** | C loss makes Tessera useful as a constrained presence-logit residual; +0.0224 raw over old AlphaEarth-only champion |
| **Head** | **v35head** (presence + base + Δ_b / Δ_v, softplus; fraction head aux only) | Best-measured head on LightUNet; see [ALPHAEARTH_HEAD_REPORT.md](ALPHAEARTH_HEAD_REPORT.md), [HEAD_ARCHITECTURE_NOTES.md](HEAD_ARCHITECTURE_NOTES.md) |
| **Loss** | `presence_centered`: presence BCE + presence Tversky + height boost + aux height + weak fraction MAE | Aligns segmentation supervision with submitted presence logits, so Tessera residual branch receives IoU-aligned gradients |
| **Height Activation** | `softplus` on base and per-class deltas | Smooth negative-half gradient is load-bearing for RMSE_bH (ablation in head notes) |
| **Fusion** | AlphaEarth primary + compressed Tessera residual presence branch | Tessera does not enter height; it only corrects presence / IoU logits |
| **Post & Ensemble** | Raw threshold 0.5; best val sweep `(bld=0.600, veg=0.490, wat=0.940)` | Threshold sweep lifts score from `0.4437` to `0.4491` by recovering water IoU |

**Run:** `runs/alphaearth_tessera_iou_fusion_C_presence_centered/model_best.pth`  ·  config: bs=32, lr=2e-4, aux=1.0, 30 epochs, seed 42, `loss_preset=presence_centered`

**Score breakdown (val, 405 samples):**

| Metric | Value (local) | Value (submission) |
|---|---:|---:|
| iou_buildings | `0.4813` | — |
| iou_trees | `0.7579` | — |
| iou_water | `0.4070` | — |
| RMSE_building_height | `1.9195 m` | — |
| RMSE_vegetation_height | `3.5336 m` | — |
| **Weighted score** | **`0.4437`** | — |

Threshold-swept val score: `0.4491` with thresholds `(0.600, 0.490, 0.940)`.

---

## 🧪 Per-Dimension Leaderboard

Best known option for each axis.

### Data & Target
- [x] **AlphaEarth + Tessera residual** — current champion input pair, best val 0.4437 raw / 0.4491 threshold-swept
- [x] AlphaEarth — former champion single source, best val 0.4213
- [ ] TerraMind S2 — trails significantly under new metric (see eval table)
- [ ] *idea:* height-tail weighting / oversampling for rare >30m pixels
- [ ] *idea:* stratified split by building density

### Backbone (head fixed at v35head)
- [x] **AlphaEarth LightUNet + Tessera residual branch** — current champion, val 0.4437 raw / 0.4491 threshold-swept
- [x] LightUNet — former AlphaEarth-only champion, val 0.4213
- [x] HRNet-W18 — val 0.3997 (−0.0216)
- [ ] EmbeddingRefiner + v35head — **not yet run**; cheapest missing cell
- [ ] *idea:* HRNet-W32 retuning (lower LR / warmup / stochastic depth)
- [ ] *idea:* unified-hyperparameter backbone rerun for pure architecture delta

### Head (backbone fixed at LightUNet)
- [x] **v35head / MultiTaskPredictionHead** — champion head family (presence / base+Δ / softplus)
- [x] v3head — val 0.4194 (inside 0.006 noise floor; essentially tied)
- [x] softplus (v1/old) — val 0.3861
- [x] v2head — val 0.3731
- [ ] *todo:* multi-seed v35head vs v3head (≥3 seeds) to pick a definitive head

### Loss
- [x] **presence_centered** — current fusion champion; direct presence BCE/Tversky is better than fraction-centered structure losses
- [x] Composite(MAE + SSIM + Grad + Tversky) + aux heads — former AlphaEarth-only champion loss
- [x] **Softplus on height projections** — confirmed load-bearing by nobase/hybrid ablation
- [ ] *idea:* focal-Tversky, scale-aware height loss
- [ ] *idea:* log-height auxiliary loss / tail-weighted loss above 30m

### Fusion
- [x] AlphaEarth + Tessera residual IoU — current champion
- [ ] *running:* C loss + larger Tessera stem capacity (`ch16_h64d1`, `ch16_h96d2`)
- [ ] *idea:* concat AlphaEarth + TerraMind S2 (early)
- [ ] *idea:* cross-attn between pixel and ViT-token streams
- [ ] *idea:* late ensemble of LightUNet+v35head and LightUNet+v3head

### Post & Ensemble
- [x] `tools/sweep_thresholds.py` on C fusion: raw 0.4437 -> threshold-swept 0.4491
- [ ] *idea:* TTA (flip / rot) on v35head checkpoints
- [ ] *idea:* building-mask-gated height regression

---

## 📜 Experiment Log (new metric)

| Date | Run | Δ vs Champion | Score | Notes |
|---|---|---|---|---|
| 2026-04-20 | `alphaearth_tessera_iou_fusion_C_presence_centered` | champion | **0.4437** raw / **0.4491** threshold-swept | Presence-centered loss; building IoU 0.4813 at 0.5. |
| 2026-04-20 | `alphaearth_tessera_iou_residual` | −0.0101 raw | 0.4336 | Previous best fusion; old fraction-centered loss. |
| 2026-04-20 | `alphaearth_tessera_iou_fusion_B_no_ssim_grad` | −0.0129 raw | 0.4308 | Removing SSIM/Grad alone helped less than moving Tversky to presence. |
| 2026-04-19 | `lightunet_v35head` | champion | **0.4213** | LightUNet + v35head on AlphaEarth. |
| 2026-04-19 | `lightunet_alphaearth_v3head` | −0.0019 | 0.4194 | Inside 0.006 noise floor; essentially tied with v35head. |
| 2026-04-19 | `baseline_v35_alphaearth` | −0.0062 | 0.4151 | Identical config to `lightunet_v35head` except `num_workers`; gives noise-floor estimate. |
| 2026-04-19 | `alphaearth_hrnet_w18_v3head` | −0.0162 | 0.4051 | Best HRNet-W18 result. |
| 2026-04-19 | `hrnet_w18_v35head` | −0.0216 | 0.3997 | Confirms LightUNet > HRNet-W18 under v35head. |
| 2026-04-19 | `lightunet_alphaearth` | −0.0352 | 0.3861 | LightUNet + old softplus head. |
| 2026-04-19 | `alphaearth_refiner_softplus_bs16_lr1e4_aux005` | −0.0390 | 0.3823 | EmbeddingRefiner, old head. |
| 2026-04-19 | `lightunet_v2head` | −0.0482 | 0.3731 | LightUNet + v2head. |
| 2026-04-19 | `alphaearth_hrnet_w18_softplus_bs16_lr1e4_aux005` | −0.0503 | 0.3710 | HRNet-W18, old head. |
| 2026-04-19 | `hrnet_w18_v2head` | −0.0519 | 0.3694 | HRNet-W18 + v2head. |
| 2026-04-19 | `alphaearth_hrnet_w32_softplus_bs16_lr5e5_aux005` | −0.0649 | 0.3564 | HRNet-W32 under-tuned. |

### Archived (pre-probe metric)

The following entries used `mean(IoU_pos, IoU_neg)` at threshold 0.5 with global pixel-accumulated RMSE and `X=30` placeholder normalization. **Not comparable** to current scores.

| Date | Run | Score (old metric) |
|---|---|---:|
| 2026-04-15 | `weighted_metric_v1` in-memory ensemble (W18 + LightUNet + Refiner) | 0.7641 |
| 2026-04-15 | `alphaearth_hrnet_w18_softplus...` | 0.7587 |
| 2026-04-15 | `lightunet_alphaearth` | 0.7559 |
| 2026-04-15 | `alphaearth_refiner_softplus...` | 0.7554 |
| 2026-04-15 | `alphaearth_hrnet_w32_softplus...` | 0.7421 |
| 2026-04-14 | Old AlphaEarth LightUNet report | 0.7410 |

---

## 📝 Promotion Rules

A candidate becomes the new champion when:
- Val score ≥ champion + 0.005 (≥ 0.006 noise floor), **and**
- No single metric drops > 0.02 (avoid over-fitting one axis), **and**
- Run is reproducible: split file + seed + commit SHA recorded.

---

## 🔎 Notes

- Noise floor on val score ≈ **0.006** (measured from two seed-42 `v35head` runs differing only in `num_workers`).
- Current champion is v35head + LightUNet, but v3head is inside the noise floor — multi-seed confirmation needed before declaring a definitive head.
- The leaderboard metric's tighter RMSE ceilings (3m / 5m) weight height errors much more heavily than the old 30m placeholder, which is part of why the backbone/head ranking flipped from the 2026-04-15 report.
- Current best is local validation only, not public/private leaderboard.
- Evaluation uses threshold `0.5`; per-class sweep under the new metric is a high-priority next step.
- Use `model_best.pth` for prediction/evaluation.
