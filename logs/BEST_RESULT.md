
# 🏆 Best Strategy Board

> Pinned tracker for the current best strategy combo on the ESA Embed2Heights leaderboard.
> **Last updated:** 2026-04-22 (late) · **Current best public score:** `0.4209` — 5-way mean ensemble `ECCpDB` packaged as **binary with val-tuned thresholds** `(0.575, 0.525, 0.725)` · **Ensemble val anchor:** `0.4692`
> **Current single-model champion:** `alphaearth_tessera_iou_fusion_N_base48` at val `0.4644` raw / `0.4673` per-class threshold-swept — LightUNet widened to `base_ch=48` on top of J's loss recipe. First architectural move since E to clear the 0.006 noise floor (+0.0076 raw over J). See [N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md).
> **Key empirical lesson from 2026-04-22 A/B:** the threshold bake carries ~`+0.047` of the public lift. Same ensemble shipped as *continuous* scored only `0.3740` (slightly below the prior single-model continuous baseline `0.3750`). Threshold-sweep + binarize is now a **mandatory** final step; continuous submissions are confirmed worse on this leaderboard. See [ENSEMBLE_ECCpDB_REPORT.md](ENSEMBLE_ECCpDB_REPORT.md) Outcome section.
> **Previous single-model champion:** `alphaearth_tessera_iou_fusion_C_presence_centered` at val `0.4437` raw / `0.4491` threshold-swept (base_ch=32, J-era loss off). Retained in experiment log for reference.
>
> **Scoring:** Leaderboard composite with per-class RMSE ceilings `X_bld=3.0m`, `X_veg=5.0m`, positive-only per-image IoU at `label > 0`, per-image RMSE on GT-positive pixels. See [METRIC_PROBE_REPORT.md](METRIC_PROBE_REPORT.md). Earlier entries in the 0.74–0.77 range used the pre-probe metric and are **not comparable** to current numbers.

## 📊 Current Champion

| Dim | Choice | Rationale |
|---|---|---|
| **Data & Target** | AlphaEarth 64ch + Tessera 128ch, no-data mask, fixed 80/20 split (`splits/split.json`) | AlphaEarth remains primary; Tessera is constrained to presence-logit residual correction |
| **Model** | **AlphaEarth LightUNet (base_ch=48) + Tessera residual IoU branch** | Widening LightUNet from 32→48 base channels lifts all five metrics simultaneously (+0.0076 raw over J); capacity knee at 48, 64 saturates. See [N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md) |
| **Head** | **v35head** (presence + base + Δ_b / Δ_v, softplus) with `height_specialist_depth=2` | Best-measured head on LightUNet; specialist depth 2 gives +0.0102 over depth 0 (E), depth 4 regresses IoU. See [HEAD_ARCHITECTURE_NOTES.md](HEAD_ARCHITECTURE_NOTES.md), [K_L_EXPERIMENTS_REPORT.md](K_L_EXPERIMENTS_REPORT.md) |
| **Loss** | `presence_centered` + `veg_height_boost=1.5` + `build_height_boost=5.0` | Presence BCE/Tversky + height boost + aux heights. `build_boost=5` is load-bearing for RMSE_bH (K confirmed). `veg_boost=1.5` is the J compromise between G's 3.0 (hurt RMSE_bH) and 0.0 (higher RMSE_vH). |
| **Height Activation** | `softplus` on base and per-class deltas | Smooth negative-half gradient is load-bearing for RMSE_bH (ablation in head notes) |
| **Fusion** | AlphaEarth primary + compressed Tessera residual presence branch (`hidden_ch=96, depth=2`) | Tessera does not enter height; it only corrects presence / IoU logits |
| **Post & Ensemble** | Per-class threshold sweep `(bld=0.630, veg=0.510, wat=0.850)` | Lifts N from `0.4644` to `0.4673` by recovering water IoU primarily |

**Run:** `runs/alphaearth_tessera_iou_fusion_N_base48/model_best.pth`  ·  config: bs=32, lr=2e-4, aux=1.0, 30 epochs, seed 42, `loss_preset=presence_centered`, `--lightunet-base-ch 48 --tessera-hidden-ch 96 --tessera-hidden-depth 2 --height-specialist-depth 2 --veg-height-boost 1.5`  ·  sbatch: [run_exp_N_base48.sbatch](../run_exp_N_base48.sbatch)

**Score breakdown (val, 405 samples):**

| Metric | N (current champion) | C (prev single-model) | Δ |
|---|---:|---:|---:|
| iou_buildings | `0.5087` | `0.4813` | **+0.0274** |
| iou_trees | `0.7618` | `0.7579` | +0.0039 |
| iou_water | `0.4537` | `0.4070` | **+0.0467** |
| RMSE_building_height | `1.9111 m` | `1.9195 m` | −0.0084 m |
| RMSE_vegetation_height | `3.3963 m` | `3.5336 m` | **−0.1373 m** |
| **Weighted score (raw)** | **`0.4644`** | `0.4437` | **+0.0207** |
| **Weighted score (per-class TS)** | **`0.4673`** | `0.4491` | **+0.0182** |

Per-class threshold-swept val score: `0.4673` with thresholds `(0.630, 0.510, 0.850)`. All five axes improve vs C; no single-metric regression — N satisfies promotion rules cleanly.

---

## 🧪 Per-Dimension Leaderboard

Best known option for each axis.

### Data & Target
- [x] **AlphaEarth + Tessera residual** — current champion input pair, best val 0.4644 raw / 0.4673 TS (N at base_ch=48)
- [x] AlphaEarth — former champion single source, best val 0.4213
- [ ] TerraMind S2 — trails significantly under new metric (see eval table)
- [ ] *idea:* height-tail weighting / oversampling for rare >30m pixels
- [ ] *idea:* stratified split by building density

### Backbone (head = v35head + height_specialist_depth=2)
- [x] **LightUNet base_ch=48 + Tessera residual** (N) — **current champion**, val 0.4644 raw / 0.4673 TS
- [x] LightUNet base_ch=64 + Tessera residual (O) — val 0.4633 raw / 0.4672 TS (saturated; best RMSE_bH on record)
- [x] LightUNet base_ch=32 + Tessera residual (J / E family) — val 0.4568–0.4574 raw; prior single-model baseline
- [x] LightUNet base_ch=32 standalone (AlphaEarth-only) — val 0.4213
- [x] HRNet-W18 — val 0.3997 (−0.0216 vs lightunet at same recipe)
- [ ] EmbeddingRefiner + v35head — **not yet run**; cheapest missing cell
- [ ] *idea:* base_ch=48 + per-class head-width split (keep presence at 48, push height branch to 64) — motivated by O's RMSE_bH global-best (1.8971) while IoU regressed
- [ ] *idea:* HRNet-W32 retuning (lower LR / warmup / stochastic depth) — de-prioritized; widening LightUNet is proven cheaper per score point

### Head (backbone fixed at LightUNet)
- [x] **v35head / MultiTaskPredictionHead** — champion head family (presence / base+Δ / softplus)
- [x] v3head — val 0.4194 (inside 0.006 noise floor; essentially tied)
- [x] softplus (v1/old) — val 0.3861
- [x] v2head — val 0.3731
- [ ] *todo:* multi-seed v35head vs v3head (≥3 seeds) to pick a definitive head

### Loss
- [x] **presence_centered** — current fusion champion; direct presence BCE/Tversky is better than fraction-centered structure losses
- [x] `veg_height_boost=1.5` (J) — Pareto point between G (3.0: +iou_wat, −RMSE_bH) and 0.0 (E)
- [x] `build_height_boost=5` — load-bearing, confirmed by K (3× regressed RMSE_bH); do not lower
- [x] **height_specialist_depth=2** — optimal on base_ch=32 trunk (E: +0.0102). Depth 4 regresses IoU at base_ch=32 (L); worth re-running at base_ch=48 per N_O_REPORT
- [x] `aux_veg_weight=1.0` — M closed at base_ch=32 (aux-veg isolation is a false premise on shared-trunk head); worth re-checking at base_ch=48
- [x] Composite(MAE + SSIM + Grad + Tversky) + aux heads — former AlphaEarth-only champion loss
- [x] **Softplus on height projections** — confirmed load-bearing by nobase/hybrid ablation
- [ ] **Closed for good:** Huber height loss (D), focal IoU any α (F, H), MSE height loss (I), symmetric b/v boost (K)
- [ ] *idea:* focal-Tversky, scale-aware height loss
- [ ] *idea:* log-height auxiliary loss / tail-weighted loss above 30m

### Fusion
- [x] AlphaEarth + Tessera residual IoU — current champion
- [ ] *running:* C loss + larger Tessera stem capacity (`ch16_h64d1`, `ch16_h96d2`)
- [ ] *idea:* concat AlphaEarth + TerraMind S2 (early)
- [ ] *idea:* cross-attn between pixel and ViT-token streams
- [ ] *idea:* late ensemble of LightUNet+v35head and LightUNet+v3head

### Post & Ensemble
- [x] `tools/sweep_thresholds.py` on N (base_ch=48): raw 0.4644 → per-class TS 0.4673 `(0.630, 0.510, 0.850)`
- [x] 5-way mean ensemble `ECCpDB` (pre-N era): val anchor 0.4692, public 0.4209 (binarized). See [ENSEMBLE_ECCpDB_REPORT.md](ENSEMBLE_ECCpDB_REPORT.md)
- [ ] **high priority:** rebuild `ECCpDB`-style ensemble substituting N for weakest member — expected push toward 0.48 given N's clean five-axis win
- [ ] **high priority:** TTA (flip + D4) on N via [predict_tta.py](../predict_tta.py) — near-free +0.005–0.01
- [ ] **high priority:** N at seeds {0, 1, 2} — noise-floor-ceiling check at new champion width
- [ ] *idea:* re-run L / M / G at base_ch=48 — K_L's "closed" verdicts were scoped to 2.75M trunk
- [ ] *idea:* building-mask-gated height regression

---

## 📜 Experiment Log (new metric)

| Date | Run | Δ vs Champion | Score | Notes |
|---|---|---|---|---|
| 2026-04-22 | `alphaearth_tessera_iou_fusion_N_base48` | **champion** | **0.4644** raw / **0.4673** per-class TS | LightUNet base_ch 32→48. All 5 axes improve simultaneously over J. First clean noise-floor-clearing move since E. See [N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md). |
| 2026-04-22 | `alphaearth_tessera_iou_fusion_O_base64` | −0.0011 raw / ~0 TS | 0.4633 raw / 0.4672 TS | base_ch 48→64 saturates. Best RMSE_bH on record (1.8971m), but all 3 IoUs regress vs N. Tie with N under TS. |
| 2026-04-22 | `alphaearth_tessera_iou_fusion_K_classbal_b3v3` | −0.0025 (vs E) | 0.4549 raw | Symmetric 3×/3× weighting. Confirmed 5× building weight is load-bearing; closed. |
| 2026-04-22 | `alphaearth_tessera_iou_fusion_L_specialist_d4` | −0.0027 (vs E) | 0.4547 raw | specialist_depth 2→4. Best RMSE_bH at base_ch=32 (1.8962m) but iou_bld drops 1.6pt. Depth curve saturates at 2 on 32-trunk. |
| 2026-04-22 | `alphaearth_tessera_iou_fusion_M_auxveg2` | −0.0023 (vs E) | 0.4551 raw | aux_veg_weight=2. Closed: aux-veg isolation is false premise on shared-trunk head (see K_L_REPORT). Best iou_tree serendipity (0.7606). |
| 2026-04-22 | `alphaearth_tessera_iou_fusion_J_specialist_d2_veg15` | −0.0006 (vs E) | 0.4568 raw / 0.4614 TS | veg_height_boost tuned 0→1.5 on E. Within noise on raw; pre-widening baseline for N. |
| 2026-04-21 | `alphaearth_tessera_iou_fusion_G_specialist_d2_veg3` | −0.0039 (vs E) | 0.4535 raw | veg_height_boost=3.0. Global-best iou_wat at base_ch=32 (0.4584) but RMSE_bH regresses. |
| 2026-04-21 | `alphaearth_tessera_iou_fusion_E_specialist_d2` | +0.0137 (vs C) | **0.4574 raw** / 0.4615 TS | First promotion past C. height_specialist_depth 0→2 lifts all 5 axes. Prior single-model champion from 04-21 through 04-22 morning. |
| 2026-04-20 | `alphaearth_tessera_iou_fusion_C_presence_centered` | previous champion | 0.4437 raw / 0.4491 TS | Presence-centered loss. Prior single-model champion through 04-21. |
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
- N clears the noise floor against every baseline: +0.0076 raw vs J, +0.0207 raw vs C, +0.0137 vs E. First clean single-knob win since E.
- Capacity plateau: N (base_ch=48, 5.44M params) ≈ O (base_ch=64, 9.20M params) under per-class TS — going wider than 48 trades IoU for RMSE_bH without net gain. 48 is the Pareto point.
- The leaderboard metric's tighter RMSE ceilings (3m / 5m) weight height errors much more heavily than the old 30m placeholder, which is part of why the backbone/head ranking flipped from the 2026-04-15 report.
- Current best is local validation only, not public/private leaderboard.
- Use `model_best.pth` for prediction/evaluation. Inference CLI auto-reads `lightunet_base_ch` from `training_params.json`, so no manual flag needed when predicting with N.
