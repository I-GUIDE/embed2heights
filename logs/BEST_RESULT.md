
# 🏆 Best Strategy Board

> Pinned tracker for the current best strategy combo on the ESA Embed2Heights leaderboard.
> **Last updated:** 2026-05-02 · **Current best public score:** `0.4282` — 4-way mean ensemble `ensemble_Q_Qpretrain_N_Qterramind_s2_mean` packaged as **binary with val-tuned thresholds** `(0.620, 0.530, 0.690)`. Public breakdown: IOU_BUILD `0.4319`, IOU_VEG `0.8146`, IOU_WATER `0.4497`, RMSE_H_BUILD `2.1156`, RMSE_H_VEG `3.5781`. Submission artifact: `submission/ensemble_Q_Qpretrain_N_Qterramind_s2_mean_binary_062_053_069.zip`.
> **Current best local validation run:** `alphaearth_tessera_iou_fusion_Y_gated_rich_tied_3way` at `0.5107` raw / **`0.5145` per-class TS** with thresholds `(0.570, 0.580, 0.900)` — rich gated AlphaEarth+Tessera fusion with split 3-way presence heads.
> **Previous best local validation ensemble:** `ensemble_Q_Qpretrain_N_Qterramind_s2_mean` at `0.4813` raw / **`0.4846` per-class TS** with thresholds `(0.620, 0.530, 0.690)` — 4-way mean of Q, Q-pretrain, N, and Q+TerraMind-S2. Public score: **`0.4282`**. See [ENSEMBLE_Q_QPRETRAIN_N_QTERRAMIND_REPORT.md](ENSEMBLE_Q_QPRETRAIN_N_QTERRAMIND_REPORT.md).
> **Previous raw single-model champion:** `alphaearth_tessera_iou_fusion_N_base48` at val `0.4644` raw / `0.4673` per-class threshold-swept — LightUNet widened to `base_ch=48` on top of J's loss recipe. See [N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md).
> **Previous threshold-swept single-model best:** `alphaearth_tessera_iou_fusion_Q_base48_fused_height_gate_pretrain_m55` at val `0.4710` raw / **`0.4745` per-class TS** — Q initialized from train+test self-supervised AlphaEarth+Tessera pretraining.
> **Key empirical lesson from 2026-04-22 A/B:** the threshold bake carries ~`+0.047` of the public lift. Same ensemble shipped as *continuous* scored only `0.3740` (slightly below the prior single-model continuous baseline `0.3750`). Threshold-sweep + binarize is now a **mandatory** final step; continuous submissions are confirmed worse on this leaderboard. See [ENSEMBLE_ECCpDB_REPORT.md](ENSEMBLE_ECCpDB_REPORT.md) Outcome section.
> **Previous single-model champion:** `alphaearth_tessera_iou_fusion_C_presence_centered` at val `0.4437` raw / `0.4491` threshold-swept (base_ch=32, J-era loss off). Retained in experiment log for reference.
>
> **Scoring:** Leaderboard composite with per-class RMSE ceilings `X_bld=3.0m`, `X_veg=5.0m`, positive-only per-image IoU at `label > 0`, per-image RMSE on GT-positive pixels. See [METRIC_PROBE_REPORT.md](METRIC_PROBE_REPORT.md). Earlier entries in the 0.74–0.77 range used the pre-probe metric and are **not comparable** to current numbers.

## 📊 Current Champion

| Dim | Choice | Rationale |
|---|---|---|
| **Data & Target** | AlphaEarth 64ch + Tessera 128ch, no-data mask, fixed 80/20 split (`splits/split.json`) | AlphaEarth remains primary; Tessera now contributes through rich gated fusion rather than residual-only presence correction. |
| **Model** | **AlphaEarth LightUNet (base_ch=48) + rich gated Tessera fusion** | Keeps the proven base_ch=48 trunk, but replaces the older residual IoU branch with rich gated cross-modality fusion (`gate_mode=rich`, tied gates, bias 4.0). |
| **Head** | **Split 3-way presence head + v35-style height specialists** (`height_specialist_depth=2`) | `tessera_iou_fusion_gated_presence_3way` uses independent building / tree / water presence heads while retaining the base + Δ_b / Δ_v height structure. |
| **Loss** | `presence_centered` + `veg_height_boost=1.5` + `build_height_boost=5.0` | Presence BCE/Tversky + height boost + aux heights. `build_boost=5` is load-bearing for RMSE_bH (K confirmed). `veg_boost=1.5` is the J compromise between G's 3.0 (hurt RMSE_bH) and 0.0 (higher RMSE_vH). |
| **Height Activation** | `softplus` on base and per-class deltas | Smooth negative-half gradient is load-bearing for RMSE_bH (ablation in head notes) |
| **Height Routing** | AlphaEarth height gate with rich fusion feeding shared features | Large RMSE gains vs N: building `1.9111→1.6684`, vegetation `3.3963→2.9740`. |
| **Fusion** | AlphaEarth primary + Tessera gated branch (`hidden_ch=96, depth=2`) | Rich tied gates with no modality dropout; stronger than residual-only N/Q family on all score axes. |
| **Post & Ensemble** | Per-class threshold sweep `(bld=0.570, veg=0.580, wat=0.900)` | Lifts Y-3way from `0.5107` to `0.5145`, mostly via water IoU `0.4730→0.4949`. |

**Run:** `runs/alphaearth_tessera_iou_fusion_Y_gated_rich_tied_3way/model_best.pth`  ·  config: bs=32, lr=2e-4, aux=1.0, 30 epochs, seed 42, `loss_preset=presence_centered`, `--model-type tessera_iou_fusion_gated_presence_3way --lightunet-base-ch 48 --tessera-hidden-ch 96 --tessera-hidden-depth 2 --height-specialist-depth 2 --veg-height-boost 1.5 --gate-mode rich --gate-init-bias 4.0 --modality-dropout 0.0 --tessera-presence-ch 0`  ·  sbatch: [run_exp_Y_gated_rich_tied_3way.sbatch](../runs/alphaearth_tessera_iou_fusion_Y_gated_rich_tied_3way/job_logs/run_exp_Y_gated_rich_tied_3way.sbatch)

**Score breakdown (val, 405 samples):**

| Metric | Y gated rich tied 3way (current champion) | N (previous raw champion) | Δ |
|---|---:|---:|---:|
| iou_buildings | `0.5313` | `0.5087` | **+0.0226** |
| iou_trees | `0.7664` | `0.7618` | +0.0046 |
| iou_water | `0.4730` | `0.4537` | **+0.0193** |
| RMSE_building_height | `1.6684 m` | `1.9111 m` | **−0.2427 m** |
| RMSE_vegetation_height | `2.9740 m` | `3.3963 m` | **−0.4223 m** |
| **Weighted score (raw)** | **`0.5107`** | `0.4644` | **+0.0463** |
| **Weighted score (per-class TS)** | **`0.5145`** | `0.4673` | **+0.0472** |

Per-class threshold-swept val score: `0.5145` with thresholds `(0.570, 0.580, 0.900)`. All five axes improve vs N; Y-3way clears the promotion bar cleanly.

---

## 🧪 Per-Dimension Leaderboard

Best known option for each axis.

### Data & Target
- [x] **AlphaEarth + Tessera gated fusion** — current champion input pair, best val 0.5107 raw / 0.5145 TS (Y gated rich tied 3way)
- [x] AlphaEarth + Tessera residual — previous champion input pair, best val 0.4644 raw / 0.4673 TS (N at base_ch=48)
- [x] **AlphaEarth + Tessera self-supervised pretrain** — train+test masked cross-reconstruction gives Q a moderate lift: `0.4646 -> 0.4710` raw and `0.4712 -> 0.4745` TS. See [PRETRAIN_AE_TESSERA_REPORT.md](PRETRAIN_AE_TESSERA_REPORT.md).
- [x] AlphaEarth — former champion single source, best val 0.4213
- [ ] TerraMind S2 — trails significantly under new metric (see eval table)
- [ ] *idea:* height-tail weighting / oversampling for rare >30m pixels
- [ ] *idea:* stratified split by building density

### Backbone (head = v35head + height_specialist_depth=2)
- [x] **LightUNet base_ch=48 + rich gated Tessera fusion** (Y gated rich tied 3way) — **current champion**, val 0.5107 raw / 0.5145 TS
- [x] LightUNet base_ch=48 + Tessera residual (N) — previous raw champion, val 0.4644 raw / 0.4673 TS
- [x] LightUNet base_ch=64 + Tessera residual (O) — val 0.4633 raw / 0.4672 TS (saturated; best RMSE_bH on record)
- [x] LightUNet base_ch=32 + Tessera residual (J / E family) — val 0.4568–0.4574 raw; prior single-model baseline
- [x] LightUNet base_ch=32 standalone (AlphaEarth-only) — val 0.4213
- [x] HRNet-W18 — val 0.3997 (−0.0216 vs lightunet at same recipe)
- [ ] EmbeddingRefiner + v35head — **not yet run**; cheapest missing cell
- [ ] *idea:* base_ch=48 + per-class head-width split (keep presence at 48, push height branch to 64) — motivated by O's RMSE_bH global-best (1.8971) while IoU regressed
- [ ] *idea:* HRNet-W32 retuning (lower LR / warmup / stochastic depth) — de-prioritized; widening LightUNet is proven cheaper per score point

### Head (backbone fixed at LightUNet)
- [x] **Split 3-way presence head + v35-style height specialists** — current champion head family in `tessera_iou_fusion_gated_presence_3way`
- [x] v35head / MultiTaskPredictionHead — previous champion head family (presence / base+Δ / softplus)
- [x] Q fused height gate + AE/Tessera pretrain — previous best single-model per-class TS (`0.4745`), mostly by lifting building IoU and height RMSE. The non-pretrained Q remains the cleaner architecture ablation (`0.4712` TS). See [PRETRAIN_AE_TESSERA_REPORT.md](PRETRAIN_AE_TESSERA_REPORT.md).
- [x] Q fused height gate — non-pretrained ablation (`0.4712` TS), improves RMSE_vH (`3.3963→3.3286`) but raw IoU calibration shifts; ensemble candidate
- [x] R independent height h96d4 — closed in this form: extreme height tail improves, common bins and water IoU regress; see [Q_R_HEIGHT_ROUTING_REPORT.md](Q_R_HEIGHT_ROUTING_REPORT.md)
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
- [x] **AlphaEarth + Tessera rich gated 3-way presence fusion** — current champion (`0.5107` raw / `0.5145` TS)
- [x] AlphaEarth + Tessera residual IoU — previous champion family
- [x] Fused Tessera presence for height routing (Q) — useful when threshold-swept; not a raw replacement for N
- [ ] *running:* C loss + larger Tessera stem capacity (`ch16_h64d1`, `ch16_h96d2`)
- [ ] *idea:* concat AlphaEarth + TerraMind S2 (early)
- [ ] *idea:* cross-attn between pixel and ViT-token streams
- [ ] *idea:* late ensemble of LightUNet+v35head and LightUNet+v3head

### Post & Ensemble
- [x] `tools/sweep_thresholds.py` on Y gated rich tied 3way: raw 0.5107 → per-class TS 0.5145 `(0.570, 0.580, 0.900)`
- [x] 4-way mean ensemble `Q + Q_pretrain + N + Q_TerraMind-S2` — previous best local ensemble: `0.4813` raw / **`0.4846` per-class TS** local val, **`0.4282` public** with thresholds `(0.620, 0.530, 0.690)`. Val output: `runs/ensemble_Q_Qpretrain_N_Qterramind_s2_mean/predictions`; test output: `runs/ensemble_Q_Qpretrain_N_Qterramind_s2_mean/test_predictions`; submission zip: `submission/ensemble_Q_Qpretrain_N_Qterramind_s2_mean_binary_062_053_069.zip`. See [ENSEMBLE_Q_QPRETRAIN_N_QTERRAMIND_REPORT.md](ENSEMBLE_Q_QPRETRAIN_N_QTERRAMIND_REPORT.md).
- [x] `tools/sweep_thresholds.py` on N (base_ch=48): raw 0.4644 → per-class TS 0.4673 `(0.630, 0.510, 0.850)`
- [x] **Water connected-component empty-patch filter probe** — after water thresholding, clear the whole water mask if the largest predicted water component is too small. On val this lifts water IoU by `+0.0174` for N (`K=16`), `+0.0137` for Q (`K=8`), and `+0.0182` for `xfusion_005` (`K=16`), worth about `+0.0021` to `+0.0027` total score before ensemble retuning. See [XFUSION_CROSSLEVEL_PLAN.md §Water Connected-Component Postprocess Probe](XFUSION_CROSSLEVEL_PLAN.md#water-connected-component-postprocess-probe).
- [x] 5-way mean ensemble `ECCpDB` (pre-N era): val anchor 0.4692, public 0.4209 (binarized). See [ENSEMBLE_ECCpDB_REPORT.md](ENSEMBLE_ECCpDB_REPORT.md)
- [x] rebuild `ECCpDB`-style ensemble with newer Q/Q-pretrain/N/TerraMind members — confirmed push to 0.4846 local val TS
- [ ] Re-sweep final ensemble over `(water_threshold, largest-water-component K)` before the next binary submission; start with `K=8` and `K=16`.
- [ ] TTA (flip + D4) on N via [predict.py](../predict.py) `--tta d4` — near-free +0.005–0.01
- [ ] N at seeds {0, 1, 2} — noise-floor-ceiling check at new champion width
- [ ] *idea:* re-run L / M / G at base_ch=48 — K_L's "closed" verdicts were scoped to 2.75M trunk
- [ ] *idea:* building-mask-gated height regression

---

## 📜 Experiment Log (new metric)

| Date | Run | Δ vs Champion | Score | Notes |
|---|---|---|---|---|
| 2026-05-02 | `alphaearth_tessera_iou_fusion_Y_gated_rich_tied_3way` | **new champion**; +0.0463 raw / +0.0472 TS vs N; +0.0261 raw / +0.0299 TS vs prior best local ensemble | **0.5107 raw / 0.5145 TS** | Rich gated AlphaEarth+Tessera fusion with tied gates and split 3-way presence heads. Metrics: iou_bld 0.5313, iou_tree 0.7664, iou_wat 0.4730, RMSE_bH 1.6684, RMSE_vH 2.9740. Per-class thresholds `(0.570, 0.580, 0.900)` lift water IoU to 0.4949. |
| 2026-04-25 | `ensemble_Q_Qpretrain_N_Qterramind_s2_mean` | +0.0101 TS vs Q-pretrain single-model TS; +0.0173 TS vs N; **+0.0073 public vs ECCpDB** | **0.4813 raw / 0.4846 TS val; 0.4282 public** | 4-way mean of Q, Q-pretrain, N, and `alphaearth_tessera_Q_shared_probe_terramind_s2`; per-class thresholds `(0.620, 0.530, 0.690)`. Public breakdown: IOU_BUILD 0.4319, IOU_VEG 0.8146, IOU_WATER 0.4497, RMSE_H_BUILD 2.1156, RMSE_H_VEG 3.5781. `s1s2_fusion` control reached 0.4834 TS, so S2 is the better fourth member. See [ENSEMBLE_Q_QPRETRAIN_N_QTERRAMIND_REPORT.md](ENSEMBLE_Q_QPRETRAIN_N_QTERRAMIND_REPORT.md). |
| 2026-04-25 | `alphaearth_tessera_iou_fusion_Q_base48_fused_height_gate_pretrain_m55` | +0.0066 raw / **+0.0072 TS vs N**; +0.0064 raw / +0.0033 TS vs Q | **0.4710 raw / 0.4745 TS** | Q initialized from `pretrain_ae_tessera_train_test_m55_b16_base48/pretrain_best.pth`. Positive but moderate: improves building IoU and both height RMSEs, slightly regresses water IoU. Former best TS single model before Y-3way; not a clean raw champion replacement for N because it uses transductive self-supervised pretraining and needs threshold tuning. See [PRETRAIN_AE_TESSERA_REPORT.md](PRETRAIN_AE_TESSERA_REPORT.md). |
| 2026-04-25 | `alphaearth_tessera_iou_fusion_Q_base48_fused_height_gate` | +0.0002 raw / **+0.0039 TS vs N** | 0.4646 raw / **0.4712 TS** | Height routing uses fused `presence_logits` instead of Alpha-only gate. RMSE_vH improves 3.3963→3.3286, but raw IoUs drop due to calibration shift; threshold sweep recovers. See [Q_R_HEIGHT_ROUTING_REPORT.md](Q_R_HEIGHT_ROUTING_REPORT.md). |
| 2026-04-25 | `alphaearth_tessera_iou_fusion_R_base48_indep_height_h96d4` | −0.0080 raw / −0.0059 TS vs N | 0.4564 raw / 0.4614 TS | Independent base/building/veg height trunks (`h96d4`) improve only extreme height tail; common height bins and water IoU regress. Closed in current form. |
| 2026-04-22 | `alphaearth_tessera_iou_fusion_N_base48` | previous raw champion | **0.4644** raw / **0.4673** per-class TS | LightUNet base_ch 32→48. All 5 axes improve simultaneously over J. First clean noise-floor-clearing move since E. See [N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md). |
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
- Y-3way clears the noise floor against every local baseline: +0.0463 raw vs N and +0.0261 raw vs the previous best local ensemble.
- Capacity plateau: N (base_ch=48, 5.44M params) ≈ O (base_ch=64, 9.20M params) under per-class TS — going wider than 48 trades IoU for RMSE_bH without net gain. 48 is the Pareto point.
- The leaderboard metric's tighter RMSE ceilings (3m / 5m) weight height errors much more heavily than the old 30m placeholder, which is part of why the backbone/head ranking flipped from the 2026-04-15 report.
- Current best public score is still `0.4282` from the threshold-baked 4-way Q/Q-pretrain/N/Q-TerraMind-S2 ensemble; Y-3way has not been submitted yet.
- Use `model_best.pth` for prediction/evaluation. Inference CLI auto-reads model settings from `resolved_config.yml` for new runs, with legacy `training_params.json` fallback for archived runs.
- Y gated rich tied 3way is the current best single-model and best local-val run (`0.5107` raw / `0.5145` TS), clearing both the prior single-model and ensemble baselines.
- Q + AE/Tessera self-supervised pretrain remains a useful ensemble candidate, but is no longer the best threshold-swept local-val single model.
- The best current local-val submission candidate is Y-3way with per-class thresholds `(0.570, 0.580, 0.900)` until an updated ensemble around Y is tested.
- Water has an actionable postprocess: after thresholding, clear a patch's water mask when the largest connected water component is below a small `K`. Single-model val probes improved water IoU by `+0.0137` to `+0.0182` with `K=8` or `K=16`, directly targeting empty-water false positives under the per-image IoU metric. Re-sweep this jointly with the final ensemble water threshold before submission.
- R shows that late height capacity can improve the extreme high-value tail, but the current h96d4 independent-trunk design worsens the common height bins and water IoU. Do not rerun R as-is.
