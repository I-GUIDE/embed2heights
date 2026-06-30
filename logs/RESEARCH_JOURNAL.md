# Embed2Heights — Master Research Journal

> The architecture journey: every direction we created, tested, kept, and refuted on the
> ESA **Embed2Heights** challenge, from the first LightUNet baseline (2026-04) to the
> two-stage seg-purify + token-SSL pipeline (2026-06).
>
> **Compiled:** 2026-06-30. Sources stitched together: `logs/BEST_RESULT.md` (Apr–May
> champion board), `logs/techniques.md`, the full git history (92 commits, Apr 1 → Jun 30),
> the June root briefs (`OVERNIGHT_PLAN`, `MORNING_SUMMARY_*`, `RESEARCH_BRIEF_*`,
> `DIAGNOSIS_building_iou.md`, `ARCHITECTURE_DIAGNOSTIC_METHODOLOGY.md`,
> `SHAPE_DECODER_DESIGN.md`, `SESSION_SUMMARY.md`), the `runs/` + `runs/_archive/`
> experiment tree, and the two Claude session transcripts.
>
> This is a narrative + ledger. For the live "best strategy board" see `logs/BEST_RESULT.md`;
> for the challenge spec see `logs/GOAL.md`; for the cross-session memory see the personal
> memory notes (`seg-purify-stage3`, `height-offset-metric-trap`, `dinghye-refuted-archs`).

---

## 0. The problem and the metric (the constants everything is measured against)

**Task.** ESA Φ-lab "Reaching new heights with GeoFM." Per 256×256 patch (10 m/px, France),
predict a 4-channel raster `[building_frac, veg_frac, water_frac, height_m]` from **frozen
GFM embeddings only** — we never run the foundation models, we only design **fusion +
decoders**. Train = 2,024 labeled patches (French cities + rural, 2024); test ≈ 946 patches
from **different regions and years** — i.e. a deliberate **distribution shift**. Labels come
from 1 m IGN airborne LiDAR aggregated to 10 m cells, so they are **continuous fractions**,
not hard classes. GT bands: `0..2` = building/veg/water %, `3` = nDSM height (m), a single
shared height channel.

**The exact scoring formula** (reverse-engineered 2026-04-17 from an all-zero submission —
see `METRIC_PROBE_REPORT.md`; encoded in `core/metrics.py`):

```
Score = 0.25·IoU_build + 0.15·IoU_veg + 0.15·IoU_water        (segmentation, 55%)
      + 0.25·max(0, 1 − RMSE_buildH/3.0)                       (building height, 25%)
      + 0.20·max(0, 1 − RMSE_vegH /5.0)                        (veg height, 20%)
```

- **IoU** = positive-only, **per-image**: binarize GT at `label > 0` and pred at
  `pred > threshold`; both-empty → 1.0 (sklearn `zero_division=1.0`), exactly-one-empty → 0.0;
  mean over tiles. Implication: **a single false-positive blob in a class-empty tile flips that
  tile's IoU from 1.0 to 0.0** — punishing under the per-image average (this is why water FP
  suppression matters so much).
- **RMSE** = **per-tile** macro, GT-masked: `sqrt(mean((pred−gt)² over pixels where gt_class>0))`,
  then mean over tiles. **NOT global-pixel.** This is the single most expensive trap in the
  project (see §7, the height-offset metric trap): per-tile RMSE_bld ≈ 1.67 matches LB 1.778;
  global-pixel gave 3.42 and sent us chasing phantom offsets.
- **By weight:** building ≈ 50% of score, vegetation ≈ 35% (15 IoU + 20 height), water ≈ 15%.
  **Height is ≈ 45% of the total.**
- RMSE ceilings `X_build = 3.0 m`, `X_veg = 5.0 m` (small ceilings ⇒ height errors weighted
  heavily; this is why the backbone/head ranking *flipped* vs the pre-probe metric, where a
  30 m placeholder made IoU dominate).

**Submission discipline (mandatory, learned 2026-04-22):** ship **binarized** ch0-2 at
**swept per-class thresholds**, ch3 continuous. The threshold bake alone is worth ≈ **+0.047**
public; the identical ensemble shipped *continuous* scored −0.05. Continuous submissions are
confirmed worse on this leaderboard.

**The six frozen sources, and what each actually carries** (linear-probe R²,
build / height / veg / water — the recurring "which source wins which task" table):

| Source | Shape | Type | build | height | veg | water |
|---|---|---|---|---|---|---|
| **AlphaEarth** (AE) | 256²×64 | pixel-dense | 0.22 | **0.77** | 0.68 | **0.32** |
| **Tessera** | 256²×128 | pixel-dense | 0.29 | 0.70 | **0.70** | −0.23 |
| **TerraMind S1** | 16²×768 | token grid | **0.41** | 0.42 | 0.48 | −0.01 |
| TerraMind S2 | 16²×768 | token grid | broken (R² ≪ 0, unnormalized) |
| THOR S1 | 16²×768 | token grid | 0.31 | 0.39 | 0.49 | −0.02 |
| THOR S2 | 16²×768 | token grid | 0.30 | 0.45 | 0.39 | 0.05 |

**The single most important standing fact:** *different sources win different tasks.* AE →
height & water; Tessera → veg & building presence; TerraMind-S1 → building. Water is genuinely
hard (only AE is positive). The two pixel-dense sources (AE+Tess) carry the bulk; the four
16×16 token grids are spatially 16× downsampled and remain **barely exploited** — they boost
building IoU but repeatedly **overfit fold0 → collapse on the LB** (the "token-overfit signature,"
§4).

---

## 1. Timeline at a glance

| Period | Phase | Headline arc |
|---|---|---|
| Apr 1–18 | **Foundation** | Template, no-data masking, Tversky loss, LightUNet, head v2/v3/v35 |
| Apr 17 | **Metric probe** | All-zero submission reverse-engineers the true metric; local scores drop 0.25–0.35 |
| Apr 19–22 | **Champion ladder (single model)** | C→E→J→N: presence-centered loss, height specialists, base_ch widening |
| Apr 27–May 2 | **Gated fusion** | GMU/rich-gated AE+Tess; champion **Y-3way = 0.5107 raw / 0.5145 TS** |
| Apr 28 | **Multi-GFM fusion** | 4-modality hier-GMU + Dirichlet + cross-task attention (the breakthrough log) |
| May 10–17 | **Component sweep + lightweight fusion** | softbin + smooth → 0.4927; Lovász, copy-paste, SE, xfusion token cross-attn |
| May 19 | **Ensembling era** | 5-fold OOF 0.5245; `bul_cp` refuted (−0.048); thresholds tapped |
| May 20–29 | **Domain-gen + xfusion SoTA** | MixStyle, AdaBN; xf085→xf107 cross-source FiLM line |
| Jun 4–10 | **ViT rehabilitation → breakthrough** | Schedule fix falsified; capacity+region-balance+ensemble cross **0.51**; distillation 0.5165 |
| Jun 11–18 | **Two-stage + multi-backbone** | split-trunk head, presence/height grad-scale decouple, MultiBackboneFusion, DeiT distill |
| Jun 22–26 | **Diagnostic methodology** | Binary-fork probes; building-IoU diagnosis; label-redaction; shape-decoder scoped |
| Jun 23–28 | **U-Net++ submission** | Dinghye's unetpp + delmask + cov0.10 → **public 0.5018** (team best pushed) |
| Jun 27–30 | **SSL + seg-purify + combo** | 6-source token SSL (rejected); seg-purify Stage 3; height de-compression combo |

---

## 2. Phase: Foundation + the metric probe (April)

**Backbone.** The workhorse from day one is **`LightUNet`**: DoubleConv blocks
(3×3 → norm → act), 3 downsamples (bottleneck ≈ 32×32), U-Net skips, `base_ch` configurable.
On 2026-04-23 it was modernized: **GroupNorm → BatchNorm reconsidered**, ReLU → **GELU**,
optional **ResU-Net residual** blocks ("modern" mode). HRNet-W18/W32 were tried as alternatives
and **lost** (W18 0.3997, W32 under-tuned 0.3564 vs LightUNet 0.4213) — the recurring lesson
that **heavy/attention encoders are redundant on already-high-level GFM embeddings**.

**Heads.** A progression of multi-task prediction heads: `softplus` (v1) 0.3861 → `v2head`
0.3731 → `v3head` 0.4194 → **`v35head` 0.4213** (presence logits + base+Δ height structure +
softplus). v35 became the head family that everything built on.

**The metric probe (2026-04-17) — the pivot that reset all numbers.** Official metric defs
were never disclosed. An all-zero submission reverse-engineered them: it is **NOT**
`mean(IoU_pos, IoU_neg)@0.5` with global `/30` RMSE (our assumed default). It is positive-only
per-image IoU + per-tile GT-masked RMSE with tight 3 m/5 m ceilings. Consequence: every
pre-probe local score (the 0.74–0.77 era) was inflated by 0.25–0.35 and is **not comparable**.
All subsequent numbers are on the true metric.

---

## 3. Phase: The single-model champion ladder + gated fusion (late April → May 2)

Once the metric was correct, a disciplined single-model ladder ran on AE+Tessera, each rung a
one-flag edit (promotion bar: +0.005 val, no axis down > 0.02, reproducible). The lineage:

| Run | Score (raw / TS) | The change, and why it stuck |
|---|---|---|
| `C_presence_centered` | 0.4437 / 0.4491 | **Move Tversky onto presence logits, down-weight fraction MAE.** Direct presence BCE/Tversky beats fraction-centered structure losses. |
| `E_specialist_d2` | **0.4574** / 0.4615 | **`height_specialist_depth=2`** (per-class height trunks). Lifted all 5 axes; depth 4 (`L`) regressed at base_ch=32. |
| `J_specialist_d2_veg15` | 0.4568 / 0.4614 | `veg_height_boost=1.5` — Pareto point between G (3.0: +water, −RMSE_bH) and 0.0. |
| `N_base48` | **0.4644** / 0.4673 | **Widen LightUNet base_ch 32→48.** All 5 axes up at once; 5.44 M params. |
| `O_base64` | 0.4633 / 0.4672 | base_ch 64 **saturates** — best RMSE_bH on record (1.897 m) but IoUs regress. **48 is the Pareto point.** |
| `Q_fused_height_gate` | 0.4646 / 0.4712 | Height routing uses **fused** presence logits, not AE-only gate. RMSE_vH 3.40→3.33; IoU calibration shifts, recovered by threshold sweep. |
| `Q + AE/Tess SSL pretrain` | 0.4710 / **0.4745** | Transductive masked cross-reconstruction pretrain (train+test). Moderate lift; first hint SSL *can* help — later overturned at scale (§7). |

**The gated-fusion jump (2026-04-27 → 05-02).** Replacing the residual IoU branch with a
**rich gated cross-modality fusion** (GMU-style; `gate_mode=rich`, tied gates, `gate_init_bias=4.0`,
zero-init so output starts ≈ AE-only) produced the standing single-model champion:

> **`Y_gated_rich_tied_3way` = 0.5107 raw / 0.5145 per-class TS** (thresholds 0.570/0.580/0.900).
> iou_bld 0.5313, iou_tree 0.7664, iou_wat 0.4730, **RMSE_bH 1.668, RMSE_vH 2.974**.
> +0.0463 raw over N. Split **3-way presence head** (independent building/tree/water) + v35-style
> height specialists. This is still the best *clean single-model* fold-equivalent on the board.

**Loss anatomy that proved load-bearing here** (the "do-not-lower" list):
`build_height_boost=5` (K confirmed 3× weighting regressed RMSE_bH — closed); softplus on base
and per-class deltas (smooth negative-half gradient is load-bearing for RMSE_bH);
`presence_centered` preset. **Closed for good:** Huber height (D), focal IoU any α (F,H), MSE
height (I), symmetric b/v boost (K).

**Multi-GFM fusion breakthrough log (2026-04-28).** A parallel thrust built the
**4-modality `multi_gfm`** stack (AE+Tessera+TerraMind+THOR) with **hierarchical GMU**,
**Dirichlet auxiliary loss**, and **cross-task spatial attention** (`ctaskattn`), plus
**`pyramid_gated`** (multi-scale GMU at *every* U-Net level — CMGFNet's actual contribution).
These produced the `ens_canon_*` / `ctaskattn_*` 5-fold pipelines but did **not** beat the
2-modal gated champion on a clean basis; tokens added building-IoU but not net score.

---

## 4. Phase: Ensembling era + the OOF discipline (May)

Focus shifted from architecture to **variance-reduction ensembling** to climb the public LB.

- **Canonical 5-fold OOF baseline = 0.5245**, with a stable **OOF→LB gap ≈ 0.046** (OOF 0.5245 →
  LB 0.4788, confirmed by `submit_e100_5fold_bin.zip`). The historical **conv 6-seed ensemble
  ceiling = 0.5240** fold0. Multi-seed ensembling is the **biggest proven lever** (reliably
  +0.012–0.017; a single 0.50 seed → 0.524 ensemble).
- **`bul_cp` (bigUNet + Lovász + copy-paste) — REFUTED.** 5-fold mean OOF **0.4767 (−0.048)**;
  worse on heights too. Mechanism: its lower-confidence building probs can't reach the proven
  0.725 threshold → recall collapses (prec 0.73/rec 0.42 vs canon 0.66/0.69). **Blending it into
  canon HURTS** (−0.005 to −0.010). Do not submit.
- **Thresholds are tapped:** per-fold optima (cross-fold mean 0.5249) ≈ LB-proven 0.5245.
  Proven `(0.725, 0.575, 0.875) + water K=8` is near-optimal.
- **Marginal levers quantified (honest LOFO):** density-adaptive building threshold +0.0012;
  per-region (KE-only, 12/946 tiles) +0.0014; per-region thresholds judged **noise-fitting**.
- **Class-only TTA is dead** (−0.016 fold0). **D4 TTA on pixel models is fine** (+0.005–0.01);
  **TTA on token models is BROKEN** — flipping the 16×16 grid corrupts cross-attention (−0.027).
- **Calibration rule discovered:** the public LB tracks roughly the **worst-fold** CV score, not
  the 5-fold mean. Expect **LB ≈ worst-fold val**. fold0 is the hardest fold (KE-heavy, dense
  buildings) and the canonical dev fold.

**xfusion SoTA line (May 20–29).** A `CrossSourceHybridFiLMFusion` family (`xf085 → xf107`)
fused token sources via FiLM with a `use_additive` toggle; `MixStyle` (Zhou 2021) and AdaBN were
added for OOF→LB domain generalization. **MixStyle/AdaBN both ultimately neutral-to-negative**
and are on the do-not-resuggest list.

---

## 5. Phase: ViT rehabilitation → the 0.51 breakthrough (June 4–10)

**Mandate:** test whether the standing "ViT loses to conv" verdict was a *training* artifact.
Vehicle: **SegFormer-Lite ViT trained correctly**.

- **Schedule-fix hypothesis FALSIFIED.** Earlier SegFormer runs froze LR to ~1e-6 by e50 via
  hardcoded `ReduceLROnPlateau`. Added `--lr-schedule cosine --warmup-epochs --no-wd-on-norm`.
  Result: `vit_proper` = 0.4979 per-tile — **identical** to the old frozen-schedule run.
  Under-training was *not* the cause; **"ViT loses to conv" was correct** — ViT genuinely caps
  ≈ 0.498 on these embeddings. (The fix's +0.005 only surfaces *after* TTA.)
- **AugReg (Steiner 2022) — partially transferable.** From-scratch ViT on small data needs
  augmentation more than regularization. But **Mixup/CutMix HURT (−0.006)** — they blend feature
  maps, which is *not label-preserving for embedding inputs* (only H/V flips are). Dropout +
  stochastic-depth + copy-paste helped base48 +0.0014.
- **What actually worked: capacity + sampling + ensembling.** base64 > base48 (+0.0033);
  **region-balanced sampler** reproducibly best single config (`vit_regbal` 0.4948 no-TTA,
  highest iou_bld 0.5119).

> **The breakthrough (2026-06-10):** **3-seed `vit_regbal` ensemble = 0.5106** (+TTA, sweep) —
> the reliable cross-0.51 result, +0.012 over best single. **The ensemble, not multi-source
> fusion, was the breakthrough.**

- **Token fusion + distillation — strong but suspect.** `vit_msrc` (TerraMind-S1 cross-attn
  fusion) hit the best building IoU of all (0.5343 — TM-S1 *is* the best building feature) but
  **alone regressed height**. **DeiT-style distillation from the conv teacher RESCUED height**
  (best RMSEs: bH 1.671, vH 2.975 — the CNN-inductive-bias-transfer hypothesis held). Combined:
  **`vit_msrc_distill` = 0.5165 fold0 (no-TTA)**, best single by far.
- **⚠️ The token-overfit signature.** That +0.02 fold0 jump **mirrors the TM-S2 episode:
  fold0 0.5197 → LB 0.4588 (−0.06 collapse)**. TM-S1 tokens + a fold0 teacher are both fold0-overfit
  risks. **Standing discipline: fold0 ≠ LB; gate every token/distill result on 5-fold OOF; never
  submit from fold0.** The safe submission candidate from this phase is the **no-token 3-seed
  ensemble (0.5106)**.
- **Strategic redirect (from the public board):** #1 DisasterM3 0.5448 (outlier veg-IoU 0.892),
  #2 Alchimist 0.5313 (outlier build-IoU 0.567), **#3 us 0.5137**. Our gap to #1 = water-IoU
  −0.014, veg-height −0.011, veg-IoU −0.011, partly offset by our **field-best building-height
  (1.76)**. Read at the time as "building is maxed, points are in water+veg" — later *re-litigated*
  by the diagnostic methodology (§6).

---

## 6. Phase: The diagnostic methodology + building-IoU diagnosis (June 22–26)

The central intellectual pivot: replace *"try a change, see if the score moves"* with
**decisive binary forks, input-exonerating probes, and causal ablation**
(`ARCHITECTURE_DIAGNOSTIC_METHODOLOGY.md`). A probe must measure what is *extractable from the
input* (not just what the model got), so it can **exonerate the input**; mechanism is proven
**causally** (ablate X, watch the metric move).

**The diagnostic ladder (with embed2heights answers):**
- **Q1 decision-rule vs ranking?** Per-tile oracle IoU − global-threshold IoU = "op-gap".
  Building **op-gap = 0.0185** ⇒ ranking/representation-limited, *not* fixable by a smarter
  threshold. RankSEG predicted to fail, did (**−0.0126**).
- **Q2 input vs model?** No-pool per-pixel logistic on 1–4px buildings **0.380 vs UNet 0.275** —
  direct proof that **8× downsampling pools small buildings away** — but the logistic itself only
  0.380, so **~62% of tiny buildings are input-invisible** (resolution-capped).
- **Q4 boundaries vs interiors?** veg/water lose massively at boundaries (interior 0.98 vs
  boundary 0.75); the per-pixel probe collapses *more* than the model ⇒ an **input-resolution
  wall**, and the model already exceeds the per-pixel floor via context.
- **Q5 discrimination axis tapped?** veg ≈ 0.82 across linear / MLP / ±250 m-context / full UNet
  ⇒ **per-pixel discrimination is tapped**; the only remaining lever is **spatial structure /
  object shape**, which probes structurally cannot measure.
- **Q6 LB gap by component weight?** **~75% of the gap to #1 is segmentation (building+water IoU),
  ~25% height** — *refuting* the earlier "height is the lever" steer.

**Mechanistic proof example.** `mechanism_proof.py` zeros the Tessera input → building IoU
**0.530 → 0.303** ⇒ the building signal lives in Tessera.

**The building-IoU diagnosis (`DIAGNOSIS_building_iou.md`, 2026-06-24, ~18 h autonomous, zero
training launched).** Our building IoU 0.475 vs Alchimist 0.628 = the #1 LB gap.
- **It's small objects, not boundaries.** Interior−boundary gap only +0.029 (vs veg +0.233).
  Detection recall by size: **1–10px = 0.584**, 10–50px 0.92, ≥50px ≥0.94. **84% of missed
  building pixels are predicted as background** (vanish, not misclassify). 23.7% of buildings
  wholly missed.
- **Recoverable but bounded.** Pooling erasure is real and proven, but ~62% of tiny buildings are
  input-invisible; recoverable ≈ 24% of misses, capped ~0.38 recall on 1–4px, mostly on sparse
  rural tiles. **NOT a path to a big jump.**
- **The reconciliation that killed the obvious fix.** The natural implementation —
  `--detail-bypass` (full-res zero-init detail branch) and `--sharp-upsample` — **already exists
  and already failed** (0.4883/0.4891 vs 0.4960): they nudged veg up but **HURT building IoU
  −0.005 to −0.017**. Lesson: **building IoU is fragile; anything reweighting toward
  veg/water/boundary costs it.** Hedged cause: the shared detail branch fed the building head a
  raw full-res path that *increased hallucination* (FP isolated-blob fraction 0.37).
- **Honest bottom line:** our model is at *its own* oracle ceiling (0.531) yet Alchimist hits
  0.628 — so **0.628 is reachable on the task, just not from AE+Tess at our resolution.** The
  leader's *uniform* seg lift across all three classes is the signature of **higher effective
  input resolution / more bands — a data lever we don't have**, not a decoder fix.

**Follow-ups falsified with proof (2026-06-25):**
- **Resolution-preserving encoders FALSIFIED** — `--encoder-arch {shallow,dilated,hrnet}`: 1–4px
  recall flat (0.276/0.282 vs 0.275), iou_bld −0.01. The no-pool 0.38 ceiling was a *probe
  artifact a trained conv cannot reach.*
- **Height-capacity screen FALSIFIED** (`ss_hcap`) — bigger 256×4 height trunk: RMSE_bH
  **1.753 → 1.791 (worse)**, iou_bld −0.015. **Height is near-ceiling.**
- **Per-class label tweaks FAILED** — `--water-argmax-bg` over-cleaned (water IoU 0.466→0.454);
  `--building-presence-thr 0.1` flat.

**The label-redaction thread (corrected twice — a good example of evidence overturning inference):**
~29.8% of training pixels are all-zero GT over 99.8%-intact embeddings. First inferred "censored
buildings"; direct inspection (`inspect_gt_holes*.py`) overturned it. There are **two blank
types**: (a) **big rectangular = sparse rural annotation** (building frac 0.021 < global 0.053,
mostly fields — the 56-tile set), and (b) **small scattered blanks inside cities = real urban
content** (13.6%/7.8% of those tiles). The urban blanks cost a measured **+0.0075 building IoU**
locally — **but fictional w.r.t. the LB** because the hidden test shares the same blanking, so
our blank-region predictions are penalized there too. `--cls-hole-mode exclude` = neutral;
`impute` (height-derived fake labels) **backfired (−0.0095**, precision cratered to 0.55).
Not a recoverable lever.

**Shape-decoder design (`SHAPE_DECODER_DESIGN.md`) — scoped, gated, then tested.** Reasoning:
per-pixel discrimination is tapped (Q5), capacity saturated ~0.515, and the one untested
mechanism is **object shape**. Design: reuse the proven super_stack encoder → FPN pixel decoder
→ per-pixel embedding `E ∈ ℝ^{C×H×W}`; transformer decoder with **N=32–64 learned object
queries**; per-query mask `m_q = sigmoid(E · q)`; final per-class map = Σ_q class-prob · m_q;
keep the softbin height head unchanged; Hungarian-matched dice+BCE adapted for fractional soft
targets. **Result (`shape{16,32,64}_fold0`): shape16 = 0.5103, shape32/64 = 0.5051/0.5050 —
the shape decoder did not beat the gated baseline, and more queries did not help.** Confirms the
honest expectation in the design doc: if shape-modeling also stalls at ~0.82 veg, the embeddings
genuinely don't support better shape recovery and ~0.52–0.53 is the fold0 single-model ceiling.

---

## 7. Phase: SSL pretraining, two-stage seg-purify, and the height-generalization combo (June 27–30)

**Self-supervised pretraining of the backbone — TESTED AND REJECTED.** Built `core/pretrain/`:
masked-reconstruction and denoise pretext objectives, including a **6-source token SSL +
denoise pretext** (commit `719aa00`). Trained `ssl6_pretrain_unetppwave_v1`, then fine-tuned.

> **SSL init consistently HURT vs from-scratch:**
> `full_ssl` (unetpp_wave + SSL init) = **0.4898** vs `full_nossl` (scratch) = **0.5116** (−0.022);
> `ts_ssl` = 0.4843 vs `ts_scratch` = 0.4878. The early transductive AE/Tess pretrain lift (§3,
> Q-pretrain +0.0035 TS) **did not survive** in the bigger unetpp_wave setting. SSL is a
> tested-and-rejected bet.

**The two-stage train→purify pipeline (the canonical "P3" end state).** Built around a
**dual-trunk / split-trunk head** (`split_trunk`, commit `0fc061a`) that gives segmentation and
height **separate trunks**, plus symmetric gradient-decoupling knobs so one task can re-tune the
shared backbone without the other's gradients interfering:

```
# core/models/heads.py — the symmetric grad-scale cut
x_seg = s_pg·x + (1 − s_pg)·x.detach()    # presence_grad_scale
x_hgt = s_hg·x + (1 − s_hg)·x.detach()    # height_grad_scale  (s=0 ⇒ path sees x.detach())
# FiLM modulation inside the head:  h = x·(1 + scale) + shift
```

- **Stage 1 (coupled)** — train everything normally (~80 ep).
- **Stage 2 (height-purify)** — `--presence-grad-scale 0 --select-on height`: only the height
  path re-tunes the backbone; supplies ch3.
- **Stage 3 (seg-purify, the +5th-place lever)** — the mirror: init from S2, `--height-grad-scale 0`,
  retrain segmentation ~20 ep; supplies ch0-2. Rationale: building IoU was still climbing inside
  the height-purify stage, so an explicit seg re-tune banks that headroom.
- **Merge** — seg ch0-2 from S3, height ch3 from S2 (`tools/merge_twostage_preds.py`), with
  per-class height offsets.

**Dinghye's verified recipe (teammate, the current pushed team best).** Documented in
`docs/UNETPP_SUBMISSION.md` (branch `exp/unetpp-3seed-submission`):

> **Public 0.5018** — model `xfusion_unet_hybrid_cross_source`: AE(64) + Tess(128) each through
> their own **`LightUNetPP`** (nested U-Net++ decoder, base_ch=48, ~12.7 M), rich gated fusion
> (`gate_init_bias=4.0`), 4 token sources (TerraMind+THOR S1/S2) via `CrossSourceHybridFiLMFusion`
> with `token_calibration=true`; split-trunk head, **softbin height** (64 log-bins, max 80 m),
> independent height branches. **3 verified levers:** (1) **U-Net++ pixel backbone** (+0.005 bld
> IoU); (2) **delmask** (drop seg loss on human-deleted building footprints); (3) **cov>0.10 GT
> alignment** (`presence_coverage_threshold=0.1`) — "the single biggest metric-definition fix."

> **Her local→public gap (0.5345 → 0.5018) is ENTIRELY height:** RMSE_H_BUILD 1.49→1.81 and
> RMSE_H_VEG 2.76→3.08, **both ≈ +0.32 m** on the held-out test biome; all three IoUs transfer
> perfectly. Her two *unpushed* local levers reaching **0.5061** are **seg-purify (Stage 3)** +
> a **+0.25 m building-height offset**.

> **Her do-NOT-redo list** (refuted in *her* pipeline; strong prior, not proof, for ours):
> wavelet/Haar-DWT downsampling, deep supervision, TransUNet bottleneck self-attention,
> ViT/SegFormer/HRNet/UPerNet/CBAM/Attention-UNet. **Audit finding:** our `backbones.py` ships
> all three refuted archs (`HaarDownsample`/`_wave`, `BottleneckSelfAttention`, `_HRNetLite`), and
> the running two-stage sbatch was **unconditionally forcing `--use-bottleneck-attn`** on every
> job + one run used `ENC_ARCH=hrnet`. We had already burned **8 `unetpp_wave` + 2 `hrnet`** runs
> on refuted archs before this was caught; bottleneck-attn made opt-in (default off).

**The height-offset metric trap (the most consequential gotcha, encoded in memory).** Fitting
height offsets on the **wrong RMSE** sends you the wrong way:
- Under *global-pixel* RMSE, the building optimum looked like **+0.333 m**.
- Under the **real per-tile metric with predicted-class masks** (what merge actually does):
  building optimum **+0.10 m** (−0.0024); **+0.25 m HURTS** (+0.0042); +0.50 m clearly hurts.
  Veg optimum −0.10 m. Multiplicative/affine calibration refuted; flat offset only.
- **Resolution of the paradox:** local OOF *cannot see* the +0.32 m public biome shift. Dinghye's
  board-validated **+0.25 m calibrates to that shift, not to local bias** — which is exactly why
  it reads net-negative locally yet works on the LB. **Takeaway: the height offset is a
  leaderboard knob (~ +0.25 to +0.42 m), never fit it on local val.** The merge tool now defaults
  `--public-height-shift 0.32` and recommends `--build-height-offset ≈ +0.42` (local +0.10 +
  public +0.32).

**Error anatomy that set the combo's direction (`error_anatomy.py`, fold0 405 tiles):**
- **building IoU 0.467 is FP-heavy** (px-precision 0.524 < recall 0.761) → *recall boosters push
  the wrong way*; the old `bldtune` building-Tversky lever was dropped from the combo.
- **water IoU 0.463:** 27 of 42 GT-empty tiles get a false water blob (each 1.0→0.0) — the per-image
  IoU killer.
- **Height range-compression is the real height problem:** `pred_build ≈ 0.536·gt + 2.09`
  (slope ≪ 1 ⇒ tall structures under-predicted; tall-GT bias −1.84 m, RMSE 5.33 dominates);
  `pred_veg ≈ 0.810·gt + 1.92`. **De-compression as a post-process is REFUTED** (any gain k>1
  worsens RMSE because residual std 3.4 swamps the bias) — *compression must be fixed in training.*

**Levers tested for the combo:**
- 🟢 **Water FP suppression CONFIRMED** — threshold 0.70→0.80 + connected-component K=8:
  water IoU 0.463→0.502 (**+0.0059 score**). Free post-process.
- 🔴 Building morphology (open/close/fill/remove-CC) all **REFUTED** (−0.07 to −0.17) — per-tile
  IoU punishes erosion; building shape must come from training.
- The combo's training-side fix for compression: **LDS sampler** (`--lds-sampler --lds-score
  building_frac_p95`, label-distribution-smoothing oversampler that up-weights tall/dense KE-like
  patches up to **4.4×**) + **sharper softbin target** (`--height-bin-sigma-bins 0.75` from 1.5,
  less mean-regression) + **seg-purify Stage 3**, on a clean **unetpp + delmask + cov0.10, no
  bottleneck-attn** base.

**Current best validated numbers (fold0 merged, headline per-class + water K=12 Score):**

| Run | iou_bld | iou_wat | RMSE_bH | RMSE_vH | **Score** |
|---|---|---|---|---|---|
| **full_nossl** (unetpp_wave, from scratch) | 0.5266 | 0.5035 | 1.688 | 2.952 | **0.5116** ← best recent single |
| full_hdrop / full_aligned | 0.474 | 0.503/0.506 | 1.67 | 3.00 | 0.4970 / 0.4969 |
| full_combo (tokens + gated) | 0.4715 | 0.5070 | 1.657 | 3.039 | 0.4968 (lev 0.4973) |
| full_ssl (SSL init) | 0.4955 | 0.4866 | 1.755 | 3.083 | 0.4898 |

**The GO/NO-GO guardrail (user-set, still in force).** Submit only if confident of a *significant*
beat over the teammate's 0.5018/0.5061: (1) building-height slope moves 0.54 → ≥ 0.75 and the
−1.84 m tall-bias at least halves; (2) no IoU regression vs 0.4949 merged baseline; (3) ≥3-seed
ensemble footing; (4) the combo clearly beats our own clean baseline. **The only plausible path
to beat her is height generalization to the test biome** — her documented weakness, the +0.32 m
shift being 100% of her local→public gap — and **local OOF cannot measure that**; it must be
reasoned by mechanism and confirmed only on the LB.

---

## 8. The model zoo (architecture catalog, for reference)

Implemented across `core/models/{backbones,heads,pixel_fusion,token_fusion,registry,factory}.py`:

- **Backbones:** `LightUNet` (base_ch 32/48/64, GN/BN, GELU, ResU-Net residual, optional ASPP /
  coord-attn / SE / bottleneck self-attention), **`LightUNetPP`** (nested U-Net++ decoder — the
  teammate win), `_HRNetLite` (W18/W32 — lost), SegFormer-Lite ViT (AugReg, capped ~0.498),
  `HaarDownsample`/`_wave` (wavelet downsampling — refuted), `shallow`/`dilated` resolution-
  preserving encoders (falsified).
- **Fusion:** `ae_tessera_gated` (GMU, the workhorse), rich/tied/untied/dropout gate variants,
  `pyramid_gated` (GMU at every U-Net level), `multi_gfm` (4-modality hier-GMU + Dirichlet +
  cross-task spatial attention), `AeTesseraMlpFusion`, `AeTesseraMoeFusion` (pixel-level Top-K=2
  MoE), `CrossSourceHybridFiLMFusion` (xfusion FiLM, `use_additive` toggle), `token_fusion` /
  `xfusion_*` (TerraMind/THOR token cross-attention at bottleneck), `MultiBackboneFusion`
  (per-source backbones, pretrained, fused — needed an AMP-fp16 BatchNorm-corruption fix at eval),
  `xfusion_unet_hybrid_cross_source` (the teammate submission model).
- **Heads:** `v2/v3/v35head`, `MultiTaskPredictionHead` (presence logits + base+Δ height +
  softplus), **split-trunk** (separate seg/height trunks) with `presence_grad_scale` /
  `height_grad_scale` decouple, **softbin** height (64 log-bins 0–80 m → expectation, bin-CE aux,
  presence-gated class blend, berhu loss), height specialists (per-class trunks, depth 1–2),
  split 3-way presence, shape-query decoder (MaskFormer-style, N=16/32/64 — did not beat baseline).
- **Losses (`core/losses/`):** `ImprovedCompositeLoss` = MAE (fg/bg split) + Tversky (α=0.3,
  β=0.7) + per-class height boost (build 5×, veg 1.5×) + aux multi-head; `presence_centered`
  preset; Lovász-hinge (building); BerHu / softbin bin-CE (height); building-smoothness TV
  (`--building-smooth-weight 0.5`); boundary-ring-weighted BCE; **delmask** (drop seg loss on
  deleted footprints); homoscedastic-uncertainty multi-task weighting; pinball/quantile height.
- **Decoder upsamplers:** bilinear (baseline), PixelShuffle, **CARAFE**, **DySample** (arch-A line).
- **Augmentation:** D4 / flip-rot180 (label-preserving), building copy-paste, region-balanced
  sampler, LDS oversampler; **Mixup/CutMix refuted** (not label-preserving on embeddings).
- **Pretraining (`core/pretrain/`):** masked-reconstruction + denoise, 6-source token SSL —
  **rejected** (hurt vs scratch).

---

## 9. Consolidated ledger — what works vs what's refuted

**✅ Load-bearing / kept:**
- LightUNet / **LightUNet++** at **base_ch=48** (Pareto point; 64 saturates).
- Rich **gated AE+Tessera fusion** (`gate_mode=rich`, bias 4.0, zero-init).
- **Split 3-way presence head** + **height specialists** (depth 2) + **softbin+BerHu** height.
- **presence_centered** loss; `build_height_boost=5`; softplus on height projections.
- **delmask** + **cov>0.10 GT alignment** (biggest metric-definition fix, +0.005 bld IoU).
- **Two-stage purify** (height-purify S2, **seg-purify S3** = the +5th-place lever).
- **Multi-seed × multi-fold ensembling** — the biggest proven lever (+0.012–0.017).
- **Threshold bake + binarize** (mandatory, +0.047 public); **water thr 0.80 + CC K=8** (+0.0059).
- **Region-balanced sampler**; **D4 TTA on pixel models**; building copy-paste; LDS oversampler.
- Building height offset as a **leaderboard knob** (~ +0.25 to +0.42 m), never fit locally.

**❌ Refuted / closed (with proof):**
- SSL pretraining (−0.022 vs scratch); HRNet (−0.02); from-scratch ViT (caps ~0.498).
- Wavelet/Haar downsampling, deep supervision, TransUNet bottleneck self-attention (teammate-refuted;
  we still carried bottleneck-attn until June 29).
- Mixup/CutMix (not label-preserving); MixStyle/AdaBN (neutral-negative); class-only TTA (−0.016);
  token-model TTA (−0.027, grid-flip breaks cross-attention).
- `bul_cp` ensemble (−0.048 OOF) and all canon⊕bul_cp blends.
- Detail-bypass / sharp-upsample (hurt building IoU); resolution-preserving encoders (falsified);
  height-capacity (RMSE worse); shape-query decoder (did not beat gated baseline).
- Post-hoc height de-compression, building morphology, per-region/per-tile threshold tuning
  (noise-fitting); cls-hole `impute` (−0.0095); RankSEG (−0.0126).
- Pre-2026-04-17 metric (mean(IoU_pos,IoU_neg)@0.5, global /30 RMSE) — all those scores void.

**⚠️ Open / unresolved:**
- **The #1 gap (building IoU 0.475 → 0.628) is most likely an input-resolution / extra-band data
  advantage** the leaders have and we don't — not a decoder fix. Our model sits at its own oracle
  ceiling (0.531).
- **Beating the teammate requires height generalization to the shifted test biome**, which local
  OOF cannot measure — a mechanism-reasoned, LB-confirmed bet.
- Token sources remain barely exploited and overfit-prone (TM-S1 helps building but collapses on
  LB without 5-fold gating).

---

## 10. Standing numbers to anchor against

| Quantity | Value |
|---|---|
| Best **clean single-model** (Apr–May champion board) | `Y_gated_rich_tied_3way` **0.5107 raw / 0.5145 TS** (fold-equiv) |
| Best recent **single-model fold0** | `full_nossl` (unetpp_wave scratch) **0.5116** |
| Canonical **5-fold OOF** ceiling | **0.5245** (15-model conv ensemble); OOF→LB gap ≈ 0.046 |
| Historical **conv 6-seed ensemble** | 0.5240 fold0 |
| **ViT** ceiling on these embeddings | ≈ 0.498 single; 0.5106 (3-seed regbal ens, +TTA) |
| Best **public LB (pushed)** | teammate **0.5018** (unetpp + delmask + cov0.10) |
| Best **local (teammate, unpushed)** | 0.5061 (+ seg-purify + +0.25 m offset) |
| Our standing **public rank** | #3 ≈ 0.5137 (gap to #1 ≈ water-IoU −0.014, veg-height −0.011, veg-IoU −0.011) |
| Noise floor on val score | ≈ 0.006 |
| Promotion bar | +0.005 val, no axis down > 0.02, reproducible (split+seed+SHA) |

---

*End of master journal. Keep `logs/BEST_RESULT.md` as the live champion board; append new phases
here as the work continues.*
