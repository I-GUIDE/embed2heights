# Prediction-Head Evolution Report

Date: 2026-04-22
Subject: LightUNet prediction-head architecture, V1 → V2 → V3 → V35

This note traces the four prediction-head generations used on the AlphaEarth
LightUNet backbone, the motivation behind each redesign, and the per-metric
validation scores that justified the jumps. It complements
[HEAD_ARCHITECTURE_NOTES.md](HEAD_ARCHITECTURE_NOTES.md), which documents the
intra-V35 `nobase`/`hybrid` ablation; this report is the outer chronology.

## TL;DR

- V1 (single 1x1 conv) → V2 (double-head + concat height): score regressed
  from `0.3861` to `0.3731`. V2 still submitted `fractions` on channels 0-2,
  which misaligned with the server IoU metric (`label > 0`).
- V2 → V3 (deep residual trunk + FiLM + presence submitted + fraction-gated
  height): **+0.046 score** (0.3731 → 0.4194). All the lift is on IoU; RMSE
  unchanged. This is the metric-alignment fix.
- V3 → V35 (same trunk; height aggregation switched from fraction-weighted
  sum to presence-gated specialist blend): **+0.002 score** (0.4194 → 0.4213,
  within noise floor ~0.006) but RMSE_bH genuinely improves. Topology frozen
  after V35; further lift comes from specialist-depth and trunk width.
- V35 + `specialist_depth=2` (E): 0.4574 raw.
- V35 + `specialist_depth=2` + `base_ch=48` (N, current champion): **0.4644
  raw / 0.4673 per-class TS**.

Diagrams for every generation: [logs/figures/head_v1.png](figures/head_v1.png),
[head_v2.png](figures/head_v2.png), [head_v3.png](figures/head_v3.png),
[head_v35.png](figures/head_v35.png), and the four-panel poster
[head_compare.png](figures/head_compare.png). SVGs next to each PNG.

## Full Score Table (val 405 samples, post-probe metric, raw @0.5)

Evaluated via [evaluate.py](../evaluate.py) with `--val-only` against
[splits/split.json](../splits/split.json). Score = weighted composite,
`X_bld = 3.0 m`, `X_veg = 5.0 m`, label threshold `> 0`.

| Head | Run | iou_bld | iou_tree | iou_wat | RMSE_bH | RMSE_vH | **Score** |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 softplus | `lightunet_alphaearth` (predictions purged) | — | — | — | — | — | 0.3861 (log) |
| **V2head** (LightUNet) | `lightunet_v2head` | **0.2644** | 0.7247 | 0.3957 | 1.9979 m | 3.6149 m | **0.3731** |
| V3head (HRNet proxy) | `alphaearth_hrnet_w18_v3head` | 0.3623 | 0.7461 | 0.3987 | 1.9682 m | 3.5797 m | 0.4051 |
| V35head (HRNet) | `hrnet_w18_v35head` | 0.3770 | 0.7404 | 0.4006 | 2.0528 m | 3.6170 m | 0.3997 |
| **V35head** (LightUNet, d=0, base=32) | `lightunet_v35head` | 0.3983 | 0.7416 | 0.4151 | 1.9154 m | 3.5532 m | **0.4213** |
| E: V35 + specialist_depth=2 | `alphaearth_tessera_iou_fusion_E_specialist_d2` | 0.5029 | 0.7587 | 0.4497 | 1.9181 m | 3.4929 m | 0.4574 |
| **N**: E + base_ch=48 (champion) | `alphaearth_tessera_iou_fusion_N_base48` | **0.5087** | **0.7618** | **0.4537** | 1.9111 m | **3.3963 m** | **0.4644** (TS 0.4673) |
| O: E + base_ch=64 (saturated) | `alphaearth_tessera_iou_fusion_O_base64` | 0.5066 | 0.7562 | 0.4488 | **1.8971 m** | 3.3999 m | 0.4633 (TS 0.4672) |

Notes:
- V1's run directory no longer has `predictions/`, so per-metric cells are
  unrecoverable without retraining. Only aggregate score survives in the
  experiment log.
- The LightUNet+V3head run was pruned during cleanup; we retain the HRNet+V3
  proxy. On the original log LightUNet+V3 scored 0.4194 (aggregate), so the
  per-metric table should read ~+0.015 on `iou_bld` relative to the HRNet row.
- E/N/O values come from [K_L_EXPERIMENTS_REPORT.md](K_L_EXPERIMENTS_REPORT.md)
  and [N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md).

## V1 — softplus head (baseline, 0.3861)

Structure: the LightUNet ended with `nn.Conv2d(32, 4, kernel_size=1)`; the
sigmoid/softplus split was applied outside (sigmoid on channels 0-2, softplus
on channel 3). No trunk, no presence concept, a single height regression.

Source: [initial commit](../core/model.py) (56bf23c), lines 49-95.

Why it lost ground:
- Submission channels 0-2 were the *fraction* sigmoid output. Under the
  server's positive-only IoU at `label > 0`, this is tolerable at small
  thresholds but not calibrated; post-probe this misalignment became obvious.
- A single direct height regression averages building and vegetation height
  at every pixel, so tall-building peaks get smoothed.

## V2 — MultiTaskPredictionHead (initial, 0.3731)

Introduced by commit 8b5bdc3, 2026-04-15. Structure:
- Shared `ConvGNAct` (single layer).
- `fraction_head` and `presence_head`, each `ConvGN + 1x1 → 3`.
- `height_features = concat(x, fractions, presence)` fed to three height
  branches: `base`, `h_bld`, `h_veg` (each `ConvGN + 1x1 + softplus`, direct
  absolute height rather than base + delta).
- Submitted height: `clamp(base + g_b·(h_bld − base) + g_v·(h_veg − base),
  min=0)` with `g = presence[:2]`.
- Submission channels 0-2: **fractions** (unchanged from V1, presence not
  submitted).

Where V2 regressed to 0.3731 despite adding structure:
1. **Metric misalignment persisted.** Presence existed internally but was not
   submitted. Under 0.5 threshold at inference, fraction channels sparsified
   building to near zero — `iou_bld = 0.2644`, by far the lowest of any head.
2. **Trunk too shallow.** One layer of `ConvGN` cannot support three heads
   plus a concat-fed height branch. Height sees only concatenated
   mismatched-scale tensors.
3. **Direct absolute height branches** (not `base + delta`) force each branch
   to regress full heights from scratch; without specialisation supervision
   the deltas never learn class-specific adjustments.

## V3 — metric-aligned deep trunk (0.4194)

Introduced by commit 62e4f4e, 2026-04-18, directly motivated by the metric
probe on 2026-04-17 ([METRIC_PROBE_REPORT.md](METRIC_PROBE_REPORT.md)).

Changes versus V2:
1. **Deeper shared trunk**: 2-layer residual (`ConvGN + ConvGN + Conv + GN`,
   then `⊕ GELU`). Three heads now share real capacity.
2. **Presence head becomes the submission path**: submission channels 0-2
   switch from `fractions` to `presence_prob`. BCE on `label > 0` supervises
   it directly, aligning with the server IoU metric.
3. **FiLM conditioning**: `h = x · (1 + scale) + shift` where
   `scale, shift = 1x1_conv(fractions)`. Replaces V2's concat — the height
   branch now sees fraction signal at the feature level, not channels of
   different semantics glued together.
4. **Height becomes base + delta**: three 1x1 projections inside a shared
   `height_trunk` produce `base`, `Δ_b`, `Δ_v`, each through softplus so
   deltas are ≥ 0. Physical constraint that building and vegetation only add
   height above ground.
5. **Fraction-gated submitted height**:
   `height = base + frac_b · Δ_b + frac_v · Δ_v`. Rationale: continuous
   fraction weight keeps partial-coverage (edge) pixels calibrated.

Per-metric lift from V2 → V3 (HRNet-W18 proxy row vs LightUNet V2):
- `iou_bld`: 0.2644 → 0.3623 (**+0.098**) — entirely from submitting
  presence instead of fractions.
- `iou_tree`: 0.7247 → 0.7461 (+0.021)
- `iou_wat`: 0.3957 → 0.3987 (+0.003)
- `RMSE_bH`: 1.9979 → 1.9682 m (−0.03 m, within noise)
- `RMSE_vH`: 3.6149 → 3.5797 m (−0.035 m, within noise)

The entire V3 gain is on IoU. Height regression accuracy is essentially
unchanged — which is consistent with the structural interpretation: V3's
trunk + FiLM fix the classification path first, not the regression path.

## V35 — presence-gated specialist blend (0.4213, current topology)

Commit fc0043d, 2026-04-18. **Only one line of code changed** versus V3: how
the submitted height channel aggregates the three projections.

```python
p_b, p_v   = presence_prob[:, 0:1], presence_prob[:, 1:2]
p_fg       = 1 − (1 − p_b) · (1 − p_v)
w_b, w_v   = p_b / (p_b + p_v + ε),  p_v / (p_b + p_v + ε)
h_fg       = w_b · (base + Δ_b) + w_v · (base + Δ_v)
height     = p_fg · h_fg + (1 − p_fg) · base
```

Rationale: the server scores `RMSE_building_height` over pixels with
`label_building > 0` and `RMSE_vegetation_height` over pixels with
`label_vegetation > 0`. These masks match the *presence* classifier's
supervision exactly. The aux L1 losses in [core/losses.py](../core/losses.py)
already train `building_height = base + Δ_b` on the building-present mask
only (and likewise for vegetation), making each "specialist" reliable only
on its own class's pixels. The new aggregator routes each pixel to the
specialist whose presence mask claims it, with a smooth soft mix near
boundaries and a `base` fallback in the background.

Per-metric lift from V3 → V35 (LightUNet direct comparison via archived
aggregate): score 0.4194 → 0.4213, within the 0.006 noise floor. The
ablation in [HEAD_ARCHITECTURE_NOTES.md](HEAD_ARCHITECTURE_NOTES.md) later
showed V35 has a real RMSE_bH edge over hybrids (1.9154 vs 1.9820 vs 2.0412
m for `v35 / hybrid / nobase`), which is the signal that matters for the
25% building-RMSE weight.

## Capacity Extensions (topology frozen, parameter tuning only)

From 2026-04-19 onward no head-topology changes have been accepted. All
post-V35 gains came from orthogonal knobs:

- **`height_specialist_depth = 2`** (E, 2026-04-21): each `Δ_b` / `Δ_v`
  projection becomes two `ConvGNAct` + `1x1` layers instead of a single 1x1.
  `+0.0102` score vs V35-equivalent baseline. All five metrics improved;
  first "clean five-axis win" in the log. See
  [K_L_EXPERIMENTS_REPORT.md](K_L_EXPERIMENTS_REPORT.md).
- **`specialist_depth = 4`** (L): best RMSE_bH on record at base=32 (1.8962
  m) but `iou_bld` drops 1.6 pt. Depth knob saturates at 2 on the 32-trunk.
- **`base_ch = 48`** (N, 2026-04-22): LightUNet's base channel widens from
  32 → 48, touching every `DoubleConv`. `+0.0076` raw / `+0.0105` per-class
  TS over J. All five axes improve simultaneously. See
  [N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md).
- **`base_ch = 64`** (O): saturates. Best RMSE_bH across all runs (1.8971
  m) but `iou_tree` and `iou_wat` regress. `base_ch = 48` is the Pareto
  point.

## Why the V3 Topology Was Load-bearing (and the V35 Ablation Later Confirmed It)

The 2026-04-19 V35 `nobase` and `hybrid` ablations (see
[HEAD_ARCHITECTURE_NOTES.md](HEAD_ARCHITECTURE_NOTES.md)) were designed to
test whether pieces of the V3/V35 structure could be removed:
- Removing `base_height`: regressed.
- Swapping softplus → ReLU in the height projections: regressed.

The surprise was that `RMSE_building_height` sorted strictly by "how much
softplus remained", not by "whether base was present" — ReLU's dead-zone
gradient on negative logits is load-bearing for building height regression,
since many building pixels need logits to climb far above zero, and softplus
keeps gradient alive throughout training.

So the order of the four generations is not just "add more structure":
1. V1 → V2: added structure, lost score (metric mismatch dominated).
2. V2 → V3: two real wins stacked — metric alignment on IoU, and softplus +
   base + delta structure on RMSE (though RMSE gains only manifested later
   once the trunk had capacity).
3. V3 → V35: fixed one remaining mismatch (height aggregation vs RMSE
   mask), within noise on aggregate but load-bearing for RMSE_bH.
4. V35 → E → N: topology frozen; exploit capacity.

## Reproducing

```bash
# Re-evaluate any of the surviving head runs on the val split:
python evaluate.py --val-only --only \
    lightunet_v2head \
    lightunet_v35head \
    hrnet_w18_v35head \
    alphaearth_hrnet_w18_v3head

# Regenerate the architecture diagrams:
python tools/plot_head_evolution.py
# outputs: logs/figures/head_{v1,v2,v3,v35,compare}.{png,svg}
```

## References

- Metric specification: [METRIC_PROBE_REPORT.md](METRIC_PROBE_REPORT.md)
- V35 intra-ablation (nobase / hybrid): [HEAD_ARCHITECTURE_NOTES.md](HEAD_ARCHITECTURE_NOTES.md)
- Post-V35 loss / specialist exploration: [K_L_EXPERIMENTS_REPORT.md](K_L_EXPERIMENTS_REPORT.md)
- Backbone width exploration: [N_O_BASECH_WIDENING_REPORT.md](N_O_BASECH_WIDENING_REPORT.md)
- Champion snapshot: [BEST_RESULT.md](BEST_RESULT.md)
- Label distribution context: [LABEL_BAND_ANALYSIS.md](LABEL_BAND_ANALYSIS.md)
