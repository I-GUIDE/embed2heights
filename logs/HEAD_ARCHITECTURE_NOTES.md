# Head Architecture Notes

Running log of head design experiments, especially the non-obvious findings
from ablations. Read this before making another head change — the experiments
here cost training cycles and the conclusions aren't always visible from code.

## Current head: v35head (softplus + base + deltas, 2026-04-19)

```
backbone features
      │
      ▼
 shared trunk (2-layer residual, hidden_ch)
      │
      ├─ fraction head  (3ch sigmoid, aux only — NOT submitted)
      ├─ presence head  (3ch sigmoid → submission ch 0-2, BCE on label>0)
      │
      └─ FiLM(fractions) → height_trunk
              │
              ├─ base_proj    → softplus → base
              ├─ delta_b_proj → softplus → Δ_b
              └─ delta_v_proj → softplus → Δ_v
                      │
                      h_bld = base + Δ_b   (aux L1 on building mask)
                      h_veg = base + Δ_v   (aux L1 on vegetation mask)
                      │
                      presence-gated mix → submission ch 3
                      h = p_fg·(w_b·h_bld + w_v·h_veg) + (1-p_fg)·base
```

Key design choices:
- **Presence ≠ fraction**: presence head outputs submission channels 0-2;
  fraction head is auxiliary, feeds FiLM.
- **Base + delta** (not direct per-class): confirmed by ablation — removing
  base cost 0.13 m RMSE_building.
- **Softplus, not ReLU**: confirmed by ablation — softplus's smooth gradient
  is load-bearing for height regression. ReLU variants consistently regressed
  on RMSE_bH regardless of whether base was present.
- **Presence-gated specialist blend**: each `h_bld`/`h_veg` is reliable only
  on its own class's pixels (trained that way via aux L1). The gate routes
  the submitted single height channel to whichever specialist the presence
  map says is active.

## Ablation summary (lightunet_v35head family, val)

| Variant | base | act | iou_wat | RMSE_bH | tuned score |
|---|---|---|---|---|---|
| **v35head** (current) | yes | softplus | 0.4151 | **1.9154** | **0.6897** |
| v35_nobase           | no  | ReLU     | **0.4298** | 2.0412 | 0.6890 |
| v36_hybrid           | yes | ReLU     | 0.4087 | 1.9820 | 0.6861 |

## Learning #1: softplus is load-bearing for RMSE_building

Reading the RMSE_bH column top to bottom: **1.9154 < 1.9820 < 2.0412**. The
sort order matches "how much softplus was preserved", NOT "whether base is
present":
- v35 (all softplus): best
- hybrid (base present but ReLU): +0.066 m worse
- nobase (no base + ReLU): +0.126 m worse

**My original hypothesis was wrong**: I attributed nobase's regression to
lost representational capacity (3 projections → 2) and expected hybrid to
recover v35's RMSE. It didn't. The regression tracks softplus loss, not
projection count.

**Revised hypothesis**: softplus's smooth gradient on the negative half is
what matters. ReLU has a dead zone (negative logits → zero gradient), and at
initialization many height projection outputs start near 0. The network
can't push them past the ReLU threshold for pixels that need strictly
positive heights — especially buildings (typically tall, so logits need to
climb far). Softplus keeps that gradient alive throughout training.

## Learning #2: nobase's iou_water gain was (probably) seed luck

nobase showed iou_water +0.015 at default threshold, with its optimal
threshold dropping from 0.675 → 0.500. I speculated this was a real
architectural effect: removing base_height changed gradient flow through
the shared trunk, which rippled into presence heads (especially water, the
weakest class).

Hybrid was designed to test this: it kept base but used ReLU. If my theory
were right, hybrid should behave like v35 on water (because base is back).
**Instead hybrid's iou_water = 0.4087 — worse than BOTH v35 (0.4151) and
nobase (0.4298).** If the "base-gradient-suppresses-water" mechanism were
real, hybrid should have matched v35 or been worse; it's worse than both,
which is inconsistent with any simple architectural story.

**Revised conclusion**: the water gain in nobase was most likely run-to-run
variance on a single seed. Water has the fewest positive pixels (1.95%) and
the most patches where it's entirely absent (10.6%), so its IoU is the
noisiest of the three classes across runs.

This also nullifies the "interesting side-effect" framing from the earlier
version of this doc — there's no mechanism worth reporting, just noise.

## Learning #3: simplification ≠ improvement (for this head)

I recommended a "remove fraction head + FiLM + shared_res + base" direction
earlier as a complexity cut. The nobase and hybrid experiments together are
a warning that this is probably wrong. Each simplification I tried moved
score DOWN at tuned thresholds. The current head is complex but its pieces
are earning their keep (at least within the range tested).

Future head tweaks should:
1. Get multi-seed evidence (2-3 runs) before concluding. Single-seed Δ of
   ±0.001 tuned score is in the noise floor.
2. Preserve softplus on height projections unless there's a strong reason
   to swap it.
3. Not expect "simpler is better" to hold by default here.

## Decision (2026-04-19)

Reverted to v35head (the best-measured configuration, tuned 0.6897). Moving
on from head architecture to orthogonal directions:
- Test-Time Augmentation (4-flip or D4) on v35head checkpoints
- Ensemble across backbones (already have LightUNet + HRNet + Refiner)
- Loss tweaks that don't touch architecture (e.g. BCE pos_weight)

If a future head change is tried, run at least 2 seeds per variant or don't
bother — the noise floor on val tuned score is ~0.002.
