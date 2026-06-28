# 3-Seed delmask + U-Net++ submission

Reproduction reference for the current best submission.

```
repo:   github.com/I-GUIDE/embed2heights
branch: exp/unetpp-3seed-submission   (commit 61805dd)
config: configs/active/xfusion_095_p3_2stage_softbin_covgt10_delmask_unetpp.yml
```

**Public leaderboard:** SCORE **0.5018** | IOU_BUILD 0.4979 / VEG 0.8195 / WATER 0.5211 | RMSE_H_BUILD 1.8070 / RMSE_H_VEG 3.0792
(beats the previous covgt10 best 0.4979; building IoU and veg height gains both transferred local→public.)

---

## Architecture (`model_type: xfusion_unet_hybrid_cross_source`)

Dual pixel source + 4 token streams, fused into a split-trunk multi-task head.

- **AlphaEarth (64ch) and Tessera (128ch) each run their own `LightUNetPP`** — the nested U-Net++ decoder, `base_ch=48`, ~12.7M params. **This is the key change of this submission: `pixel_backbone_kind` switched from `unet` to `unetpp`.**
- The two pixel branches are merged by a **gated fusion** (`gate_mode=rich`, `gate_init_bias=4.0`).
- **Token fusion**: TerraMind S1/S2 + Thor S1/S2 (4 × 768×16×16) via `CrossSourceHybridFiLMFusion`, `token_calibration=true` (sources `[0,1]`).
- **Split-trunk multi-task head** (`split_trunk=true`; presence and height have independent trunks):
  - presence head: `split_all` (per-class), `presence_tower_depth=2`, `presence_branch_ch=48`
  - height head: `softbin` (64 bins, max 80 m, log spacing), `height_specialist_depth=1`, independent height branch (`height_independent_branches=true`)

## The 3 effective levers (all required)

1. **U-Net++ backbone** (`pixel_backbone_kind: unetpp`) — the new gain (building IoU +0.005, veg height −0.034 m; both held on the public board). Light nested multi-scale fusion generalizes better than plain LightUNet; heavier/attention backbones do NOT help on these GFM embeddings.
2. **delmask** (`missing_building_mask_dir: ${REPO_DIR}/runs/missing_masks`) — drop the presence/seg loss on suspected human-deleted building footprints (height loss kept).
3. **cov>0.10 GT alignment** (`presence_coverage_threshold: 0.1`) — supervise presence against the official GT contour (`coverage > 0.10`), not the legacy argmax+any-present. This is the single biggest metric-definition fix.

## Training (two stages)

**Stage 1 (80 ep, coupled):** `batch=16, lr=2e-4, weight_decay=1e-4, height_loss_kind=l1`
- `build_height_boost=5.0`, `veg_height_boost=1.5`, `aux_weight=1.0`
- `height_bin_aux_weight=0.5`, `height_bin_sigma_bins=1.5`
- `building_ring_presence_alpha=2.0`, `building_ring_kernel=5`, `building_boundary_weight=0.25`
- `water_empty_topk=512`, `weight_water_empty_topk=0.03`, `tversky_water_alpha=0.3`
- output: `model_best.pth` = segmentation (channels 0–2)

**Stage 2 purify (20 ep):** `lr=1.5e-4`, `--presence-trunk-grad-scale 0.0` (freeze the presence trunk, train only the channel-3 height) — output `model_best.pth` = height (channel 3).

**Ensemble:** 3 seeds × 5 folds (leave-region-out, `splits/group_code_5fold_seed42`) = 15 members.

## Submission recipe (`assemble_unetpp_submission.py`)

- seg (ch0-2) = **mean of the 15 stage-1 members'** test predictions, binarized at **bld 0.60 / veg 0.55 / wat 0.70 + water connected-component min-size filter k=4**.
- height (ch3) = **mean of the 15 purify members'** test predictions.
- Thresholds are tuned on OOF (leave-fold-out, `binary_iou` with the leaderboard empty-tile convention: a tile with no GT and no prediction for a class counts as IoU=1.0).

## How to run

```bash
# 1. train: 3 seeds x 5 folds, full two-stage (stage-1 seg + purify height), node-local NVMe staging
sbatch array_delmask_unetpp.sbatch          # SLURM array 0-14

# 2. predict all 15 members on the test set (data/test)
sbatch array_testpred_unetpp.sbatch         # SLURM array 0-14

# 3. OOF threshold tuning + 15-member test ensemble + zip
python assemble_unetpp_submission.py
#   -> submission/unetpp_3seed_b0.6_v0.55_w0.7.zip   (946 tiles, [4,256,256] per tile)
```

Per-fold cov>0.10 evaluation (sanity): `python cov0p10_eval.py <EXP_NAME> <FOLD>`.
The delmask masks live in `runs/missing_masks` (per-tile `<core>.npy`).

## Tried and refuted (do NOT redo)

- ❌ **wavelet downsampling** (Haar-DWT in the encoder): looked good on fold0 but washed out at 5-fold; veg height even worse than baseline.
- ❌ **deep supervision** on the U-Net++ nested nodes: neutral (~+0.0004, within seed noise) — the nesting already captures the multi-scale benefit.
- ❌ **TransUNet bottleneck self-attention** (transformer on X^3_0): worse on all 3 metrics (20.6M params, overfits).
- ❌ **ViT / SegFormer / HRNet / UPerNet / CBAM / Attention-UNet** backbones: all lost to LightUNet/U-Net++ — heavy/attention encoders are redundant on already-high-level GFM embeddings.

## Local (OOF) vs public — where the gap is

| metric | local OOF | public | note |
|---|---|---|---|
| SCORE | 0.5345 | 0.5018 | −0.033 gap, **entirely from height** |
| IOU_BUILD | 0.4969 | 0.4979 | transfers (≈equal) |
| IOU_VEG | 0.7687 | 0.8195 | public favorable |
| IOU_WATER | 0.5307 | 0.5211 | ≈equal |
| RMSE_H_BUILD | 1.4898 | 1.8070 | +0.32 m distribution shift |
| RMSE_H_VEG | 2.7617 | 3.0792 | +0.32 m distribution shift |

All three IoUs transfer local→public; the OOF→public score gap is a genuine **height distribution shift** (test = held-out region/biome), not an overfit artifact of this model.
