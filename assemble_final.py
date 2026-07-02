"""Assemble the FINAL submission (public 0.5067) from the 25 trained member-folds.

  seg   ch0-2 = mean of 50 test predictions  (25 clDice + 25 seg-purify),
               binarised at OOF-tuned per-class thresholds (+ water CC filter).
  height ch3  = mean of 25 height-purify test predictions, then
               building pixels: DE-COMPRESS  h -> 1.05*h + 0.116
               veg pixels:      ADDITIVE      h -> h + 0.12

This is a thin orchestrator: it loads the predictions and delegates all
post-processing (threshold sweep, seg binarisation, height calibration, zipping)
to core.inference.postprocess. Thresholds are tuned on out-of-fold validation
predictions only (no public board).

Run:  python assemble_final.py   ->   submission/FINAL_*.zip
"""
import os
import glob
import json

import numpy as np
import rasterio

from core.inference.postprocess import (
    assemble_tile,
    sweep_class_thresholds,
    write_submission_zip,
)

REPO = os.environ.get("REPO", os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "data"))
LABELS = os.path.join(DATA_ROOT, "train", "labels")
SPLITS = os.path.join(REPO, "splits", "group_code_5fold_seed42")

# 5 ensemble members = (config, seed)
CFG = "xfusion_095_"
MEMBERS = [(CFG + "unetpp", 0), (CFG + "unetpp", 1), (CFG + "unetpp", 2),
           (CFG + "unet3plus", 0), (CFG + "unetpp_trans", 0)]
SEG_SUBS = ["cldice", "segpurify"]          # 50-member seg (25 each)
FOLDS = [0, 1, 2, 3, 4]
GT_COV, WATER_K = 0.10, 4
PROBS = np.round(np.arange(0.20, 0.951, 0.025), 3)
A_BLD, B_BLD, VEG_SHIFT = 1.05, 0.116, 0.12  # bld de-compression, veg additive shift
# OOF val: a tile is predicted only by its own held-out fold's models.
N_SEG_VAL = len(MEMBERS) * len(SEG_SUBS)                    # 10
# Test: every member-fold predicts every tile.
N_SEG_TEST = len(MEMBERS) * len(FOLDS) * len(SEG_SUBS)     # 50
N_HEIGHT_TEST = len(MEMBERS) * len(FOLDS)                  # 25
OUT = os.path.join(REPO, "submission")
os.makedirs(OUT, exist_ok=True)


def lab_path(cid):
    g = glob.glob(f"{LABELS}/label_{cid}_*.tif") + glob.glob(f"{LABELS}/*{cid}*.tif")
    return g[0]


def run(base, seed, fold, stage, sub="predictions"):
    return f"{REPO}/runs/{base}_s{seed}_f{fold}_{stage}/{sub}"


def _load_seg(base, seed, fold, tile, sub):
    """Mean of the SEG_SUBS (clDice + seg-purify) seg maps for one member-fold, or None."""
    segs = []
    for stage in SEG_SUBS:
        p = f"{run(base, seed, fold, stage, sub)}/{tile}"
        if os.path.exists(p):
            segs.append(np.load(p).astype(np.float32)[:3])
    return segs


# ---- 1) tune per-class thresholds on the 50-member OOF-val seg ----
print("[oof] tuning thresholds on the 50-member (clDice + seg-purify) seg ...", flush=True)
pairs = []
for F in FOLDS:
    for cid in json.load(open(f"{SPLITS}/fold_{F}/split.json"))["val"]:
        segs = [s for base, seed in MEMBERS for s in _load_seg(base, seed, F, f"{cid}.npy", "predictions")]
        if len(segs) != N_SEG_VAL:
            continue
        seg = np.mean(segs, 0)
        with rasterio.open(lab_path(cid)) as s:
            lab = s.read().astype(np.float32)
        h = min(seg.shape[1], lab.shape[1]); w = min(seg.shape[2], lab.shape[2])
        pairs.append((seg[:, :h, :w], lab[:, :h, :w]))
THR = sweep_class_thresholds(pairs, PROBS, gt_cov=GT_COV, water_k=WATER_K)
thresholds = (THR[0], THR[1], THR[2])
print(f"   thresholds bld/veg/wat = {thresholds}")

# ---- 2) assemble the test submission ----
tiles = sorted(os.path.basename(p) for p in
               glob.glob(f"{run(MEMBERS[0][0], MEMBERS[0][1], 0, 'cldice', 'test_predictions')}/*.npy"))
print(f"\n[test] {len(tiles)} tiles; 50-seg + 25 height-purify "
      f"(bld-decomp {A_BLD}*h+{B_BLD}, veg+{VEG_SHIFT}) ...", flush=True)
tmp = f"{OUT}/_asm_final"
os.makedirs(f"{tmp}/predictions", exist_ok=True)
miss = 0
for t in tiles:
    segs, h3s = [], []
    for base, seed in MEMBERS:
        for F in FOLDS:
            segs += _load_seg(base, seed, F, t, "test_predictions")
            pp = f"{run(base, seed, F, 'purify', 'test_predictions')}/{t}"
            if os.path.exists(pp):
                h3s.append(np.load(pp).astype(np.float32)[3])
    if len(segs) != N_SEG_TEST or len(h3s) != N_HEIGHT_TEST:
        miss += 1
    out = assemble_tile(np.mean(segs, 0), np.mean(h3s, 0), thresholds,
                        water_k=WATER_K, a_bld=A_BLD, b_bld=B_BLD, veg_shift=VEG_SHIFT)
    np.save(f"{tmp}/predictions/{t}", out)

ZIP = f"{OUT}/FINAL_50seg_blddecomp_b{THR[0]}_v{THR[1]}_w{THR[2]}.zip"
n = write_submission_zip(f"{tmp}/predictions", ZIP)
print(f"\n==== SUBMISSION READY ====\n  {ZIP}\n  {n} tiles ({miss} with <full members)")
print(f"  thr bld{THR[0]}/veg{THR[1]}/wat{THR[2]}  bld-decomp {A_BLD}h+{B_BLD}  veg+{VEG_SHIFT}")
