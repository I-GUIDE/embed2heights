"""Assemble the FINAL submission (public 0.5067) from the 25 trained member-folds.

  seg   ch0-2 = mean of 50 test predictions  (25 clDice + 25 seg-purify),
               binarised at OOF-tuned per-class thresholds (+ water CC filter).
  height ch3  = mean of 25 height-purify test predictions, then
               building pixels: DE-COMPRESS  h -> 1.05*h + 0.116   (a=1.05)
               veg pixels:      ADDITIVE      h -> h + 0.12

Thresholds are tuned on the out-of-fold validation predictions only (no public
board). Paths are all relative to this file's directory (the repo root); set
DATA_ROOT / REPO env vars only if your layout differs.

Run:  python assemble_final.py   ->   submission/FINAL_*.zip
"""
import os, glob, json, zipfile
import numpy as np, rasterio
from core.inference.calibration import apply_water_cc_filter
from core.metrics import binary_iou

REPO = os.environ.get("REPO", os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "data"))
LABELS = os.path.join(DATA_ROOT, "train", "labels")
SPLITS = os.path.join(REPO, "splits", "group_code_5fold_seed42")

# 5 ensemble members = (config, seed)
CFG = "xfusion_095_p3_2stage_softbin_covgt10_delmask_"
MEMBERS = [(CFG+"unetpp", 0), (CFG+"unetpp", 1), (CFG+"unetpp", 2),
           (CFG+"unet3plus", 0), (CFG+"unetpp_trans", 0)]
SEG_SUBS = ["cldice", "segpurify"]     # 50-member seg (25 each)
FOLDS = [0, 1, 2, 3, 4]
GT_COV = 0.10; WATER_K = 4; CH = {"bld": 0, "veg": 1, "wat": 2}
PROBS = np.round(np.arange(0.20, 0.951, 0.025), 3)
A_BLD, B_BLD, VEG_SHIFT = 1.05, 0.116, 0.12   # bld de-compression, veg additive
OUT = os.path.join(REPO, "submission"); os.makedirs(OUT, exist_ok=True)


def lab_path(cid):
    g = glob.glob(f"{LABELS}/label_{cid}_*.tif") + glob.glob(f"{LABELS}/*{cid}*.tif")
    return g[0]


def run(base, seed, fold, stage, sub="predictions"):
    return f"{REPO}/runs/{base}_s{seed}_f{fold}_{stage}/{sub}"


# ---- 1) tune per-class thresholds on OOF val (50-member seg) ----
print("[oof] tuning thresholds on the 50-member (clDice + seg-purify) seg ...", flush=True)
acc = {c: {p: [] for p in PROBS} for c in CH}
for F in FOLDS:
    for cid in json.load(open(f"{SPLITS}/fold_{F}/split.json"))["val"]:
        segs = []
        for base, s in MEMBERS:
            for stage in SEG_SUBS:
                p = f"{run(base, s, F, stage)}/{cid}.npy"
                if os.path.exists(p):
                    segs.append(np.load(p).astype(np.float32)[:3])
        if len(segs) != len(MEMBERS) * len(SEG_SUBS):
            continue
        seg = np.mean(segs, 0)
        with rasterio.open(lab_path(cid)) as s:
            lab = s.read().astype(np.float32)
        H = min(seg.shape[1], lab.shape[1]); W = min(seg.shape[2], lab.shape[2])
        seg = seg[:, :H, :W]; lab = lab[:, :H, :W]
        for c, ch in CH.items():
            gt = lab[ch] > GT_COV
            for p in PROBS:
                m = seg[ch] >= p
                if c == "wat":
                    m = apply_water_cc_filter(m, WATER_K)
                acc[c][p].append(binary_iou(m, gt))
THR = {}
for c in CH:
    THR[c] = float(max(PROBS, key=lambda p: np.mean(acc[c][p])))
    print(f"   {c}: thr {THR[c]}  mIoU {np.mean(acc[c][THR[c]]):.4f}")

# ---- 2) assemble the test submission ----
tiles = sorted(os.path.basename(p) for p in
               glob.glob(f"{run(MEMBERS[0][0], MEMBERS[0][1], 0, 'cldice', 'test_predictions')}/*.npy"))
print(f"\n[test] {len(tiles)} tiles; 50-seg + 25 height-purify "
      f"(bld-decomp {A_BLD}*h+{B_BLD}, veg+{VEG_SHIFT}) ...", flush=True)
ZIP = f"{OUT}/FINAL_50seg_blddecomp_b{THR['bld']}_v{THR['veg']}_w{THR['wat']}.zip"
tmp = f"{OUT}/_asm_final"; os.makedirs(f"{tmp}/predictions", exist_ok=True)
miss = 0
for t in tiles:
    segs, h3s = [], []
    for base, s in MEMBERS:
        for F in FOLDS:
            for stage in SEG_SUBS:
                sp = f"{run(base, s, F, stage, 'test_predictions')}/{t}"
                if os.path.exists(sp):
                    segs.append(np.load(sp).astype(np.float32)[:3])
            pp = f"{run(base, s, F, 'purify', 'test_predictions')}/{t}"
            if os.path.exists(pp):
                h3s.append(np.load(pp).astype(np.float32)[3])
    if len(segs) != 50 or len(h3s) != 25:
        miss += 1
    seg = np.mean(segs, 0); h3 = np.mean(h3s, 0)
    out = np.zeros((4, seg.shape[1], seg.shape[2]), np.float32)
    out[0] = (seg[0] >= THR['bld']).astype(np.float32)
    out[1] = (seg[1] >= THR['veg']).astype(np.float32)
    out[2] = apply_water_cc_filter(seg[2] >= THR['wat'], WATER_K).astype(np.float32)
    h = h3.copy()
    bm = out[0] > 0.5; vm = (out[1] > 0.5) & (~bm)
    h[bm] = A_BLD * h3[bm] + B_BLD     # building de-compression
    h[vm] = h3[vm] + VEG_SHIFT         # veg additive shift
    out[3] = h
    np.save(f"{tmp}/predictions/{t}", out)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(glob.glob(f"{tmp}/predictions/*.npy")):
        z.write(p, os.path.join("predictions", os.path.basename(p)))
n = len(zipfile.ZipFile(ZIP).namelist())
print(f"\n==== SUBMISSION READY ====\n  {ZIP}\n  {n} tiles ({miss} with <full members)")
print(f"  thr bld{THR['bld']}/veg{THR['veg']}/wat{THR['wat']}  "
      f"bld-decomp {A_BLD}h+{B_BLD}  veg+{VEG_SHIFT}")
