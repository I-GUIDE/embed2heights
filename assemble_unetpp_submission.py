"""Assemble the 3-seed delmask+U-Net++ submission.
  seg ch0-2 = mean of the 15 stage-1 models' TEST seg, binarized at OOF-tuned thr
  height ch3 = mean of the 15 PURIFY models' TEST height
OOF (val) is used ONLY to tune thresholds (leave-fold-out: tile in fold F uses the
3 seeds of fold F). Test uses all 15 members.
"""
import os, sys, glob, json, zipfile
import numpy as np
import rasterio
from core.inference.calibration import apply_water_cc_filter
from core.metrics import binary_iou   # LEADERBOARD empty-case convention (empty->1.0)

WATER_K = 4   # water connected-component min-size filter (matches dmwav recipe)

REPO = "/u/dingqi2/workspace/esa/embed2heights/.worktrees/combo-stack"
CFG = "xfusion_095_p3_2stage_softbin_covgt10_delmask_unetpp"
LABELS = "/u/dingqi2/workspace/esa/data/train/labels"
SPLITS = f"{REPO}/splits/group_code_5fold_seed42"
SEEDS, FOLDS = [0, 1, 2], [0, 1, 2, 3, 4]
GT_COV = 0.10
CH = {"bld": 0, "veg": 1, "wat": 2}
PROBS = np.round(np.arange(0.30, 0.86, 0.025), 3)
OUT_DIR = f"{REPO}/submission"; os.makedirs(OUT_DIR, exist_ok=True)


def biou(pred, gt):
    return binary_iou(pred, gt)   # leaderboard convention (empty tile -> 1.0)


def lab_path(cid):
    g = glob.glob(f"{LABELS}/label_{cid}_*.tif") + glob.glob(f"{LABELS}/*{cid}*.tif")
    return g[0]


# ---------- Part 1: OOF threshold tuning (leave-fold-out, 3-seed per fold) ----------
print("[oof] tuning thresholds on val (3-seed per fold)...", flush=True)
acc = {c: {p: [] for p in PROBS} for c in CH}
rmse = {"bH": [], "vH": []}
for F in FOLDS:
    val = json.load(open(f"{SPLITS}/fold_{F}/split.json"))["val"]
    for cid in val:
        segs, h3s = [], []
        for S in SEEDS:
            sp = f"{REPO}/runs/{CFG}_s{S}_f{F}/predictions/{cid}.npy"
            pp = f"{REPO}/runs/{CFG}_s{S}_f{F}_purify/predictions/{cid}.npy"
            if os.path.exists(sp):
                segs.append(np.load(sp).astype(np.float32))
            if os.path.exists(pp):
                h3s.append(np.load(pp).astype(np.float32)[3])
        if not segs:
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
                    m = apply_water_cc_filter(m, WATER_K)   # water cc filter
                acc[c][p].append(biou(m, gt))
        if h3s:
            h3 = np.mean(h3s, 0)[:H, :W]
            for tag, ch in (("bH", 0), ("vH", 1)):
                m = lab[ch] > GT_COV
                if m.sum() > 0:
                    d = (h3 - lab[3, :H, :W])[m]
                    rmse[tag].append(float(np.sqrt((d * d).mean())))

THR = {}
print("[oof] per-class best threshold:")
for c in CH:
    bp = max(PROBS, key=lambda p: np.mean(acc[c][p]))
    THR[c] = float(bp)
    print(f"   {c}: thr {bp:.3f} -> mIoU {np.mean(acc[c][bp]):.4f}")
print(f"[oof] RMSE_bH {np.mean(rmse['bH']):.4f}  RMSE_vH {np.mean(rmse['vH']):.4f}")
SCORE = (0.25 * np.mean(acc['bld'][THR['bld']]) + 0.15 * np.mean(acc['veg'][THR['veg']])
         + 0.15 * np.mean(acc['wat'][THR['wat']])
         + 0.25 * max(0, 1 - np.mean(rmse['bH']) / 3) + 0.20 * max(0, 1 - np.mean(rmse['vH']) / 5))
print(f"[oof] estimated OOF score = {SCORE:.4f}  (thr bld{THR['bld']}/veg{THR['veg']}/wat{THR['wat']})")

if "--oof-only" in sys.argv:
    sys.exit(0)

# ---------- Part 2: TEST ensemble (all 15 members) + zip ----------
test_tiles = sorted(os.path.basename(p) for p in
                    glob.glob(f"{REPO}/runs/{CFG}_s0_f0/test_predictions/*.npy"))
print(f"\n[test] {len(test_tiles)} tiles; ensembling 15 members...", flush=True)
ZIP = f"{OUT_DIR}/unetpp_3seed_b{THR['bld']}_v{THR['veg']}_w{THR['wat']}.zip"
tmp = f"{OUT_DIR}/_asm_unetpp"; os.makedirs(f"{tmp}/predictions", exist_ok=True)
missing = 0
for t in test_tiles:
    segs, h3s = [], []
    for S in SEEDS:
        for F in FOLDS:
            sp = f"{REPO}/runs/{CFG}_s{S}_f{F}/test_predictions/{t}"
            pp = f"{REPO}/runs/{CFG}_s{S}_f{F}_purify/test_predictions/{t}"
            if os.path.exists(sp):
                segs.append(np.load(sp).astype(np.float32)[:3])
            if os.path.exists(pp):
                h3s.append(np.load(pp).astype(np.float32)[3])
    if len(segs) != 15 or len(h3s) != 15:
        missing += 1
    seg = np.mean(segs, 0)
    h3 = np.mean(h3s, 0)
    out = np.zeros((4, seg.shape[1], seg.shape[2]), np.float32)
    out[0] = (seg[0] >= THR['bld']).astype(np.float32)
    out[1] = (seg[1] >= THR['veg']).astype(np.float32)
    out[2] = apply_water_cc_filter(seg[2] >= THR['wat'], WATER_K).astype(np.float32)
    out[3] = h3
    np.save(f"{tmp}/predictions/{t}", out)
print(f"[test] assembled {len(test_tiles)} tiles ({missing} with <15 members)")

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(glob.glob(f"{tmp}/predictions/*.npy")):
        z.write(p, os.path.join("predictions", os.path.basename(p)))
n = len(zipfile.ZipFile(ZIP).namelist())
print(f"\n==== SUBMISSION READY ====\n  {ZIP}\n  tiles in zip: {n}")
