"""Decisive eval under the official GT (coverage>0.10): per-class fold-val IoU with
threshold sweep. Building gate baseline (current softbin presence, fold0) = 0.4663.
Usage: python cov0p10_eval.py <EXP_NAME> [FOLD] [PRED_SUBDIR]
  PRED_SUBDIR (default 'predictions') lets you eval an alternate dir, e.g.
  'predictions_frac' for the --seg-from-fraction coverage seg.
  NOTE: building presence baseline (covgt10 softbin, fold0, best-thr) = 0.4795
  (seeds 0/1/2 = 0.4795/0.4823/0.4793, spread ~0.003). The old 0.4663 annotation
  was STALE and produced a false +0.0168 read on covhead (2026-06-25); use 0.4795."""
import os, sys, glob, json
import numpy as np, rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.metrics import CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT, binary_iou, build_label_map

EXP  = sys.argv[1]
FOLD = sys.argv[2] if len(sys.argv) > 2 else "0"
PRED_SUBDIR = sys.argv[3] if len(sys.argv) > 3 else "predictions"
REPO = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(REPO, "runs", EXP, PRED_SUBDIR)
LABELS   = "/u/dingqi2/workspace/esa/data/train/labels"
VAL = set(json.load(open(os.path.join(REPO, f"splits/group_code_5fold_seed42/fold_{FOLD}/split.json")))["val"])
GT_COV = 0.10
CLS = [("bld", CH_BUILDING, 0.4795), ("veg", CH_VEGETATION, None), ("wat", CH_WATER, None)]
LM = build_label_map(LABELS)
grid = np.round(np.arange(0.20, 0.951, 0.025), 3)

def core_of(path):  # "0041_FQ_2023.npy" -> "0041_FQ"
    b = os.path.basename(path)[:-4]
    return "_".join(b.split("_")[:2])

acc = {c: {t: [] for t in grid} for c, _, _ in CLS}
rmse_acc = {"bH": [], "vH": []}   # height RMSE on cov>0.10 pixels (official scoring set)
n = 0
for pf in sorted(glob.glob(os.path.join(PRED_DIR, "*.npy"))):
    core = core_of(pf)
    if core not in VAL or core not in LM:
        continue
    pred = np.load(pf)
    with rasterio.open(LM[core]) as s:
        lab = s.read().astype(np.float32)
    h = min(pred.shape[1], lab.shape[1]); w = min(pred.shape[2], lab.shape[2])
    pred = pred[:, :h, :w]; lab = lab[:, :h, :w]; n += 1
    for c, ch, _ in CLS:
        gt = lab[ch] > GT_COV
        prob = pred[ch]
        for t in grid:
            acc[c][t].append(binary_iou(prob > t, gt))
    th = lab[CH_HEIGHT]
    for tag, ch in (("bH", CH_BUILDING), ("vH", CH_VEGETATION)):
        m = lab[ch] > GT_COV
        if m.sum() > 0:
            d = pred[CH_HEIGHT][m] - th[m]
            rmse_acc[tag].append(float(np.sqrt((d * d).mean())))

print(f"\n==== cov>0.10 eval  EXP={EXP}  fold{FOLD}  n={n} val tiles ====")
for c, _, base in CLS:
    bt = max(grid, key=lambda t: np.mean(acc[c][t]))
    bi = np.mean(acc[c][bt])
    tag = ""
    if base is not None:
        tag = f"   [baseline {base:.4f}  ->  {'BEATS +' if bi > base else 'below '}{bi-base:+.4f}]"
    print(f"  {c}: best thr {bt:.3f} -> IoU {bi:.4f}{tag}")
for tag, norm in (("bH", 3.0), ("vH", 5.0)):
    if rmse_acc[tag]:
        r = float(np.mean(rmse_acc[tag]))
        print(f"  RMSE_{tag}: {r:.4f} m  (captured {max(0.0, 1 - r / norm):.4f})")
print("  (building must clearly beat 0.4663 to justify scaling to full 5fold x 3seed)")
