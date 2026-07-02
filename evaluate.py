"""Decisive eval under the official GT (presence = coverage > 0.10).

Per-class fold-val IoU with a threshold sweep, plus building/vegetation height
RMSE on the cov>0.10 pixels (the official scoring set). This is the evaluation
the final submission is tuned against.

Usage: python evaluate.py <EXP_NAME> [FOLD] [PRED_SUBDIR]
  PRED_SUBDIR (default 'predictions') evaluates runs/<EXP_NAME>/<PRED_SUBDIR>.
"""
import os, sys, glob, json
import numpy as np, rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.metrics import CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT, binary_iou, build_label_map

if len(sys.argv) < 2:
    sys.exit("usage: python evaluate.py <EXP_NAME> [FOLD] [PRED_SUBDIR]")

EXP  = sys.argv[1]
FOLD = sys.argv[2] if len(sys.argv) > 2 else "0"
PRED_SUBDIR = sys.argv[3] if len(sys.argv) > 3 else "predictions"
REPO = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(REPO, "runs", EXP, PRED_SUBDIR)
LABELS   = os.path.join(os.environ.get("DATA_ROOT", os.path.join(REPO, "data")), "train", "labels")
VAL = set(json.load(open(os.path.join(REPO, f"splits/group_code_5fold_seed42/fold_{FOLD}/split.json")))["val"])
GT_COV = 0.10
CLS = [("bld", CH_BUILDING), ("veg", CH_VEGETATION), ("wat", CH_WATER)]
LM = build_label_map(LABELS)
grid = np.round(np.arange(0.20, 0.951, 0.025), 3)

def core_of(path):  # "0041_FQ_2023.npy" -> "0041_FQ"
    b = os.path.basename(path)[:-4]
    return "_".join(b.split("_")[:2])

acc = {c: {t: [] for t in grid} for c, _ in CLS}
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
    for c, ch in CLS:
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
for c, _ in CLS:
    bt = max(grid, key=lambda t: np.mean(acc[c][t]))
    bi = np.mean(acc[c][bt])
    print(f"  {c}: best thr {bt:.3f} -> IoU {bi:.4f}")
for tag, norm in (("bH", 3.0), ("vH", 5.0)):
    if rmse_acc[tag]:
        r = float(np.mean(rmse_acc[tag]))
        print(f"  RMSE_{tag}: {r:.4f} m  (captured {max(0.0, 1 - r / norm):.4f})")
