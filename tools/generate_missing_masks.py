"""Detect under-labelled building regions and export per-tile "delmask" masks.

Some training tiles have building footprints that were (we believe) deleted from
the GT: AlphaEarth + nDSM say "building" but the label channel is empty there.
Training zeroes the presence/seg loss on these pixels so the model isn't punished
for correctly predicting a building (height loss is kept). See README section 5.

Detector (region-level density gap, robust to salt-and-pepper):
  E(x) = building EVIDENCE, label-independent:  height > H_THR & p_build > SIG_THR
  L(x) = labelled building:                      bld > GT
  gap  = boxblur(E, R) - boxblur(L, R)           # high where evidence >> labels
  flag = gap > GAP_THR  &  no class labelled here  &  the pixel is itself raised
  then morphological-close and keep connected regions >= AREA_MIN px.
`p_build` is a logistic-regression building-vs-vegetation classifier fit on
L2-normalised AlphaEarth embeddings (seeded, so the run is deterministic).

Writes runs/missing_masks/<core>.npy (uint8, label-native HxW) for each flagged
tile; only flagged tiles get a file (absent file = no masking). The precomputed
masks already ship in the repo; this script regenerates them from scratch.

Usage:
    python tools/generate_missing_masks.py            # generate masks
    python tools/generate_missing_masks.py --report   # + ranked summary of hits
Env: DATA_ROOT (default <repo>/data) -> DATA_ROOT/train/{labels,alphaearth_emb}
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from core.metrics import build_label_map

# Detector thresholds.
GT, H_THR, SIG_THR = 0.10, 3.0, 0.60   # class-coverage, nDSM height, building-signature prob
R, GAP_THR, AREA_MIN, H_GUARD = 6, 0.25, 80, 2.0   # blur radius, density-gap, region size, per-px height guard
ST = ndimage.generate_binary_structure(2, 2)

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "data"))
LABELS = os.path.join(DATA_ROOT, "train", "labels")
AE = os.path.join(DATA_ROOT, "train", "alphaearth_emb")
OUT = os.path.join(REPO, "runs", "missing_masks")


def build_ae_index(ae_dir):
    idx = {}
    for p in glob.glob(os.path.join(ae_dir, "*.tif")):
        m = re.search(r"(\d{4}_[A-Z]{2})", os.path.basename(p))
        if m:
            idx[m.group(1)] = p
    return idx


def read_tile(label_path, ae_path):
    with rasterio.open(label_path) as s:
        lab = s.read().astype(np.float32)
    with rasterio.open(ae_path) as s:
        e = np.nan_to_num(s.read().astype(np.float32), nan=0.0)
    h = min(lab.shape[1], e.shape[1])
    w = min(lab.shape[2], e.shape[2])
    return lab, e, h, w


def fit_building_signature(label_map, ae_index, n_tiles=60):
    """Logistic-regression building-vs-vegetation classifier on L2-normalised AE."""
    rng = np.random.default_rng(0)
    Xb, Xv = [], []
    for t in rng.choice(sorted(label_map), n_tiles, replace=False):
        if t not in ae_index:
            continue
        lab, e, h, w = read_tile(label_map[t], ae_index[t])
        lab, e = lab[:, :h, :w], e[:, :h, :w]
        eu = e.reshape(e.shape[0], -1).T
        eu = eu / (np.linalg.norm(eu, axis=1, keepdims=True) + 1e-6)
        bi = np.flatnonzero((lab[0] > 0.5).ravel())
        vi = np.flatnonzero((lab[1] > 0.5).ravel())
        if len(bi):
            Xb.append(eu[rng.choice(bi, min(300, len(bi)), replace=False)])
        if len(vi):
            Xv.append(eu[rng.choice(vi, min(300, len(vi)), replace=False)])
    Xb, Xv = np.vstack(Xb), np.vstack(Xv)
    y = np.r_[np.ones(len(Xb)), np.zeros(len(Xv))]
    clf = LogisticRegression(max_iter=300).fit(np.vstack([Xb, Xv]), y)
    return clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0])


def p_build(e, weight, bias):
    eu = e.reshape(e.shape[0], -1).T
    eu = eu / (np.linalg.norm(eu, axis=1, keepdims=True) + 1e-6)
    return (1.0 / (1.0 + np.exp(-(eu @ weight + bias)))).reshape(e.shape[1], e.shape[2])


def _boxblur(x, r):
    return ndimage.uniform_filter(x.astype(np.float32), size=2 * r + 1, mode="nearest")


def detect(lab, e, weight, bias):
    """Return a boolean missing-building mask for one (label, embedding) tile."""
    bld, veg, wat, hgt = lab
    evidence = ((hgt > H_THR) & (p_build(e, weight, bias) > SIG_THR)).astype(np.float32)
    gap = _boxblur(evidence, R) - _boxblur((bld > GT).astype(np.float32), R)
    flag = (gap > GAP_THR) & (bld <= GT) & (veg <= GT) & (wat <= GT) & (hgt > H_GUARD)
    flag = ndimage.binary_closing(flag, structure=ST, iterations=1)
    labelled, n = ndimage.label(flag, ST)
    if not n:
        return np.zeros_like(flag, bool)
    sizes = ndimage.sum(np.ones_like(labelled), labelled, range(1, n + 1))
    keep = np.flatnonzero(sizes >= AREA_MIN) + 1
    return np.isin(labelled, keep) if len(keep) else np.zeros_like(flag, bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="Print a ranked summary of flagged tiles at the end.")
    args = ap.parse_args()

    label_map = build_label_map(LABELS)
    ae_index = build_ae_index(AE)
    print(f"fitting building signature on 60 tiles ...", flush=True)
    weight, bias = fit_building_signature(label_map, ae_index)
    os.makedirs(OUT, exist_ok=True)

    tiles = sorted(label_map)
    hits, total_px = [], 0
    for i, t in enumerate(tiles):
        if t not in ae_index:
            continue
        lab, e, h, w = read_tile(label_map[t], ae_index[t])
        lh, lw = lab.shape[1], lab.shape[2]
        m = detect(lab[:, :h, :w], e[:, :h, :w], weight, bias)
        if m.sum() >= AREA_MIN:
            full = np.zeros((lh, lw), np.uint8)   # label-native shape
            full[:h, :w] = m.astype(np.uint8)
            np.save(os.path.join(OUT, f"{t}.npy"), full)
            hits.append((t, int(m.sum())))
            total_px += int(m.sum())
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(tiles)} scanned, {len(hits)} masks", flush=True)

    print(f"saved {len(hits)} masks ({total_px:,} px) to {OUT}/")
    if args.report:
        hits.sort(key=lambda x: -x[1])
        print(f"flagged {len(hits)}/{len(tiles)} tiles ({100 * len(hits) / len(tiles):.1f}%)")
        print("top 20:", [(t, px) for t, px in hits[:20]])


if __name__ == "__main__":
    main()
