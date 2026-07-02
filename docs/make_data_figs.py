"""Render the two data-observation figures used in framework_overview.tex:

  docs/fig_height_holes.pdf    -- nDSM holes: height=0 on labelled objects
  docs/fig_building_holes.pdf  -- deleted footprints: label empty, nDSM says building

Replicates the hole/mask logic in core/data/datasets.py exactly. The building
figure overlays the shipped delmasks in runs/missing_masks/.

Usage:  python docs/make_data_figs.py
Env:    DATA_ROOT (default <repo>/data) -> DATA_ROOT/train/labels
"""
import glob
import os

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.environ.get("DATA_ROOT", os.path.join(REPO, "data")), "train", "labels")
MASKS = os.path.join(REPO, "runs", "missing_masks")
OUT = os.path.join(REPO, "docs")

HEIGHT_TILES = ["1359_EO", "1290_VS", "1324_JA", "1363_FN"]
BLD_TILES = ["1454_KE", "1455_KE", "1456_KE", "1556_GD"]


def load(core):
    path = glob.glob(os.path.join(DATA, f"label_{core}_*.tif"))[0]
    with rasterio.open(path) as src:
        raw = src.read().astype(np.float32)
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)  # [bld,veg,wat,height]


def masks(raw):
    has_lc = (raw[0] > 0) | (raw[1] > 0) | (raw[2] > 0)
    ndsm_hole = (raw[3] == 0) & has_lc
    global_valid = ~np.all(raw == 0, axis=0)
    return global_valid, has_lc, ndsm_hole


# ---------- Figure A: nDSM holes ----------
fig, ax = plt.subplots(2, 4, figsize=(12, 6.3))
for j, core in enumerate(HEIGHT_TILES):
    raw = load(core); _, has_lc, hole = masks(raw)
    sel = raw[3] > 0
    vmax = np.percentile(raw[3][sel], 99) if sel.any() else 1
    # show the raw label as stored: the label writes a literal 0, so 0 renders as 0
    im = ax[0, j].imshow(raw[3], cmap="viridis", vmin=0, vmax=max(vmax, 1))
    ax[0, j].set_title(core, fontsize=11)
    plt.colorbar(im, ax=ax[0, j], fraction=0.046, pad=0.04)
    ov = np.zeros((*raw.shape[1:], 3), np.float32)
    ov[has_lc] = (0.82, 0.82, 0.82); ov[hole] = (0.86, 0.15, 0.15)
    ax[1, j].imshow(ov)
    ax[1, j].set_title(f"nDSM hole: {100.0 * hole.sum() / max(has_lc.sum(), 1):.0f}% of land cover",
                       fontsize=9)
    for a in (ax[0, j], ax[1, j]):
        a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_ylabel("height (m)", fontsize=10)
ax[1, 0].set_ylabel("hole (red)", fontsize=10)
fig.suptitle("Observation A: nDSM holes -- height is 0 where objects clearly exist "
             "(land cover labelled)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(OUT, "fig_height_holes.pdf"), bbox_inches="tight")
print("wrote fig_height_holes.pdf")

# ---------- Figure B: deleted building footprints ----------
fig, ax = plt.subplots(3, 4, figsize=(12, 9))
bin_cmap = ListedColormap([(0.96, 0.96, 0.96), (0.15, 0.15, 0.15)])
for j, core in enumerate(BLD_TILES):
    raw = load(core)
    Hh, Ww = raw.shape[1:]
    bld = (raw[0] > 0.10).astype(float)
    mm = np.load(os.path.join(MASKS, f"{core}.npy")).astype(bool)[:Hh, :Ww]
    ax[0, j].imshow(bld, cmap=bin_cmap, vmin=0, vmax=1); ax[0, j].set_title(core, fontsize=11)
    sel = raw[3] > 0
    vmax = np.percentile(raw[3][sel], 99) if sel.any() else 1
    im = ax[1, j].imshow(raw[3], cmap="viridis", vmin=0, vmax=max(vmax, 1))  # raw label; 0 shown as 0
    plt.colorbar(im, ax=ax[1, j], fraction=0.046, pad=0.04)
    ov = np.full((Hh, Ww, 3), 0.96, np.float32)
    ov[bld > 0.5] = (0.55, 0.55, 0.55); ov[mm] = (0.86, 0.15, 0.15)
    ax[2, j].imshow(ov)
    ax[2, j].set_title(f"delmask: {100.0 * mm.sum() / (Hh * Ww):.1f}% px", fontsize=9)
    for r in range(3):
        ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
ax[0, 0].set_ylabel("GT footprint\n(cov>0.10)", fontsize=10)
ax[1, 0].set_ylabel("nDSM height (m)", fontsize=10)
ax[2, 0].set_ylabel("our delmask\n(red)", fontsize=10)
fig.suptitle("Observation B: deleted building footprints -- label empty, but nDSM "
             "shows buildings; our detector recovers them", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(OUT, "fig_building_holes.pdf"), bbox_inches="tight")
print("wrote fig_building_holes.pdf")
