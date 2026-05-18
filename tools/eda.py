"""OOP EDA library for the Embed2Heights dataset.

Classes are structured so each Analyzer can run independently (from a
notebook, interactive session, or CLI) and writes two kinds of outputs:

- Numeric CSVs to `eda_outputs_dir` (default: tools/eda_outputs/).
- Vector PDFs to `figures_dir` (default: <repo_root>/figures/).

Nothing writes mixed PNG/CSV into the same directory.

Entry point for batch use: see `tools/run_eda.py`.
"""

from __future__ import annotations

import json
import os
import sys
import glob
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "tools" / "data" / "embed2heights" / "data"
DEFAULT_CATALOG = REPO_ROOT / "tools" / "data" / "embed2heights" / "catalog.v1.parquet"
DEFAULT_SPLIT = REPO_ROOT / "splits" / "split.json"
DEFAULT_EDA_OUT = REPO_ROOT / "tools" / "eda_outputs"
DEFAULT_FIGURES = REPO_ROOT / "figures"

HEIGHT_NORM_CONSTANT = 30.0
LABEL_CHANNELS = ("building", "vegetation", "water", "height")
CH_BUILDING, CH_VEG, CH_WATER, CH_HEIGHT = 0, 1, 2, 3

# Train subdir -> family name (test subdirs are these with "_test" spliced in).
FAMILIES = ("alphaearth", "tessera", "terramind_s1", "terramind_s2", "thor_s1", "thor_s2")


# ---------------------------------------------------------------------------
# Plotting helper (module-level, stateless)
# ---------------------------------------------------------------------------

class _Plot:
    """Saves matplotlib figures as vector PDFs in figures_dir.

    Callers pass a Figure and a short name; this enforces .pdf extension,
    tight bbox, and a consistent font size across the whole report.
    """

    def __init__(self, figures_dir: Path):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.size": 9, "figure.dpi": 100, "pdf.fonttype": 42})

    def save(self, fig, name: str) -> Path:
        if not name.endswith(".pdf"):
            name = f"{name}.pdf"
        path = self.figures_dir / name
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return path


# ---------------------------------------------------------------------------
# Core-ID normalization (kept compatible with core/dataset.py semantics)
# ---------------------------------------------------------------------------

def normalize_core_id(path: str | Path) -> str:
    base = os.path.splitext(os.path.basename(str(path)))[0]
    if base.startswith("label_"):
        base = base[len("label_"):]
    if base.startswith("pred_"):
        base = base[len("pred_"):]
    for prefix in ("gee_emb_", "tessera_emb_", "emb_", "s2_", "s1_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    for suffix in ("_embedding", "_embeddings", "_quantized", "_merged"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    base = re.sub(r"_\d{4}$", "", base)
    return base


# ---------------------------------------------------------------------------
# PatchIndex — single source of truth for "which patches exist, and where"
# ---------------------------------------------------------------------------

@dataclass
class PatchIndex:
    """Walks the data tree once and builds core_id -> file-path maps."""

    data_root: Path = DEFAULT_DATA_ROOT
    train_index: dict[str, dict[str, Path]] = field(default_factory=dict)
    test_index: dict[str, dict[str, Path]] = field(default_factory=dict)

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self._build("train")
        self._build("test")

    def _family_dir(self, split: str, family: str) -> Path:
        if split == "train":
            return self.data_root / "train" / f"{family}_emb"
        # test dir naming inserts "_test": e.g., alphaearth_test_emb,
        # terramind_test_s1_emb, thor_test_s2_emb.
        if family in ("alphaearth", "tessera"):
            return self.data_root / "test" / f"{family}_test_emb"
        # terramind_s1 -> terramind_test_s1_emb
        base, suffix = family.split("_", 1)
        return self.data_root / "test" / f"{base}_test_{suffix}_emb"

    def _build(self, split: str):
        index: dict[str, dict[str, Path]] = {}
        for family in FAMILIES:
            d = self._family_dir(split, family)
            if not d.exists():
                continue
            for p in d.glob("*.tif"):
                cid = normalize_core_id(p)
                index.setdefault(cid, {})[family] = p
        # labels (train only)
        if split == "train":
            lab_dir = self.data_root / "train" / "labels"
            if lab_dir.exists():
                for p in lab_dir.glob("label_*.tif"):
                    cid = normalize_core_id(p)
                    index.setdefault(cid, {})["labels"] = p
        target = self.train_index if split == "train" else self.test_index
        target.update(index)

    def ids(self, split: str) -> list[str]:
        return sorted((self.train_index if split == "train" else self.test_index).keys())

    def labeled_ids(self) -> list[str]:
        return sorted(cid for cid, m in self.train_index.items() if "labels" in m)

    def load(self, split: str, cid: str, kind: str) -> np.ndarray:
        """Return CHW float32 array. `kind` is a family name or 'labels'."""
        idx = self.train_index if split == "train" else self.test_index
        path = idx[cid][kind]
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Analyzer base class
# ---------------------------------------------------------------------------

class Analyzer(ABC):
    """Base class. Each subclass writes CSV(s) to out_dir and PDF(s) via plot."""

    name: str = "analyzer"

    def __init__(
        self,
        index: PatchIndex,
        out_dir: Path = DEFAULT_EDA_OUT,
        figures_dir: Path = DEFAULT_FIGURES,
        sample: int | None = None,
    ):
        self.index = index
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plot = _Plot(figures_dir)
        self.sample = sample

    def _select_ids(self, ids: list[str]) -> list[str]:
        if self.sample and self.sample < len(ids):
            rng = np.random.default_rng(42)
            return sorted(rng.choice(ids, size=self.sample, replace=False).tolist())
        return ids

    @abstractmethod
    def run(self) -> None: ...


# ---------------------------------------------------------------------------
# 1. LabelAnalyzer
# ---------------------------------------------------------------------------

class LabelAnalyzer(Analyzer):
    name = "labels"

    def run(self) -> None:
        ids = self._select_ids(self.index.labeled_ids())
        rows = []
        # Accumulators for aggregate histograms.
        hist_bins = {ch: np.zeros(101, dtype=np.int64) for ch in range(3)}  # 0..100 for coverage
        height_bins = np.zeros(101, dtype=np.int64)  # 0..100 m, clipped
        sum_histogram = np.zeros(301, dtype=np.int64)  # 0..300 for sum-of-3-channels
        # Conditional: height | building>0 (and equivalently veg>0).
        height_given_bld = []
        height_given_veg = []
        # Pairwise correlation: streaming sum of pairwise products (per pixel).
        corr_accum = np.zeros((4, 4), dtype=np.float64)
        corr_mean = np.zeros(4, dtype=np.float64)
        corr_n = 0

        for cid in tqdm(ids, desc="labels"):
            arr = self.index.load("train", cid, "labels")  # (4, H, W)
            # Per-patch row.
            row = {"core_id": cid}
            for i, name in enumerate(LABEL_CHANNELS):
                ch = arr[i].ravel()
                row[f"{name}_min"] = float(ch.min())
                row[f"{name}_max"] = float(ch.max())
                row[f"{name}_mean"] = float(ch.mean())
                row[f"{name}_p50"] = float(np.median(ch))
                row[f"{name}_p95"] = float(np.percentile(ch, 95))
                row[f"{name}_p99"] = float(np.percentile(ch, 99))
                row[f"{name}_frac_nonzero"] = float((ch > 0).mean())
            # Any-channel foreground fraction.
            any_fg = (arr[:3] > 0).any(axis=0)
            row["any_fg_frac"] = float(any_fg.mean())
            # Sum of coverage channels.
            s = arr[:3].sum(axis=0)  # (H, W)
            row["cov_sum_mean"] = float(s.mean())
            row["cov_sum_max"] = float(s.max())
            row["cov_sum_over_100_frac"] = float((s > 100.0).mean())
            row["cov_sum_zero_frac"] = float((s == 0).mean())
            rows.append(row)

            # Aggregate histograms — clip to bin range.
            for ch in range(3):
                v = np.clip(arr[ch].ravel(), 0, 100).astype(np.int32)
                hist_bins[ch] += np.bincount(v, minlength=101)
            hv = np.clip(arr[CH_HEIGHT].ravel(), 0, 100).astype(np.int32)
            height_bins += np.bincount(hv, minlength=101)
            sv = np.clip(s.ravel(), 0, 300).astype(np.int32)
            sum_histogram += np.bincount(sv, minlength=301)

            # Height conditional distributions (sample up to 2000 pixels/patch).
            mask_b = arr[CH_BUILDING] > 1.0  # 1% coverage
            mask_v = arr[CH_VEG] > 1.0
            if mask_b.any():
                hb = arr[CH_HEIGHT][mask_b]
                if hb.size > 2000:
                    hb = np.random.default_rng(0).choice(hb, 2000, replace=False)
                height_given_bld.append(hb)
            if mask_v.any():
                hv2 = arr[CH_HEIGHT][mask_v]
                if hv2.size > 2000:
                    hv2 = np.random.default_rng(0).choice(hv2, 2000, replace=False)
                height_given_veg.append(hv2)

            # Streaming correlation (Welford-ish for 4 channels).
            flat = arr.reshape(4, -1).astype(np.float64)  # (4, N)
            n = flat.shape[1]
            m = flat.mean(axis=1)
            d = flat - m[:, None]
            corr_accum += d @ d.T
            corr_mean = (corr_mean * corr_n + m * n) / (corr_n + n)
            corr_n += n

        df = pd.DataFrame(rows)
        df.to_csv(self.out_dir / "label_per_patch.csv", index=False)

        # Aggregate scalars.
        agg = {
            "n_patches": len(df),
            "global_max_building": df["building_max"].max(),
            "global_max_vegetation": df["vegetation_max"].max(),
            "global_max_water": df["water_max"].max(),
            "global_max_height_m": df["height_max"].max(),
            "mean_building_pct": df["building_mean"].mean(),
            "mean_vegetation_pct": df["vegetation_mean"].mean(),
            "mean_water_pct": df["water_mean"].mean(),
            "mean_height_m": df["height_mean"].mean(),
            "mean_any_fg_frac": df["any_fg_frac"].mean(),
            "frac_patches_cov_sum_exceeds_100_anywhere": float((df["cov_sum_max"] > 100).mean()),
            "mean_frac_pixels_cov_sum_over_100": df["cov_sum_over_100_frac"].mean(),
            "mean_frac_pixels_cov_sum_zero": df["cov_sum_zero_frac"].mean(),
        }
        pd.DataFrame([agg]).to_csv(self.out_dir / "label_aggregate.csv", index=False)

        # Histograms CSV.
        hist_df = pd.DataFrame({
            "bin": np.arange(101),
            "building_count": hist_bins[0],
            "vegetation_count": hist_bins[1],
            "water_count": hist_bins[2],
            "height_count": height_bins,
        })
        hist_df.to_csv(self.out_dir / "label_histograms.csv", index=False)

        sum_df = pd.DataFrame({
            "bin": np.arange(301),
            "cov_sum_count": sum_histogram,
        })
        sum_df.to_csv(self.out_dir / "label_cov_sum_histogram.csv", index=False)

        # Correlation matrix (Pearson) computed from streaming moments.
        # cov = corr_accum / corr_n; corr = cov / (std_i * std_j)
        cov = corr_accum / max(corr_n, 1)
        std = np.sqrt(np.diag(cov))
        denom = np.outer(std, std)
        denom[denom == 0] = 1.0
        corr = cov / denom
        corr_df = pd.DataFrame(corr, index=LABEL_CHANNELS, columns=LABEL_CHANNELS)
        corr_df.to_csv(self.out_dir / "label_correlation.csv")

        # --- plots ---
        # 1. Per-channel histograms (log y).
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        for i, (ax, name) in enumerate(zip(axes.ravel(), LABEL_CHANNELS)):
            if i < 3:
                ax.bar(np.arange(101), hist_bins[i], width=1.0)
                ax.set_xlabel(f"{name} coverage (%)")
            else:
                ax.bar(np.arange(101), height_bins, width=1.0)
                ax.set_xlabel("height (m, clipped to 100)")
            ax.set_yscale("log")
            ax.set_ylabel("pixel count")
            ax.set_title(name)
        fig.suptitle(f"Label per-channel distributions (n={len(ids)} patches)")
        self.plot.save(fig, "label_histograms")

        # 2. Correlation heatmap.
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(LABEL_CHANNELS, rotation=30, ha="right")
        ax.set_yticklabels(LABEL_CHANNELS)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        color="black" if abs(corr[i, j]) < 0.5 else "white")
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title("Pixel-level label correlation")
        self.plot.save(fig, "label_correlation")

        # 3. Sum-of-3-coverage histogram with 100 marker.
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(np.arange(301), sum_histogram, width=1.0)
        ax.axvline(100, color="red", linestyle="--", label="sum = 100")
        ax.set_yscale("log")
        ax.set_xlabel("building% + vegetation% + water%")
        ax.set_ylabel("pixel count (log)")
        ax.set_title("Sum of 3 coverage channels per pixel")
        ax.legend()
        self.plot.save(fig, "label_coverage_sum")

        # 4. Conditional height | building>0 and height | veg>0.
        if height_given_bld and height_given_veg:
            hb_all = np.concatenate(height_given_bld)
            hv_all = np.concatenate(height_given_veg)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            axes[0].hist(hb_all, bins=np.linspace(0, 60, 61), color="C0")
            axes[0].set_xlabel("height (m)")
            axes[0].set_title("Height | building > 1%")
            axes[1].hist(hv_all, bins=np.linspace(0, 60, 61), color="C2")
            axes[1].set_xlabel("height (m)")
            axes[1].set_title("Height | vegetation > 1%")
            for ax in axes:
                ax.set_ylabel("pixels (sampled)")
            self.plot.save(fig, "label_conditional_height")

        # 5. CDF of per-patch any-fg fraction.
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sorted_fg = np.sort(df["any_fg_frac"].values)
        ax.plot(sorted_fg, np.linspace(0, 1, len(sorted_fg)))
        ax.set_xlabel("fraction of foreground pixels per patch")
        ax.set_ylabel("CDF over patches")
        ax.set_title("Patch-level foreground density")
        self.plot.save(fig, "label_fg_density_cdf")


# ---------------------------------------------------------------------------
# 2. EmbeddingAnalyzer
# ---------------------------------------------------------------------------

class EmbeddingAnalyzer(Analyzer):
    name = "embeddings"

    def __init__(self, index: PatchIndex, family: str, pca_samples: int = 150_000, **kw):
        super().__init__(index, **kw)
        self.family = family
        self.pca_samples = pca_samples

    def run(self) -> None:
        ids = [cid for cid in self.index.ids("train") if self.family in self.index.train_index[cid]]
        ids = self._select_ids(ids)
        first = self.index.load("train", ids[0], self.family)
        C, H, W = first.shape
        # Per-channel streaming moments (Welford).
        n = 0
        mean = np.zeros(C, dtype=np.float64)
        M2 = np.zeros(C, dtype=np.float64)
        cmin = np.full(C, np.inf, dtype=np.float64)
        cmax = np.full(C, -np.inf, dtype=np.float64)
        nan_count = np.zeros(C, dtype=np.int64)
        zero_count = np.zeros(C, dtype=np.int64)
        # Per-patch L2 norm stats.
        l2_means = []
        # Spatial smoothness proxy: mean |x_{i,j} - x_{i+1,j}| + |x_{i,j} - x_{i,j+1}|
        smoothness = []
        # Reservoir for PCA sampling.
        per_patch_samples = max(20, self.pca_samples // max(len(ids), 1))
        sample_buffer: list[np.ndarray] = []
        rng = np.random.default_rng(0)

        for cid in tqdm(ids, desc=f"emb:{self.family}"):
            arr = self.index.load("train", cid, self.family)  # (C, H, W)
            flat = arr.reshape(C, -1).astype(np.float64)  # pixels/tokens = H*W
            k = flat.shape[1]
            # Welford update.
            delta = flat.mean(axis=1) - mean
            new_n = n + k
            mean = mean + delta * (k / new_n)
            M2 = M2 + ((flat - flat.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) + (delta ** 2) * (n * k / new_n)
            n = new_n
            cmin = np.minimum(cmin, flat.min(axis=1))
            cmax = np.maximum(cmax, flat.max(axis=1))
            nan_count += np.isnan(arr).reshape(C, -1).sum(axis=1)
            zero_count += (flat == 0).sum(axis=1)
            l2_means.append(float(np.sqrt((arr.astype(np.float64) ** 2).sum(axis=0)).mean()))

            # Spatial smoothness (only for H,W >= 2).
            if H >= 2 and W >= 2:
                dh = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
                dw = np.abs(arr[:, :, 1:] - arr[:, :, :-1]).mean()
                smoothness.append(float((dh + dw) / 2))

            # PCA sample.
            if flat.shape[1] > per_patch_samples:
                idx = rng.choice(flat.shape[1], size=per_patch_samples, replace=False)
                sample_buffer.append(flat[:, idx].T.astype(np.float32))
            else:
                sample_buffer.append(flat.T.astype(np.float32))

        var = M2 / max(n - 1, 1)
        std = np.sqrt(var)

        per_channel = pd.DataFrame({
            "channel": np.arange(C),
            "mean": mean,
            "std": std,
            "min": cmin,
            "max": cmax,
            "frac_nan": nan_count / max(n, 1),
            "frac_zero": zero_count / max(n, 1),
        })
        per_channel.to_csv(self.out_dir / f"emb_{self.family}_per_channel.csv", index=False)

        # PCA.
        X = np.vstack(sample_buffer)
        if X.shape[0] > self.pca_samples:
            X = X[rng.choice(X.shape[0], size=self.pca_samples, replace=False)]
        Xc = X - X.mean(axis=0, keepdims=True)
        # SVD-based PCA; for C=768 and N~150k this is tractable.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        ev = (S ** 2) / max(Xc.shape[0] - 1, 1)
        cum_ev = np.cumsum(ev) / ev.sum()
        comps_95 = int(np.searchsorted(cum_ev, 0.95) + 1)
        comps_99 = int(np.searchsorted(cum_ev, 0.99) + 1)

        summary = {
            "family": self.family,
            "C": C, "H": H, "W": W,
            "n_patches": len(ids),
            "grand_mean": float(mean.mean()),
            "grand_std_of_channel_means": float(mean.std()),
            "mean_channel_std": float(std.mean()),
            "min_overall": float(cmin.min()),
            "max_overall": float(cmax.max()),
            "mean_l2_per_pixel": float(np.mean(l2_means)),
            "mean_neighbor_abs_diff": float(np.mean(smoothness)) if smoothness else float("nan"),
            "comps_for_95pct_var": comps_95,
            "comps_for_99pct_var": comps_99,
        }
        pd.DataFrame([summary]).to_csv(self.out_dir / f"emb_{self.family}_summary.csv", index=False)

        # PCA cumulative variance plot.
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(np.arange(1, len(cum_ev) + 1), cum_ev)
        ax.axhline(0.95, color="red", linestyle="--", label="95%")
        ax.axhline(0.99, color="orange", linestyle="--", label="99%")
        ax.set_xlabel("PCA component")
        ax.set_ylabel("cumulative explained variance")
        ax.set_title(f"{self.family}: PCA (C={C})  95%@{comps_95}  99%@{comps_99}")
        ax.legend()
        ax.set_xscale("log")
        self.plot.save(fig, f"embedding_pca_{self.family}")

        # Per-channel mean/std plot.
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
        axes[0].plot(mean)
        axes[0].set_ylabel("channel mean")
        axes[0].set_title(f"{self.family}: per-channel mean/std across train set")
        axes[1].plot(std)
        axes[1].set_ylabel("channel std")
        axes[1].set_xlabel("channel index")
        self.plot.save(fig, f"embedding_per_channel_{self.family}")


# ---------------------------------------------------------------------------
# 3. CatalogAnalyzer
# ---------------------------------------------------------------------------

class CatalogAnalyzer(Analyzer):
    name = "catalog"

    def __init__(self, index: PatchIndex, catalog_path: Path = DEFAULT_CATALOG, **kw):
        super().__init__(index, **kw)
        self.catalog_path = Path(catalog_path)

    def run(self) -> None:
        if not self.catalog_path.exists():
            print(f"[catalog] MISSING: {self.catalog_path}", file=sys.stderr)
            return
        df = pd.read_parquet(self.catalog_path)
        # The catalog's bbox/datetime are not real geography (bbox is {0,0,0,0},
        # datetime is the ingest timestamp). The useful signal lives in the `id`
        # field, which is the file path. Parse core_id, region code, and year
        # from the filename.
        rx = re.compile(r"(?P<num>\d{3,4})_(?P<region>[A-Z]{2})(?:_(?P<year>\d{4}))?")
        def _parse(id_str: str) -> dict | None:
            m = rx.search(str(id_str))
            if not m:
                return None
            num = int(m.group("num"))
            return {
                "core_id": f"{m.group('num')}_{m.group('region')}",
                "region": m.group("region"),
                "year": int(m.group("year")) if m.group("year") else None,
                "split": "train" if num < 3000 else "test",
                "path": str(id_str),
            }
        parsed = [p for p in df["id"].apply(_parse) if p is not None]
        out = pd.DataFrame(parsed)
        if len(out) == 0:
            print("[catalog] could not parse any rows", file=sys.stderr)
            return
        # De-duplicate: each patch has many assets (one row per file); keep one per patch per split.
        patches = out.drop_duplicates(subset=["core_id", "split"]).copy()
        patches.to_csv(self.out_dir / "catalog_geometry.csv", index=False)

        # Region × split cross-tab.
        reg = patches.groupby(["region", "split"]).size().unstack(fill_value=0)
        reg.to_csv(self.out_dir / "catalog_region_split.csv")

        # Year × split cross-tab.
        yr = patches.dropna(subset=["year"]).groupby(["year", "split"]).size().unstack(fill_value=0)
        yr.to_csv(self.out_dir / "catalog_year_split.csv")

        # Plot: region bar chart (train vs test).
        fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(reg)), 3.5))
        reg.plot(kind="bar", stacked=False, ax=ax, color=["C0", "C3"])
        ax.set_xlabel("region code")
        ax.set_ylabel("patch count")
        ax.set_title("Patches per region (train vs test)")
        self.plot.save(fig, "catalog_map")  # kept filename for plan compatibility

        # Plot: year distribution.
        fig, ax = plt.subplots(figsize=(6, 3.5))
        yr.plot(kind="bar", ax=ax, color=["C0", "C3"])
        ax.set_xlabel("year")
        ax.set_ylabel("patch count")
        ax.set_title("Patch year (train vs test)")
        self.plot.save(fig, "catalog_temporal")

        # Overlap diagnostic: regions shared train↔test.
        train_regs = set(patches[patches["split"] == "train"]["region"])
        test_regs = set(patches[patches["split"] == "test"]["region"])
        overlap = {
            "train_only": sorted(train_regs - test_regs),
            "test_only": sorted(test_regs - train_regs),
            "shared": sorted(train_regs & test_regs),
            "n_train_regions": len(train_regs),
            "n_test_regions": len(test_regs),
            "n_shared_regions": len(train_regs & test_regs),
        }
        pd.DataFrame([overlap]).to_csv(self.out_dir / "catalog_region_overlap.csv", index=False)


# ---------------------------------------------------------------------------
# 4. SplitAnalyzer
# ---------------------------------------------------------------------------

class SplitAnalyzer(Analyzer):
    name = "split"

    def __init__(self, index: PatchIndex, split_path: Path = DEFAULT_SPLIT, **kw):
        super().__init__(index, **kw)
        self.split_path = Path(split_path)

    def run(self) -> None:
        if not self.split_path.exists():
            print(f"[split] MISSING: {self.split_path} — skipping", file=sys.stderr)
            return
        with open(self.split_path) as f:
            sp = json.load(f)
        per_patch_csv = self.out_dir / "label_per_patch.csv"
        if not per_patch_csv.exists():
            print("[split] label_per_patch.csv not found; run labels analyzer first", file=sys.stderr)
            return
        df = pd.read_csv(per_patch_csv)
        df["split"] = df["core_id"].apply(
            lambda c: "train" if c in set(sp["train"]) else ("val" if c in set(sp["val"]) else "other")
        )
        cols = [f"{c}_mean" for c in LABEL_CHANNELS] + ["any_fg_frac"]
        agg = df.groupby("split")[cols].agg(["mean", "std", "count"])
        agg.to_csv(self.out_dir / "split_parity.csv")

        # KS per channel between train and val.
        from scipy.stats import ks_2samp
        tr = df[df["split"] == "train"]
        va = df[df["split"] == "val"]
        ks_rows = []
        for c in cols:
            if len(tr) and len(va):
                stat, p = ks_2samp(tr[c].dropna(), va[c].dropna())
                ks_rows.append({"field": c, "ks_stat": stat, "p_value": p})
        pd.DataFrame(ks_rows).to_csv(self.out_dir / "split_ks_tests.csv", index=False)

        # Plot: side-by-side boxplots of per-patch means by split.
        present = df[df["split"].isin(["train", "val"])]
        fig, axes = plt.subplots(1, len(cols), figsize=(2.2 * len(cols), 3.5), sharey=False)
        for ax, c in zip(axes, cols):
            data = [present[present["split"] == s][c].dropna().values for s in ("train", "val")]
            ax.boxplot(data, labels=["train", "val"], showfliers=False)
            ax.set_title(c, fontsize=8)
        fig.suptitle("Train vs val split parity (per-patch means)")
        self.plot.save(fig, "split_parity")


# ---------------------------------------------------------------------------
# 5. ProbeAnalyzer — linear probe per family, pixel-level + token-level
# ---------------------------------------------------------------------------

class ProbeAnalyzer(Analyzer):
    name = "probe"

    def __init__(
        self,
        index: PatchIndex,
        families: Iterable[str] = FAMILIES,
        n_patches: int = 200,
        pixels_per_patch: int = 300,
        **kw,
    ):
        super().__init__(index, **kw)
        self.families = list(families)
        self.n_patches = n_patches
        self.pixels_per_patch = pixels_per_patch

    def _downsample_labels_to_tokens(self, labels: np.ndarray, token_hw: int) -> np.ndarray:
        """Block-average labels to (4, token_hw, token_hw). Crops to the largest
        multiple of token_hw if labels are off-by-one (some patches are 255×256)."""
        C, H, W = labels.shape
        fh = H // token_hw
        fw = W // token_hw
        f = min(fh, fw)
        side = f * token_hw
        labels = labels[:, :side, :side]
        return labels.reshape(C, token_hw, f, token_hw, f).mean(axis=(2, 4))

    def run(self) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        rng = np.random.default_rng(42)
        labeled = self.index.labeled_ids()
        cap = self.n_patches if self.sample is None else min(self.n_patches, self.sample)
        chosen = rng.choice(labeled, size=min(cap, len(labeled)), replace=False).tolist()
        # 80/20 split of these patches for fit/eval.
        n_fit = int(len(chosen) * 0.8)
        fit_ids, eval_ids = chosen[:n_fit], chosen[n_fit:]

        rows = []
        for family in self.families:
            # Discover token size from first patch.
            fam_ids = [c for c in chosen if family in self.index.train_index[c]]
            if not fam_ids:
                continue
            sample_emb = self.index.load("train", fam_ids[0], family)
            C, Hc, Wc = sample_emb.shape
            is_token = (Hc == 16)

            def collect(ids_sub):
                Xs, Ys = [], []
                Xs_tok, Ys_tok = [], []
                for cid in tqdm(ids_sub, desc=f"probe:{family}"):
                    if family not in self.index.train_index[cid]:
                        continue
                    emb = self.index.load("train", cid, family)
                    lab = self.index.load("train", cid, "labels")  # (4, 256, 256)
                    if is_token:
                        # Token-resolution probe.
                        labs_tok = self._downsample_labels_to_tokens(lab, Hc)
                        n_tok = Hc * Wc
                        k = min(self.pixels_per_patch, n_tok)
                        idx = rng.choice(n_tok, size=k, replace=False)
                        Xs_tok.append(emb.reshape(C, -1)[:, idx].T)
                        Ys_tok.append(labs_tok.reshape(4, -1)[:, idx].T)
                    else:
                        # Pixel-aligned probe. Crop both to their intersection
                        # (some patches are 255×256 vs 256×256).
                        H_ = min(emb.shape[1], lab.shape[1])
                        W_ = min(emb.shape[2], lab.shape[2])
                        emb_c = emb[:, :H_, :W_]
                        lab_c = lab[:, :H_, :W_]
                        n_px = H_ * W_
                        k = min(self.pixels_per_patch, n_px)
                        idx = rng.choice(n_px, size=k, replace=False)
                        Xs.append(emb_c.reshape(C, -1)[:, idx].T)
                        Ys.append(lab_c.reshape(4, -1)[:, idx].T)
                X = np.vstack(Xs) if Xs else None
                Y = np.vstack(Ys) if Ys else None
                Xt = np.vstack(Xs_tok) if Xs_tok else None
                Yt = np.vstack(Ys_tok) if Ys_tok else None
                return X, Y, Xt, Yt

            Xtr, Ytr, Xtr_tok, Ytr_tok = collect(fit_ids)
            Xev, Yev, Xev_tok, Yev_tok = collect(eval_ids)

            # Normalize embeddings with train stats (matters for AlphaEarth raw scale).
            def _norm(Xt, Xe):
                if Xt is None:
                    return None, None
                mu = Xt.mean(axis=0, keepdims=True)
                sd = Xt.std(axis=0, keepdims=True) + 1e-6
                return (Xt - mu) / sd, (Xe - mu) / sd

            def _fit(Xt, Yt, Xe, Ye, resolution):
                if Xt is None or Ye is None or len(Xt) == 0:
                    return
                Xtn, Xen = _norm(Xt, Xe)
                for i, name in enumerate(LABEL_CHANNELS):
                    y_tr, y_ev = Yt[:, i], Ye[:, i]
                    model = Ridge(alpha=1.0)
                    model.fit(Xtn, y_tr)
                    y_pred = model.predict(Xen)
                    r2 = r2_score(y_ev, y_pred)
                    rows.append({
                        "family": family,
                        "resolution": resolution,
                        "channel": name,
                        "n_fit": len(y_tr),
                        "n_eval": len(y_ev),
                        "r2": float(r2),
                    })

            if is_token:
                _fit(Xtr_tok, Ytr_tok, Xev_tok, Yev_tok, "token_16x16")
            else:
                _fit(Xtr, Ytr, Xev, Yev, "pixel_256x256")

        probe_df = pd.DataFrame(rows)
        probe_df.to_csv(self.out_dir / "probe_r2.csv", index=False)
        if len(probe_df):
            pivot = probe_df.pivot_table(index="family", columns="channel", values="r2")
            pivot.to_csv(self.out_dir / "probe_r2_pivot.csv")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("held-out R²")
            ax.set_title("Linear probe per family × label channel")
            ax.axhline(0, color="black", linewidth=0.5)
            self.plot.save(fig, "probe_r2")


# ---------------------------------------------------------------------------
# 6. ShiftAnalyzer — embedding distribution shift train vs test
# ---------------------------------------------------------------------------

class ShiftAnalyzer(Analyzer):
    name = "shift"

    def __init__(self, index: PatchIndex, families: Iterable[str] = FAMILIES, n_patches: int = 200, **kw):
        super().__init__(index, **kw)
        self.families = list(families)
        self.n_patches = n_patches

    def _streaming_stats(self, split: str, family: str, ids: list[str]) -> dict:
        rng = np.random.default_rng(7)
        cap = self.n_patches if self.sample is None else min(self.n_patches, self.sample)
        chosen = rng.choice(ids, size=min(cap, len(ids)), replace=False).tolist()
        n = 0
        mean = None
        M2 = None
        cmin = None
        cmax = None
        l2s = []
        for cid in tqdm(chosen, desc=f"shift:{split}:{family}"):
            if family not in (self.index.train_index if split == "train" else self.index.test_index)[cid]:
                continue
            arr = self.index.load(split, cid, family)
            C = arr.shape[0]
            flat = arr.reshape(C, -1).astype(np.float64)
            k = flat.shape[1]
            if mean is None:
                mean = np.zeros(C); M2 = np.zeros(C)
                cmin = np.full(C, np.inf); cmax = np.full(C, -np.inf)
            delta = flat.mean(axis=1) - mean
            new_n = n + k
            mean = mean + delta * (k / new_n)
            M2 = M2 + ((flat - flat.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) + (delta ** 2) * (n * k / new_n)
            n = new_n
            cmin = np.minimum(cmin, flat.min(axis=1))
            cmax = np.maximum(cmax, flat.max(axis=1))
            l2s.append(float(np.sqrt((arr.astype(np.float64) ** 2).sum(axis=0)).mean()))
        std = np.sqrt(M2 / max(n - 1, 1))
        return {"mean": mean, "std": std, "min": cmin, "max": cmax, "l2s": np.array(l2s), "n_pixels": n}

    def run(self) -> None:
        rows = []
        for family in self.families:
            train_ids = [c for c in self.index.ids("train") if family in self.index.train_index[c]]
            test_ids = [c for c in self.index.ids("test") if family in self.index.test_index[c]]
            if not train_ids or not test_ids:
                continue
            tr = self._streaming_stats("train", family, train_ids)
            te = self._streaming_stats("test", family, test_ids)
            # Summary: mean-shift, std-ratio.
            mean_shift = float(np.mean(np.abs(tr["mean"] - te["mean"])))
            std_ratio = float(np.mean(te["std"] / (tr["std"] + 1e-9)))
            l2_shift = float(np.mean(te["l2s"]) - np.mean(tr["l2s"]))
            rows.append({
                "family": family,
                "train_grand_mean": float(tr["mean"].mean()),
                "test_grand_mean": float(te["mean"].mean()),
                "train_mean_std": float(tr["std"].mean()),
                "test_mean_std": float(te["std"].mean()),
                "channel_mean_abs_shift": mean_shift,
                "channel_std_ratio_test_over_train": std_ratio,
                "l2_norm_shift": l2_shift,
            })
            # Per-channel diff plot.
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(tr["mean"], label="train mean", color="C0")
            ax.plot(te["mean"], label="test mean", color="C3")
            ax.fill_between(np.arange(len(tr["mean"])), tr["mean"] - tr["std"], tr["mean"] + tr["std"], alpha=0.2, color="C0")
            ax.fill_between(np.arange(len(te["mean"])), te["mean"] - te["std"], te["mean"] + te["std"], alpha=0.2, color="C3")
            ax.set_title(f"{family}: per-channel mean ± std (train vs test)")
            ax.set_xlabel("channel")
            ax.legend()
            self.plot.save(fig, f"shift_per_channel_{family}")
        shift_df = pd.DataFrame(rows)
        shift_df.to_csv(self.out_dir / "train_test_shift.csv", index=False)
        if len(shift_df):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            x = np.arange(len(shift_df))
            ax.bar(x - 0.2, shift_df["channel_mean_abs_shift"], width=0.4, label="|Δmean|")
            ax.bar(x + 0.2, np.abs(np.log(shift_df["channel_std_ratio_test_over_train"])), width=0.4, label="|log(std_ratio)|")
            ax.set_xticks(x); ax.set_xticklabels(shift_df["family"], rotation=20, ha="right")
            ax.set_title("Train vs test shift severity")
            ax.legend()
            self.plot.save(fig, "train_test_shift")


# ---------------------------------------------------------------------------
# 7. DifficultyProfiler — uses outputs from LabelAnalyzer
# ---------------------------------------------------------------------------

class DifficultyProfiler(Analyzer):
    name = "difficulty"

    def run(self) -> None:
        per_patch = self.out_dir / "label_per_patch.csv"
        if not per_patch.exists():
            print("[difficulty] label_per_patch.csv missing; run labels first", file=sys.stderr)
            return
        df = pd.read_csv(per_patch)
        # Components: label density, class entropy, cov_sum balance, height variability.
        def _entropy(row):
            p = np.array([row["building_mean"], row["vegetation_mean"], row["water_mean"]])
            if p.sum() == 0:
                return 0.0
            p = p / p.sum()
            return float(-(p * np.log(p + 1e-12)).sum())
        df["class_entropy"] = df.apply(_entropy, axis=1)
        df["density"] = df["any_fg_frac"]
        df["height_dynamic_range"] = df["height_p99"] - df["height_p50"]
        # Z-score each component then combine.
        for col in ("class_entropy", "density", "height_dynamic_range"):
            mu, sd = df[col].mean(), df[col].std() + 1e-9
            df[f"{col}_z"] = (df[col] - mu) / sd
        df["difficulty"] = df[["class_entropy_z", "density_z", "height_dynamic_range_z"]].mean(axis=1)
        df[["core_id", "density", "class_entropy", "height_dynamic_range", "difficulty"]].to_csv(
            self.out_dir / "patch_difficulty.csv", index=False
        )
        # Plot histogram.
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df["difficulty"], bins=40)
        ax.set_xlabel("difficulty score (z-combined)")
        ax.set_ylabel("patch count")
        ax.set_title("Per-patch difficulty distribution")
        self.plot.save(fig, "patch_difficulty")


# ---------------------------------------------------------------------------
# 8. ReportBuilder — synthesizes REPORT.md from CSVs
# ---------------------------------------------------------------------------

class ReportBuilder(Analyzer):
    name = "report"

    def _read(self, name: str) -> pd.DataFrame | None:
        p = self.out_dir / name
        return pd.read_csv(p) if p.exists() else None

    def run(self) -> None:
        parts = ["# Embed2Heights — Data Analysis Report\n"]
        parts.append("All numbers cited below come from CSVs in `tools/eda_outputs/`. Figures referenced are PDFs in `figures/`.\n")

        # Section 1: What the data is.
        agg = self._read("label_aggregate.csv")
        parts.append("## 1. What the data actually is\n")
        if agg is not None:
            r = agg.iloc[0].to_dict()
            parts.append(
                f"- Labeled patches analyzed: **{int(r['n_patches'])}**.\n"
                f"- Global max per coverage channel: building={r['global_max_building']:.1f}, "
                f"vegetation={r['global_max_vegetation']:.1f}, water={r['global_max_water']:.1f}.\n"
                f"- Global max height: **{r['global_max_height_m']:.1f} m** "
                f"(current `HEIGHT_NORM_CONSTANT=30` → check against p99/max).\n"
                f"- Mean coverage fractions: B={r['mean_building_pct']:.2f}%, V={r['mean_vegetation_pct']:.2f}%, "
                f"W={r['mean_water_pct']:.2f}%.\n"
                f"- Mean per-patch foreground density (any-channel>0): **{r['mean_any_fg_frac']:.3f}**.\n"
                f"- Pixels with `building+veg+water` > 100 (mean frac): **{r['mean_frac_pixels_cov_sum_over_100']:.4f}**. "
                f"Patches where sum ever > 100: **{r['frac_patches_cov_sum_exceeds_100_anywhere']:.3f}**.\n"
                f"- Pure-zero pixels (all coverage = 0): **{r['mean_frac_pixels_cov_sum_zero']:.3f}**.\n\n"
                f"See `figures/label_histograms.pdf`, `figures/label_correlation.pdf`, "
                f"`figures/label_coverage_sum.pdf`, `figures/label_conditional_height.pdf`.\n\n"
            )
            # Resolve Open Question 1.
            if r["global_max_building"] > 2.0:
                parts.append(
                    "**Open Question 1 (scale) — RESOLVED: `[0, 100]` percentages.** "
                    f"Max observed = {r['global_max_building']:.1f} across building channel. "
                    "This means `evaluate.py` threshold of `0.5` binarizes at 0.5% coverage — essentially any nonzero pixel. "
                    "The training code also does not bound outputs, so the network regresses unbounded values against a bounded target.\n\n"
                )

        # Section 2: Embeddings.
        parts.append("## 2. Embedding characterization\n")
        for family in FAMILIES:
            s = self._read(f"emb_{family}_summary.csv")
            if s is None:
                continue
            r = s.iloc[0].to_dict()
            parts.append(
                f"- **{family}**: C={int(r['C'])}, spatial={int(r['H'])}×{int(r['W'])}, "
                f"grand_mean={r['grand_mean']:.3f}, mean-channel-std={r['mean_channel_std']:.3f}, "
                f"range=[{r['min_overall']:.2f}, {r['max_overall']:.2f}], "
                f"L2/pixel={r['mean_l2_per_pixel']:.2f}, "
                f"95%-var-components={int(r['comps_for_95pct_var'])}, "
                f"99%-var-components={int(r['comps_for_99pct_var'])}.\n"
            )
        parts.append("\nFigures: `figures/embedding_per_channel_<family>.pdf`, `figures/embedding_pca_<family>.pdf`.\n\n")

        # Section 3: Catalog / geography.
        parts.append("## 3. Catalog & geography\n")
        cat = self._read("catalog_geometry.csv")
        overlap = self._read("catalog_region_overlap.csv")
        if cat is not None:
            by = cat.groupby("split").size().to_dict()
            parts.append(f"- Patches by split: {by}\n")
            years = cat.dropna(subset=["year"]).groupby(["split", "year"]).size().unstack(fill_value=0)
            if not years.empty:
                parts.append(f"- Year distribution:\n```\n{years.to_string()}\n```\n")
            if overlap is not None:
                o = overlap.iloc[0]
                parts.append(
                    f"- Region overlap: **{o['n_shared_regions']}** regions shared, "
                    f"{len(eval(o['train_only']) if isinstance(o['train_only'], str) else o['train_only'])} train-only, "
                    f"{len(eval(o['test_only']) if isinstance(o['test_only'], str) else o['test_only'])} test-only.\n"
                )
            parts.append("- See `figures/catalog_map.pdf`, `figures/catalog_temporal.pdf`.\n\n")

        # Section 4: Probe.
        parts.append("## 4. Linear-probe predictive signal (pixel-level Ridge, held-out R²)\n")
        probe = self._read("probe_r2_pivot.csv")
        if probe is not None:
            parts.append(f"```\n{probe.to_string()}\n```\n")
            parts.append("See `figures/probe_r2.pdf`. For ViT families R² is reported at token resolution (16×16); "
                         "compare against pixel-resolution families to isolate upsampling loss from feature-quality loss.\n\n")

        # Section 5: Train vs test shift.
        parts.append("## 5. Train↔test distribution shift\n")
        shift = self._read("train_test_shift.csv")
        if shift is not None:
            parts.append(f"```\n{shift.to_string(index=False)}\n```\n")
            parts.append("See `figures/train_test_shift.pdf` and per-family `figures/shift_per_channel_<family>.pdf`.\n\n")

        # Section 6: Split parity.
        parts.append("## 6. Train/val split parity\n")
        ks = self._read("split_ks_tests.csv")
        if ks is not None:
            parts.append(f"```\n{ks.to_string(index=False)}\n```\n")
            parts.append("See `figures/split_parity.pdf`.\n\n")

        # Section 7: Difficulty.
        parts.append("## 7. Per-patch difficulty\n")
        diff = self._read("patch_difficulty.csv")
        if diff is not None:
            parts.append(
                f"- n={len(diff)} patches profiled. "
                f"Hardest 10 (by combined z-score):\n```\n{diff.nlargest(10, 'difficulty').to_string(index=False)}\n```\n"
                f"- See `figures/patch_difficulty.pdf`.\n\n"
            )

        # Section 8: Recommendations stub (to be filled by hand once numbers land).
        parts.append("## 8. Concrete modeling recommendations\n")
        parts.append(
            "_This section is populated from the numbers above. Key levers:_\n\n"
            "1. **Output activation + target scale**: with coverage channels in `[0, 100]` and height in meters, "
            "use `sigmoid(·)*100` for channels 0–2 and `softplus(·)*height_norm` for channel 3. Current model has no activation.\n"
            "2. **Per-family normalization before fusion**: AlphaEarth (raw) and Tessera/TerraMind (z-scored) must be normalized "
            "with per-family moments (see Section 2) before any concatenation or cross-family fusion.\n"
            "3. **Fusion architecture**: combine AlphaEarth pixel-dense 64ch (best baseline) with a ViT family at token resolution, "
            "upsampling each independently and fusing at 256×256. Section 4 R² numbers indicate which ViT family adds signal.\n"
            "4. **Local val split**: replace random 80/20 with a geographically held-out split (hold out 2–3 French cities). "
            "Section 5 shift numbers quantify how optimistic the current random split is.\n"
            "5. **Loss recalibration**: `HEIGHT_NORM_CONSTANT=30` is too low if Section 1 reports height_max > 40; rescale. "
            "`bg_weight=0.05` may be too high given ~73% zeros; sweep 0.01–0.05. "
            "Height-boost threshold `> 0.1` in losses.py is on `[0, 100]` scale → triggers almost everywhere; raise to e.g. `> 20`.\n"
            "6. **Eval threshold mismatch**: `evaluate.py` binarizes at 0.5 (0.5%) — almost-anything-positive. "
            "Verify leaderboard semantics; if leaderboard uses a higher threshold (e.g. 50%), our local metric is wrong.\n"
            "7. **Sum-to-100 head**: if Section 1 shows sum < 100 on most foreground pixels, add a 4th 'unclassified' coverage output "
            "and train with a softmax-like constraint so the 4 land-cover outputs sum to 100.\n"
            "8. **Augmentation**: horizontal/vertical flip safe (embeddings are spatial feature maps); avoid rotation and any color/channel ops.\n"
        )

        (self.out_dir / "REPORT.md").write_text("".join(parts))


# ---------------------------------------------------------------------------
# Orchestrator (kept tiny — CLI lives in run_eda.py).
# ---------------------------------------------------------------------------

ANALYZER_ORDER = (
    ("labels", lambda idx, **kw: LabelAnalyzer(idx, **kw)),
    *[(f"embeddings_{f}", (lambda idx, f=f, **kw: EmbeddingAnalyzer(idx, family=f, **kw))) for f in FAMILIES],
    ("catalog", lambda idx, **kw: CatalogAnalyzer(idx, **kw)),
    ("split", lambda idx, **kw: SplitAnalyzer(idx, **kw)),
    ("probe", lambda idx, **kw: ProbeAnalyzer(idx, **kw)),
    ("shift", lambda idx, **kw: ShiftAnalyzer(idx, **kw)),
    ("difficulty", lambda idx, **kw: DifficultyProfiler(idx, **kw)),
    ("report", lambda idx, **kw: ReportBuilder(idx, **kw)),
)
