"""
Frequency-domain diagnostic for the six ESA Embed2Heights embedding sets.

The script asks a practical question: can a simple frequency filter preserve
label-relevant signal while suppressing likely noise?

For each embedding source it computes:
  1) FFT radial energy split into low / mid / high frequency bands.
  2) Correlation between high-frequency embedding magnitude and label edges.
  3) Linear probes on full, low-pass, low+mid-pass, and high-pass components:
       - RidgeClassifier AUC for building / vegetation / water presence.
       - Ridge regression R2/RMSE for height on building-or-vegetation cells.

Pixel embeddings are evaluated at 256x256. Token embeddings are evaluated at
their native 16x16 grid after block-aggregating labels to token cells.

Usage:
    conda run -n emb2heights python tools/diagnostic_frequency_probe.py
    conda run -n emb2heights python tools/diagnostic_frequency_probe.py --n-ids-train 200 --n-ids-eval 80
    conda run -n emb2heights python tools/diagnostic_frequency_probe.py --sources alphaearth tessera
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio

try:
    from scipy.linalg import LinAlgWarning
    warnings.filterwarnings("ignore", category=LinAlgWarning)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from core.dataset import clean_raster_array, normalize_core_id  # noqa: E402


DATA_DIR = Path("/u/dingqi2/workspace/esa/data/train")
LABEL_DIR = DATA_DIR / "labels"
SOURCES = {
    "alphaearth": DATA_DIR / "alphaearth_emb",
    "tessera": DATA_DIR / "tessera_emb",
    "terramind_s1": DATA_DIR / "terramind_s1_emb",
    "terramind_s2": DATA_DIR / "terramind_s2_emb",
    "thor_s1": DATA_DIR / "thor_s1_emb",
    "thor_s2": DATA_DIR / "thor_s2_emb",
}
TASKS = ("building", "vegetation", "water")
CH_BUILDING, CH_VEGETATION, CH_WATER, CH_HEIGHT = 0, 1, 2, 3
FREQ_MODES = ("full", "low", "low_mid", "high")
MIN_FEATURE_STD = 1e-5


def load_split(path: Path) -> tuple[list[str], list[str]]:
    with path.open() as f:
        split = json.load(f)
    return list(split["train"]), list(split["val"])


def index_tifs(directory: Path) -> dict[str, str]:
    paths = sorted(glob.glob(str(directory / "**" / "*.tif"), recursive=True))
    return {normalize_core_id(path): path for path in paths}


def load_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return clean_raster_array(src.read())


def radial_masks(h: int, w: int, low_cut: float, mid_cut: float) -> dict[str, np.ndarray]:
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy) / np.sqrt(0.5 * 0.5 + 0.5 * 0.5)
    low = radius <= low_cut
    mid = (radius > low_cut) & (radius <= mid_cut)
    high = radius > mid_cut
    return {
        "low": low,
        "mid": mid,
        "high": high,
        "low_mid": low | mid,
    }


def fft_components(emb: np.ndarray, masks: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    freq = np.fft.fft2(emb, axes=(-2, -1))
    power = np.abs(freq) ** 2
    total = float(power.sum()) + 1e-12
    energy = {
        "low": float(power[:, masks["low"]].sum() / total),
        "mid": float(power[:, masks["mid"]].sum() / total),
        "high": float(power[:, masks["high"]].sum() / total),
    }
    comps = {
        "full": emb,
        "low": np.fft.ifft2(freq * masks["low"][None, :, :], axes=(-2, -1)).real.astype(np.float32),
        "low_mid": np.fft.ifft2(freq * masks["low_mid"][None, :, :], axes=(-2, -1)).real.astype(np.float32),
        "high": np.fft.ifft2(freq * masks["high"][None, :, :], axes=(-2, -1)).real.astype(np.float32),
    }
    return comps, energy


def block_mean(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Mean-pool a 2D array to out_h x out_w, allowing uneven edge bins."""
    h, w = arr.shape
    y_edges = np.linspace(0, h, out_h + 1).round().astype(int)
    x_edges = np.linspace(0, w, out_w + 1).round().astype(int)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for iy in range(out_h):
        y0, y1 = y_edges[iy], max(y_edges[iy + 1], y_edges[iy] + 1)
        for ix in range(out_w):
            x0, x1 = x_edges[ix], max(x_edges[ix + 1], x_edges[ix] + 1)
            out[iy, ix] = float(np.mean(arr[y0:y1, x0:x1]))
    return out


def aggregate_label(label: np.ndarray, out_h: int, out_w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aggregated label, global-valid fraction, and height-valid fraction."""
    h, w = label.shape[1:]
    if (h, w) == (out_h, out_w):
        global_valid = ~np.all(label == 0, axis=0)
        any_class = (label[0] > 0) | (label[1] > 0) | (label[2] > 0)
        height_hole = (label[3] == 0) & any_class
        height_valid = global_valid & ~height_hole
        return label, global_valid.astype(np.float32), height_valid.astype(np.float32)

    global_valid = ~np.all(label == 0, axis=0)
    any_class = (label[0] > 0) | (label[1] > 0) | (label[2] > 0)
    height_hole = (label[3] == 0) & any_class
    height_valid = global_valid & ~height_hole

    out = np.zeros((4, out_h, out_w), dtype=np.float32)
    valid_frac = block_mean(global_valid.astype(np.float32), out_h, out_w)
    height_valid_frac = block_mean(height_valid.astype(np.float32), out_h, out_w)
    for ch in (CH_BUILDING, CH_VEGETATION, CH_WATER):
        out[ch] = block_mean(label[ch], out_h, out_w)

    weighted_h = block_mean(label[CH_HEIGHT] * height_valid.astype(np.float32), out_h, out_w)
    denom = np.maximum(height_valid_frac, 1e-6)
    out[CH_HEIGHT] = weighted_h / denom
    out[CH_HEIGHT][height_valid_frac <= 0] = 0.0
    return out, valid_frac, height_valid_frac


def label_edge_strength(label: np.ndarray, valid: np.ndarray) -> np.ndarray:
    parts = []
    for ch in (CH_BUILDING, CH_VEGETATION, CH_WATER):
        gy, gx = np.gradient(label[ch])
        parts.append(np.sqrt(gx * gx + gy * gy))
    height = label[CH_HEIGHT].astype(np.float32)
    if np.nanstd(height[valid > 0]) > 1e-6:
        h_norm = (height - np.nanmean(height[valid > 0])) / (np.nanstd(height[valid > 0]) + 1e-6)
        gy, gx = np.gradient(h_norm)
        parts.append(np.sqrt(gx * gx + gy * gy))
    edge = np.max(np.stack(parts, axis=0), axis=0)
    edge[valid <= 0] = 0.0
    return edge.astype(np.float32)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sample_positions(mask: np.ndarray, max_rows: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.where(mask)
    if yy.size > max_rows:
        keep = rng.choice(yy.size, size=max_rows, replace=False)
        yy, xx = yy[keep], xx[keep]
    return yy, xx


def append_samples(store: dict[str, list[np.ndarray]], comps: dict[str, np.ndarray], yy: np.ndarray, xx: np.ndarray) -> None:
    for mode in FREQ_MODES:
        store[mode].append(comps[mode][:, yy, xx].T.astype(np.float32, copy=False))


def collect_source(
    source: str,
    emb_map: dict[str, str],
    label_map: dict[str, str],
    ids: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
    phase: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    X = {mode: [] for mode in FREQ_MODES}
    for mode in FREQ_MODES:
        X[f"{mode}_height"] = []
    y = {task: [] for task in TASKS}
    y_height = []
    energy_rows = []
    edge_corrs = []
    used = 0
    t0 = time.time()

    for i, cid in enumerate(ids):
        emb_path = emb_map.get(cid)
        label_path = label_map.get(cid)
        if emb_path is None or label_path is None:
            continue

        emb = load_raster(emb_path)
        label = load_raster(label_path)
        _, h, w = emb.shape
        label, valid, height_valid = aggregate_label(label, h, w)
        masks = radial_masks(h, w, args.low_cut, args.mid_cut)
        comps, energy = fft_components(emb, masks)

        edge = label_edge_strength(label, valid)
        high_mag = np.sqrt(np.mean(comps["high"] ** 2, axis=0))
        edge_corrs.append(safe_corr(high_mag[valid > args.valid_frac], edge[valid > args.valid_frac]))
        energy_rows.append(energy)

        sample_mask = valid > args.valid_frac
        yy, xx = sample_positions(sample_mask, args.max_samples_per_id, rng)
        if yy.size == 0:
            continue

        append_samples(X, comps, yy, xx)
        y["building"].append((label[CH_BUILDING, yy, xx] > args.presence_threshold).astype(np.int8))
        y["vegetation"].append((label[CH_VEGETATION, yy, xx] > args.presence_threshold).astype(np.int8))
        y["water"].append((label[CH_WATER, yy, xx] > args.presence_threshold).astype(np.int8))

        height_mask = (
            (height_valid > args.valid_frac)
            & ((label[CH_BUILDING] > args.presence_threshold) | (label[CH_VEGETATION] > args.presence_threshold))
        )
        yy_h, xx_h = sample_positions(height_mask, args.max_height_samples_per_id, rng)
        if yy_h.size:
            for mode in FREQ_MODES:
                X[f"{mode}_height"].append(comps[mode][:, yy_h, xx_h].T.astype(np.float32, copy=False))
            y_height.append(label[CH_HEIGHT, yy_h, xx_h].astype(np.float32))

        used += 1
        if used >= args.max_ids_per_phase:
            break
        if (i + 1) % args.log_every == 0:
            print(f"  {source} {phase}: scanned={i+1}/{len(ids)} used={used} t={time.time()-t0:.1f}s")

    out_X = {}
    for key, parts in X.items():
        if parts:
            out_X[key] = np.concatenate(parts, axis=0)
    out_y = {task: np.concatenate(parts, axis=0) for task, parts in y.items() if parts}
    if y_height:
        out_y["height"] = np.concatenate(y_height, axis=0)

    mean_energy = {
        band: float(np.mean([row[band] for row in energy_rows])) if energy_rows else float("nan")
        for band in ("low", "mid", "high")
    }
    edge_stats = {
        "high_edge_corr": float(np.nanmean(edge_corrs)) if edge_corrs else float("nan"),
        "n_edge_corr": int(np.isfinite(edge_corrs).sum()),
    }
    return out_X, out_y, mean_energy, edge_stats


def fit_eval_classifiers(Xtr: dict[str, np.ndarray], ytr: dict[str, np.ndarray],
                         Xva: dict[str, np.ndarray], yva: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import roc_auc_score

    results = {}
    for mode in FREQ_MODES:
        mode_result = {}
        for task in TASKS:
            if task not in ytr or task not in yva:
                mode_result[f"{task}_auc"] = float("nan")
                continue
            if np.unique(ytr[task]).size < 2 or np.unique(yva[task]).size < 2:
                mode_result[f"{task}_auc"] = float("nan")
                continue
            X_train, X_eval = standardize_pair(Xtr[mode], Xva[mode])
            clf = RidgeClassifier(alpha=10.0, class_weight="balanced")
            clf.fit(X_train, ytr[task])
            score = clf.decision_function(X_eval)
            mode_result[f"{task}_auc"] = float(roc_auc_score(yva[task], score))
        aucs = [v for k, v in mode_result.items() if k.endswith("_auc") and np.isfinite(v)]
        mode_result["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
        results[mode] = mode_result
    return results


def fit_eval_height(Xtr: dict[str, np.ndarray], ytr: dict[str, np.ndarray],
                    Xva: dict[str, np.ndarray], yva: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    results = {}
    if "height" not in ytr or "height" not in yva:
        return {mode: {"height_r2": float("nan"), "height_rmse": float("nan")} for mode in FREQ_MODES}

    for mode in FREQ_MODES:
        key = f"{mode}_height"
        if key not in Xtr or key not in Xva or Xtr[key].shape[0] < 10 or Xva[key].shape[0] < 10:
            results[mode] = {"height_r2": float("nan"), "height_rmse": float("nan")}
            continue
        X_train, X_eval = standardize_pair(Xtr[key], Xva[key])
        reg = Ridge(alpha=100.0)
        reg.fit(X_train, np.log1p(ytr["height"]))
        pred = np.expm1(reg.predict(X_eval)).astype(np.float32)
        diff = pred - yva["height"]
        results[mode] = {
            "height_r2": float(r2_score(yva["height"], pred)),
            "height_rmse": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
        }
    return results


def standardize_pair(X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, dtype=np.float64)
    std = X_train.std(axis=0, dtype=np.float64)
    keep = std > MIN_FEATURE_STD
    if not np.any(keep):
        keep = std >= 0
        std = np.ones_like(std)
    mean = mean[keep]
    std = std[keep]
    return (
        ((X_train[:, keep] - mean) / std).astype(np.float32, copy=False),
        ((X_eval[:, keep] - mean) / std).astype(np.float32, copy=False),
    )


def interpret_source(metrics: dict[str, object]) -> str:
    probes = metrics["probes"]
    full_auc = probes["full"].get("mean_auc", float("nan"))
    low_mid_auc = probes["low_mid"].get("mean_auc", float("nan"))
    high_auc = probes["high"].get("mean_auc", float("nan"))
    full_r2 = probes["full"].get("height_r2", float("nan"))
    low_mid_r2 = probes["low_mid"].get("height_r2", float("nan"))
    edge_corr = metrics["edge"]["high_edge_corr"]
    high_energy = metrics["energy"]["high"]

    auc_ret = low_mid_auc / full_auc if np.isfinite(full_auc) and full_auc > 0 else float("nan")
    r2_drop = full_r2 - low_mid_r2 if np.isfinite(full_r2) and np.isfinite(low_mid_r2) else float("nan")

    if (
        np.isfinite(auc_ret)
        and auc_ret >= 0.95
        and (not np.isfinite(r2_drop) or r2_drop <= 0.03)
        and (not np.isfinite(high_auc) or high_auc <= full_auc - 0.05)
    ):
        return "low_mid_filter_promising"
    if np.isfinite(edge_corr) and edge_corr >= 0.20 and np.isfinite(high_auc) and high_auc >= full_auc - 0.03:
        return "high_frequency_contains_boundary_signal"
    if np.isfinite(high_energy) and high_energy > 0.35 and np.isfinite(edge_corr) and edge_corr < 0.05:
        return "high_frequency_likely_noisy"
    return "mixed_or_inconclusive"


def write_markdown_report(results: dict[str, object], out_path: Path) -> None:
    lines = [
        "# Frequency-Domain Embedding Diagnostic",
        "",
        "Interpretation rule of thumb: if `low_mid` keeps nearly the same AUC/R2 as `full` while `high` is weak, frequency filtering is a plausible denoising move. If `high` is predictive or its magnitude correlates with label edges, hard low-pass filtering risks removing useful boundary signal.",
        "",
        "| source | energy low/mid/high | high-edge corr | full mean AUC | low_mid mean AUC | high mean AUC | full height R2 | low_mid height R2 | call |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for source, metrics in results["sources"].items():
        energy = metrics["energy"]
        edge = metrics["edge"]["high_edge_corr"]
        probes = metrics["probes"]
        lines.append(
            f"| {source} | {energy['low']:.3f}/{energy['mid']:.3f}/{energy['high']:.3f} "
            f"| {edge:.3f} | {probes['full']['mean_auc']:.3f} | {probes['low_mid']['mean_auc']:.3f} "
            f"| {probes['high']['mean_auc']:.3f} | {probes['full']['height_r2']:.3f} "
            f"| {probes['low_mid']['height_r2']:.3f} | {metrics['interpretation']} |"
        )
    lines.append("")
    lines.append("## Per-Mode Probe Details")
    for source, metrics in results["sources"].items():
        lines.append("")
        lines.append(f"### {source}")
        lines.append("")
        lines.append("| mode | building AUC | vegetation AUC | water AUC | mean AUC | height R2 | height RMSE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for mode in FREQ_MODES:
            p = metrics["probes"][mode]
            lines.append(
                f"| {mode} | {p.get('building_auc', float('nan')):.3f} | "
                f"{p.get('vegetation_auc', float('nan')):.3f} | {p.get('water_auc', float('nan')):.3f} | "
                f"{p.get('mean_auc', float('nan')):.3f} | {p.get('height_r2', float('nan')):.3f} | "
                f"{p.get('height_rmse', float('nan')):.3f} |"
            )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=list(SOURCES))
    ap.add_argument("--split-file", default=str(REPO_DIR / "splits" / "split.json"))
    ap.add_argument("--output-dir", default=str(REPO_DIR / "runs" / "_frequency_probe"))
    ap.add_argument("--n-ids-train", type=int, default=80, help="0 = use all shuffled train ids")
    ap.add_argument("--n-ids-eval", type=int, default=40, help="0 = use all shuffled val ids")
    ap.add_argument("--max-ids-per-phase", type=int, default=10**9)
    ap.add_argument("--max-samples-per-id", type=int, default=700)
    ap.add_argument("--max-height-samples-per-id", type=int, default=700)
    ap.add_argument("--presence-threshold", type=float, default=0.0)
    ap.add_argument("--valid-frac", type=float, default=0.0)
    ap.add_argument("--low-cut", type=float, default=0.25)
    ap.add_argument("--mid-cut", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = load_split(Path(args.split_file))
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    if args.n_ids_train > 0:
        train_ids = train_ids[: args.n_ids_train]
    if args.n_ids_eval > 0:
        val_ids = val_ids[: args.n_ids_eval]

    label_map = index_tifs(LABEL_DIR)
    print(f"train ids={len(train_ids)} eval ids={len(val_ids)}")
    print(f"frequency cuts: low <= {args.low_cut:.2f}, mid <= {args.mid_cut:.2f}, high > {args.mid_cut:.2f}")
    print(f"output: {out_dir}")

    results = {
        "config": vars(args),
        "sources": {},
    }

    for source in args.sources:
        print("\n" + "=" * 78)
        print(f"Source: {source}")
        print("=" * 78)
        emb_map = index_tifs(SOURCES[source])
        print(f"embedding files={len(emb_map)}")

        Xtr, ytr, energy_tr, edge_tr = collect_source(source, emb_map, label_map, train_ids, args, rng, "train")
        Xva, yva, energy_va, edge_va = collect_source(source, emb_map, label_map, val_ids, args, rng, "eval")

        if "full" not in Xtr or "full" not in Xva:
            print(f"WARN: no samples for {source}; skipping")
            continue

        print(f"  train samples={Xtr['full'].shape[0]:,} eval samples={Xva['full'].shape[0]:,} D={Xtr['full'].shape[1]}")
        print(f"  height train={Xtr.get('full_height', np.empty((0, 0))).shape[0]:,} eval={Xva.get('full_height', np.empty((0, 0))).shape[0]:,}")
        print("  fitting probes ...")
        probes = fit_eval_classifiers(Xtr, ytr, Xva, yva)
        height = fit_eval_height(Xtr, ytr, Xva, yva)
        for mode in FREQ_MODES:
            probes[mode].update(height[mode])

        energy = {band: float(np.nanmean([energy_tr[band], energy_va[band]])) for band in ("low", "mid", "high")}
        edge = {
            "high_edge_corr": float(np.nanmean([edge_tr["high_edge_corr"], edge_va["high_edge_corr"]])),
            "n_edge_corr": int(edge_tr["n_edge_corr"] + edge_va["n_edge_corr"]),
        }
        metrics = {
            "energy": energy,
            "edge": edge,
            "n_train_samples": int(Xtr["full"].shape[0]),
            "n_eval_samples": int(Xva["full"].shape[0]),
            "n_train_height_samples": int(Xtr.get("full_height", np.empty((0, 0))).shape[0]),
            "n_eval_height_samples": int(Xva.get("full_height", np.empty((0, 0))).shape[0]),
            "feature_dim": int(Xtr["full"].shape[1]),
            "probes": probes,
        }
        metrics["interpretation"] = interpret_source(metrics)
        results["sources"][source] = metrics

        print(
            f"  energy low/mid/high={energy['low']:.3f}/{energy['mid']:.3f}/{energy['high']:.3f} "
            f"edge_corr={edge['high_edge_corr']:.3f}"
        )
        print(
            f"  mean AUC full={probes['full']['mean_auc']:.3f} low_mid={probes['low_mid']['mean_auc']:.3f} "
            f"high={probes['high']['mean_auc']:.3f}"
        )
        print(
            f"  height R2 full={probes['full']['height_r2']:.3f} low_mid={probes['low_mid']['height_r2']:.3f} "
            f"call={metrics['interpretation']}"
        )

        json_path = out_dir / "frequency_probe.json"
        md_path = out_dir / "frequency_probe_report.md"
        json_path.write_text(json.dumps(results, indent=2))
        write_markdown_report(results, md_path)

    json_path = out_dir / "frequency_probe.json"
    md_path = out_dir / "frequency_probe_report.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown_report(results, md_path)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
