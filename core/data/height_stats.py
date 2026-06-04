"""
Adaptive nDSM / height normalization (skew-driven z-score vs log1p vs Yeo–Johnson),
mirroring the flood-forecasting regression pipeline. Training fits stats on the train
split; inference loads the same stats for denormalization to meters.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

STATS_FILENAME = "height_regression_stats.pkl"


def _clean_raster_array(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_rule(values: np.ndarray, nonnegative_hint: bool) -> Dict[str, Any]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"transform": "zscore", "mean": 0.0, "std": 1.0, "lambda": None}

    try:
        from scipy.stats import skew as _skew
        from scipy.stats import yeojohnson as _yeojohnson
    except ImportError:
        m = float(np.mean(values))
        sd = float(np.std(values))
        if sd < 1e-8:
            sd = 1.0
        return {"transform": "zscore", "mean": m, "std": sd, "lambda": None}

    s = float(_skew(values, bias=False)) if values.size >= 3 else 0.0
    nonneg = nonnegative_hint and np.min(values) >= 0.0
    transform = "zscore"
    lam = None
    transformed = values.astype(np.float64)
    if abs(s) > 1.0:
        if nonneg:
            transform = "log1p"
            transformed = np.log1p(transformed)
        else:
            transform = "yeojohnson"
            transformed, lam = _yeojohnson(transformed)

    m = float(np.mean(transformed))
    sd = float(np.std(transformed))
    if sd < 1e-8:
        sd = 1.0
    return {"transform": transform, "mean": m, "std": sd, "lambda": lam}


def _height_valid_mask(raw_target: np.ndarray) -> np.ndarray:
    """Same validity as datasets._prepare_target for channel 3."""
    global_valid = ~np.all(raw_target == 0, axis=0)
    has_landcover = (
        (raw_target[0] > 0) | (raw_target[1] > 0) | (raw_target[2] > 0)
    )
    ndsm_hole = (raw_target[3] == 0) & has_landcover
    return global_valid & ~ndsm_hole


def collect_height_meters_from_label_pairs(
    pairs: List[Tuple],
    *,
    max_pixels: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Gather raw height (meters) from label GeoTIFFs using train-valid pixels."""
    chunks: List[np.ndarray] = []
    label_paths = {p[-1] for p in pairs if p[-1] is not None}
    for path in sorted(label_paths):
        with rasterio.open(path) as src:
            raw = _clean_raster_array(src.read())
        if raw.shape[0] < 4:
            continue
        mask = _height_valid_mask(raw)
        h = np.maximum(raw[3].astype(np.float64), 0.0)
        vals = h[mask]
        if vals.size:
            chunks.append(vals.ravel())

    if not chunks:
        return np.array([], dtype=np.float64)

    all_v = np.concatenate(chunks, axis=0)
    if max_pixels is not None and all_v.size > max_pixels:
        rng = np.random.RandomState(seed)
        idx = rng.choice(all_v.size, size=max_pixels, replace=False)
        all_v = all_v[idx]
    return all_v


def fit_height_regression_stats(values_m: np.ndarray) -> Dict[str, Any]:
    """Fit height stats; heights are nonnegative -> skew rule uses log1p when |skew|>1."""
    values_m = values_m[np.isfinite(values_m)]
    if values_m.size == 0:
        return {
            "transform": "zscore",
            "mean": 0.0,
            "std": 1.0,
            "lambda": None,
            "original_mean": 0.0,
            "original_std": 1.0,
        }
    stat = _fit_rule(values_m, nonnegative_hint=True)
    stat["original_mean"] = float(np.mean(values_m))
    stat["original_std"] = float(np.std(values_m))
    return stat


def save_height_stats(path: str, stats: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_height_stats(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_height_numpy(raw_m: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Meters -> model target space (transform then z-score)."""
    raw_m = np.maximum(np.asarray(raw_m, dtype=np.float64), 0.0)
    tr = stats.get("transform", "zscore")
    if tr == "log1p":
        tv = np.log1p(raw_m)
    elif tr == "yeojohnson":
        try:
            from scipy.stats import yeojohnson
        except ImportError:
            tv = raw_m
        else:
            lam = stats.get("lambda", None)
            tv = yeojohnson(raw_m, lmbda=lam)
    else:
        tv = raw_m
    return (tv - stats["mean"]) / (stats["std"] + 1e-8)


def denormalize_height_numpy(norm: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Model output / label space -> meters."""
    norm = np.asarray(norm, dtype=np.float64)
    t = norm * stats["std"] + stats["mean"]
    tr = stats.get("transform", "zscore")
    if tr == "log1p":
        return np.expm1(t)
    if tr == "yeojohnson":
        return _inv_yeojohnson_numpy(t, stats.get("lambda", None))
    return t


def _inv_yeojohnson_numpy(t: np.ndarray, lam: Optional[float]) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    lam = 1.0 if lam is None else float(lam)
    pos = t >= 0
    neg = ~pos
    if abs(lam) > 1e-8:
        base = lam * t[pos] + 1.0
        out[pos] = np.where(
            base > 0,
            np.power(np.maximum(base, 0.0), 1.0 / lam) - 1.0,
            0.0,
        )
    else:
        out[pos] = np.expm1(t[pos])
    if abs(2.0 - lam) > 1e-8:
        base2 = 1.0 - (2.0 - lam) * t[neg]
        out[neg] = np.where(
            base2 > 0,
            1.0 - np.power(np.maximum(base2, 0.0), 1.0 / (2.0 - lam)),
            0.0,
        )
    else:
        out[neg] = 1.0 - np.exp(-t[neg])
    return out


def inv_yeojohnson_torch(t, lam):
    """Yeo–Johnson inverse, PyTorch; used in loss / GPU path."""
    import torch

    lam = 1.0 if lam is None else float(lam)
    ge0 = t >= 0
    lt0 = ~ge0
    pos_out = torch.zeros_like(t)
    neg_out = torch.zeros_like(t)

    if abs(lam) > 1e-8:
        b = lam * t + 1.0
        pos_out = torch.where(
            ge0 & (b > 0),
            torch.pow(b.clamp(min=0.0), 1.0 / lam) - 1.0,
            torch.zeros_like(t),
        )
    else:
        pos_out = torch.where(ge0, torch.expm1(t.clamp(min=0.0)), torch.zeros_like(t))

    if abs(2.0 - lam) > 1e-8:
        b2 = 1.0 - (2.0 - lam) * t
        neg_out = torch.where(
            lt0 & (b2 > 0),
            1.0 - torch.pow(b2.clamp(min=0.0), 1.0 / (2.0 - lam)),
            torch.zeros_like(t),
        )
    else:
        neg_out = torch.where(lt0, 1.0 - torch.exp(-t), torch.zeros_like(t))

    return pos_out + neg_out


def denormalize_height_torch(norm, stats):
    """Inverse height normalization on GPU tensors."""
    import torch

    mean = torch.as_tensor(stats["mean"], device=norm.device, dtype=norm.dtype)
    std = torch.as_tensor(stats["std"], device=norm.device, dtype=norm.dtype)
    t = norm * std + mean
    tr = stats.get("transform", "zscore")
    if tr == "log1p":
        return torch.expm1(t)
    if tr == "yeojohnson":
        return inv_yeojohnson_torch(t, stats.get("lambda"))
    return t


def normalize_height_torch(raw_m, stats):
    """Meters (nonnegative) -> normalized target; for soft-bin center construction."""
    import torch

    raw_m = raw_m.clamp(min=0.0)
    tr = stats.get("transform", "zscore")
    if tr == "log1p":
        tv = torch.log1p(raw_m)
    elif tr == "yeojohnson":
        raise NotImplementedError(
            "Use numpy normalize_height_numpy for Yeo–Johnson (e.g. soft-bin init)."
        )
    else:
        tv = raw_m

    mean = torch.as_tensor(stats["mean"], device=raw_m.device, dtype=raw_m.dtype)
    std = torch.as_tensor(stats["std"], device=raw_m.device, dtype=raw_m.dtype)
    return (tv - mean) / (std + 1e-8)


def height_stats_path_for_exp(output_dir: str, experiment_name: str) -> str:
    return os.path.join(output_dir, experiment_name, STATS_FILENAME)
