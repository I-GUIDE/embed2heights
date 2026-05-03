"""Prediction postprocessing helpers."""

import numpy as np

from core.data.datasets import HEIGHT_NORM_CONSTANT


def prediction_to_numpy(pred_tensor, *, thresholds=None):
    """Convert a model output tensor to saved prediction layout."""
    pred = pred_tensor.cpu().numpy().astype(np.float32)
    pred[3] = pred[3] * HEIGHT_NORM_CONSTANT

    if thresholds is not None:
        for c, threshold in enumerate(thresholds):
            pred[c] = (pred[c] > threshold).astype(np.float32)

    return pred


def largest_component_size(mask):
    """Return largest 8-connected component size for a boolean 2D mask."""
    if not mask.any():
        return 0
    try:
        from scipy.ndimage import label as cc_label

        structure = np.ones((3, 3), dtype=np.uint8)
        comps, n_comp = cc_label(mask, structure=structure)
        if n_comp == 0:
            return 0
        sizes = np.bincount(comps.ravel())[1:]
        return int(sizes.max()) if len(sizes) else 0
    except ImportError:
        visited = np.zeros(mask.shape, dtype=bool)
        h, w = mask.shape
        best = 0
        ys, xs = np.nonzero(mask)
        for sy, sx in zip(ys, xs):
            if visited[sy, sx]:
                continue
            visited[sy, sx] = True
            stack = [(int(sy), int(sx))]
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for ny in range(max(0, y - 1), min(h, y + 2)):
                    for nx in range(max(0, x - 1), min(w, x + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            best = max(best, size)
        return best


def apply_water_cc_filter(mask, min_size):
    if int(min_size) <= 0 or not mask.any():
        return mask
    if largest_component_size(mask) < int(min_size):
        return np.zeros_like(mask, dtype=bool)
    return mask


def apply_height_channel(pred, params):
    if not params.get("height_affine", True):
        return pred[3]
    height_params = params["height_affine_params"]
    b = height_params["building"]
    v = height_params["vegetation"]
    h_b = np.maximum(0.0, b["a"] * pred[3] + b["b"])
    h_v = np.maximum(0.0, v["a"] * pred[3] + v["b"])
    p_b = np.clip(pred[0], 0.0, 1.0)
    p_v = np.clip(pred[1], 0.0, 1.0)
    fg = 1.0 - (1.0 - p_b) * (1.0 - p_v)
    denom = p_b + p_v + 1e-6
    h_fg = (p_b * h_b + p_v * h_v) / denom
    return fg * h_fg + (1.0 - fg) * pred[3]


def apply_height_affine_array(arr, params):
    out = arr.astype(np.float32, copy=True)
    if params.get("height_affine", False):
        out[3] = np.maximum(0.0, apply_height_channel(out, params)).astype(np.float32)
    out[:3] = np.clip(out[:3], 0.0, 1.0)
    return out


def binarize_predictions(
    pred_dir,
    output_dir,
    thresholds,
    water_cc_min_size=0,
):
    from pathlib import Path

    files = sorted(Path(pred_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds_arr = np.asarray(thresholds, dtype=np.float32)

    for path in files:
        arr = np.load(path).astype(np.float32)
        for channel in range(3):
            arr[channel] = (arr[channel] > thresholds_arr[channel]).astype(np.float32)
        if water_cc_min_size > 0:
            water_mask = apply_water_cc_filter(arr[2].astype(bool), water_cc_min_size)
            arr[2] = water_mask.astype(np.float32)
        np.save(output_dir / path.name, arr)
    return len(files)
