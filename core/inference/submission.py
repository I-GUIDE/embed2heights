"""Prediction metadata and submission-related helpers."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from core.data.discovery import normalize_core_id, submission_id


def json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def prediction_output_id(embedding_path, *, label_mode=False):
    return normalize_core_id(embedding_path) if label_mode else submission_id(embedding_path)


def write_prediction_config(path, *, args, exp_dir, model_path, predictions_dir,
                            selected_model, n_channels, dataset_cls, train_cfg,
                            train_config_raw, sample_count, pred_shape):
    config = {
        "schema_version": 1,
        "experiment": {
            "experiment_name": args.experiment_name,
            "exp_dir": exp_dir,
            "model_path": model_path,
            "selected_model": selected_model,
            "input_channels": json_safe(n_channels),
            "dataset": dataset_cls.__name__,
        },
        "training_config": {
            "source": train_cfg.get("_config_source"),
            "source_config": train_cfg.get("_source_config"),
            "recipe": train_cfg.get("_recipe", {}),
            "resolved_config_available": train_config_raw is not None,
        },
        "prediction": {
            "test_embeddings_dir": args.test_embeddings_dir,
            "secondary_test_embeddings_dir": args.secondary_test_embeddings_dir,
            "token_test_embeddings_dir": args.token_test_embeddings_dir,
            "test_targets_dir": args.test_targets_dir,
            "predictions_dir": predictions_dir,
            "patch_size": args.patch_size,
            "max_samples": args.max_samples,
            "sample_count": sample_count,
            "thresholds": args.thresholds,
            "tta": args.tta,
            "output_shape": json_safe(pred_shape),
        },
    }
    with open(path, "w") as f:
        json.dump(json_safe(config), f, indent=2)


def package_submission(pred_dir, output_zip):
    """Package .npy predictions under the required predictions/ zip layout."""
    pred_dir = Path(pred_dir)
    output_zip = Path(output_zip)
    files = sorted(pred_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")

    zip_cli = shutil.which("zip")
    if zip_cli is None:
        raise RuntimeError("/usr/bin/zip (Info-ZIP) not found on PATH.")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="make_submission_") as tmpd:
        tmp_path = Path(tmpd)
        staged = tmp_path / "predictions"
        staged.mkdir()
        for path in files:
            shutil.copy2(path, staged / path.name)
        if output_zip.exists():
            output_zip.unlink()
        proc = subprocess.run(
            [zip_cli, "-r", "-q", str(output_zip.absolute()), "predictions"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"zip failed (rc={proc.returncode}):\n{proc.stderr}")


def validate_prediction_dir(pred_dir, expected_count=946, sample_only=True):
    files = sorted(Path(pred_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {pred_dir}")
    if expected_count > 0 and len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} files, got {len(files)} in {pred_dir}")

    check_files = files
    if sample_only and len(files) > 3:
        check_files = [files[0], files[len(files) // 2], files[-1]]
    shapes = set()
    for path in check_files:
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[0] != 4:
            raise ValueError(f"{path.name}: expected shape [4, H, W], got {arr.shape}")
        if arr.dtype not in (np.float32, np.float64):
            raise ValueError(f"{path.name}: expected float dtype, got {arr.dtype}")
        cls = arr[:3]
        if cls.min() < -1e-4 or cls.max() > 1 + 1e-4:
            print(f"WARN {path.name}: class channels outside [0,1].", file=sys.stderr)
        if arr[3].min() < -1e-4:
            print(f"WARN {path.name}: height channel has negative values.", file=sys.stderr)
        shapes.add(tuple(arr.shape))
    return shapes
