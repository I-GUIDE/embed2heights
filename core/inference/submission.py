"""Prediction id + run-metadata helpers."""

import json

from core.data.discovery import normalize_core_id, submission_id


def json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
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
            "output_shape": json_safe(pred_shape),
        },
    }
    with open(path, "w") as f:
        json.dump(json_safe(config), f, indent=2)
