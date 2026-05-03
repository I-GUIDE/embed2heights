"""Inference helpers."""

from .calibration import (
    build_oof_report,
    collect_oof_records,
    eval_records,
    fit_params,
    format_metrics,
    run_nested_oof_cv,
    sweep_thresholds,
    write_threshold_report,
)
from .ensemble import (
    ensemble_mean,
    ensemble_weighted,
    index_prediction_dir,
    load_weighted_ensemble_spec,
)
from .postprocess import prediction_to_numpy
from .postprocess import (
    apply_height_affine_array,
    apply_water_cc_filter,
    binarize_predictions,
    largest_component_size,
)
from .predict import (
    batched,
    input_channels,
    invert_tensor,
    predict_batch,
    transform_input,
    transform_tensor,
    tta_views,
)
from .submission import (
    json_safe,
    package_submission,
    prediction_output_id,
    validate_prediction_dir,
    write_prediction_config,
)


__all__ = [
    "batched",
    "apply_height_affine_array",
    "apply_water_cc_filter",
    "binarize_predictions",
    "build_oof_report",
    "collect_oof_records",
    "ensemble_mean",
    "ensemble_weighted",
    "eval_records",
    "fit_params",
    "format_metrics",
    "index_prediction_dir",
    "input_channels",
    "invert_tensor",
    "json_safe",
    "largest_component_size",
    "load_weighted_ensemble_spec",
    "package_submission",
    "predict_batch",
    "prediction_output_id",
    "prediction_to_numpy",
    "run_nested_oof_cv",
    "sweep_thresholds",
    "transform_input",
    "transform_tensor",
    "tta_views",
    "validate_prediction_dir",
    "write_prediction_config",
    "write_threshold_report",
]
