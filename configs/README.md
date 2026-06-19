# Model Presets

`defaults.yml` is the global base recipe consumed by `train.py`. Use `active/`
for new competition experiments; those recipes override only the fields that
define a strategy. Very old settings should stay in the logs unless they become
competition-relevant again.

Active strategies (P3 two-stage line, all `model_type: xfusion_unet_hybrid_cross_source`):

- `active/xfusion_095_p3_dualtrunk_2stage.yml`: canonical two-stage train+purify
  recipe (stage-1 train, then presence-purify via `--init-checkpoint` +
  `--presence-trunk-grad-scale 0`); dual-trunk head splits seg and height.
- `active/xfusion_095_p3_split_trunk_fold0.yml`: single-stage split-trunk variant
  (height gets its own trunk; no seg/height head sharing).
- `active/xfusion_095_p3_baseline_fold0.yml`: P3 single-stage baseline.

Rule for new runs: start from one of these recipes and change only the small
part you are actively testing.

Prior champion / baseline recipes (`uw_gated_F`, `ae_tessera_gated`,
`xfusion_crosslevel`, `ae_only_baseline`, and the older `xfusion_*` fold
experiments) are archived under `legacy/`.

Run output contract:

- `resolved_config.yml`: final YAML-shaped training configuration after
  defaults, recipe, and CLI overrides are merged. This is the source of truth
  for reproducibility.
- `run_metadata.json`: execution context only: command, device, AMP, workers,
  optimizer/scheduler labels. Do not add model or loss parameters here.
- `metrics_summary.json`: compact result summary: best/last losses,
  components, selected model, input channels, and artifact paths.

Older runs may still contain `training_params.json`. New training runs do not
write it; `predict.py` only reads it as a legacy fallback when
`resolved_config.yml` is absent.
