# Model Presets

`defaults.yml` is the global base recipe consumed by `train.py`. Use `active/`
for new competition experiments; those recipes override only the fields that
define a strategy. Very old settings should stay in the logs unless they become
competition-relevant again.

Active strategies:

- `active/ae_tessera_gated.yml`: best two-modal AlphaEarth + Tessera line (`model_type: ae_tessera_gated`).
- `active/xfusion_crosslevel.yml`: best three-modal TerraMind-S2 + AlphaEarth + Tessera line (`model_type: xfusion_crosslevel`).
- `active/ae_only_baseline.yml`: AlphaEarth-only fallback and sanity-check line (`model_type: ae_only`).

Rule for new runs: start from one of these three recipes and change only the
small part you are actively testing.

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
