#!/bin/bash
# Launch the full ringw ep70 ensemble pipeline:
#   1. generate the 15 configs (seeds 0/1/42 x folds 0-4, 70 epochs)
#   2. train+predict array, 5 GPUs max concurrent (array 0-14%5, no exclusive)
#   3. ensemble + per-area-code sweep + submission packaging (CPU partition),
#      gated on the whole array finishing OK
#
# Usage: bash launch_ringw_ep70.sh
# Final artifacts: submission/ens_ringw_ep70_binary_global.zip
#                  submission/ens_ringw_ep70_binary_percode.zip
#                  runs/ens_ringw_ep70/SUMMARY.md

set -euo pipefail

PROJECT_DIR=/u/dingqi2/workspace/esa/embed2heights/.worktrees/ring-weighted-presence
cd "${PROJECT_DIR}"
mkdir -p slurm_logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emb2heights

echo "--- generating configs ---"
python tools/gen_ringw_ep70_configs.py

echo "--- submitting train+predict array (0-14%5) ---"
TRAIN_JOB=$(sbatch --parsable train_predict_ringw_ep70.sbatch)
echo "train array job: ${TRAIN_JOB}"

echo "--- submitting ensemble+sweep job (afterok:${TRAIN_JOB}) ---"
ENS_JOB=$(sbatch --parsable --dependency=afterok:"${TRAIN_JOB}" ensemble_sweep_ringw_ep70.sbatch)
echo "ensemble job: ${ENS_JOB}"

echo
echo "Pipeline submitted. Monitor with: squeue -u ${USER}"
echo "When ${ENS_JOB} finishes, read runs/ens_ringw_ep70/SUMMARY.md and upload"
echo "submission/ens_ringw_ep70_binary_global.zip (and optionally _percode.zip)"
echo "to the leaderboard."
