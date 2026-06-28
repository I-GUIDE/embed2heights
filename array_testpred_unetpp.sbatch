#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --job-name=dmwav_test
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --partition=gpu,gpu_a100
#SBATCH --array=0-14
# Test-predict the 15 delmask+wavelet models (stage-1 seg + stage-2 purify height) on the
# 946 TEST tiles. index = seed*5 + fold. No staging (single pass over test set).
set -euo pipefail
SEED=$(( SLURM_ARRAY_TASK_ID / 5 )); FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
CFG=xfusion_095_p3_2stage_softbin_covgt10_delmask_unetpp
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
RUNS_DIR=${PROJECT_DIR}/runs; T=/u/dingqi2/workspace/esa/data/test
EXP=${CFG}_s${SEED}_f${FOLD}
source ~/miniconda3/etc/profile.d/conda.sh; conda activate emb2heights
cd "${PROJECT_DIR}"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
for E in "${EXP}" "${EXP}_purify"; do
  [ -f "runs/${E}/model_best.pth" ] || { echo "skip ${E}: no ckpt"; continue; }
  python predict.py --experiment-name "${E}" --base-dir "${RUNS_DIR}" \
    --test-embeddings-dir "${T}/alphaearth_test_emb" --secondary-test-embeddings-dir "${T}/tessera_test_emb" \
    --token-test-embeddings-dir "${T}/terramind_test_s1_emb" --secondary-token-test-embeddings-dir "${T}/terramind_test_s2_emb" \
    --third-token-test-embeddings-dir "${T}/thor_test_s1_emb" --fourth-token-test-embeddings-dir "${T}/thor_test_s2_emb" \
    --predictions-dir "${RUNS_DIR}/${E}/test_predictions" --patch-size 256 --max-samples 0 \
    2>&1 | tee "${RUNS_DIR}/${E}/predict_test.log"
done
echo "test-predict ${EXP} (+purify) done"
