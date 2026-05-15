#!/bin/bash
#
# Example Slurm entrypoint for this repository.
#
# Usage:
#   1. Edit ACCOUNT, PROJECT_DIR, DATA_DIR, and Conda environment if needed.
#   2. Submit with: sbatch sample.sbatch
#
# This example trains the active AE+Tessera gated experiment, then runs
# validation prediction/evaluation. Test prediction and submission creation are
# included as an optional final stage.

#SBATCH --job-name=ae_tessera_gated
#SBATCH --output=slurm-%j-ae_tessera_gated.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_a100
# NOTE: Add #SBATCH --account=<your-slurm-account> here for clusters that
# require an account, or set SBATCH_ACCOUNT in your environment.

set -euo pipefail

echo "Start at: $(date)"
echo "Running node: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-local}"

# ---- User settings ---------------------------------------------------------

# PROJECT_DIR defaults to where this script lives. Override via env var when
# submitting from elsewhere, e.g.  PROJECT_DIR=/path/to/clone sbatch sample.sbatch
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
CONDA_ENV="${CONDA_ENV:-emb2heights}"

EXP_NAME=ae_tessera_gated_v001
CONFIG=${PROJECT_DIR}/configs/active/ae_tessera_gated.yml
SPLIT_FILE=${PROJECT_DIR}/splits/split.json

TRAIN_DIR=${DATA_DIR}/train
TEST_DIR=${DATA_DIR}/test

ALPHA_TRAIN=${TRAIN_DIR}/alphaearth_emb
TESSERA_TRAIN=${TRAIN_DIR}/tessera_emb
LABELS_DIR=${TRAIN_DIR}/labels

ALPHA_TEST=${TEST_DIR}/alphaearth_test_emb
TESSERA_TEST=${TEST_DIR}/tessera_test_emb

# Set this to 1 when you also want to create a competition submission zip.
RUN_TEST_SUBMISSION=${RUN_TEST_SUBMISSION:-0}

# ---- Environment -----------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
mkdir -p "runs/${EXP_NAME}" submission

echo ""
echo "=========================================================="
echo ">>> [${EXP_NAME}] train ae_tessera_gated at $(date)"
echo "=========================================================="
python train.py \
    --config "${CONFIG}" \
    --experiment-name "${EXP_NAME}" \
    --output-dir "${PROJECT_DIR}/runs" \
    --train-embeddings-dir "${ALPHA_TRAIN}" \
    --secondary-train-embeddings-dir "${TESSERA_TRAIN}" \
    --train-targets-dir "${LABELS_DIR}" \
    --split-file "${SPLIT_FILE}" \
    2>&1 | tee "runs/${EXP_NAME}/train.log"

echo ""
echo ">>> [${EXP_NAME}] predict validation at $(date)"
python predict.py \
    --experiment-name "${EXP_NAME}" \
    --base-dir "${PROJECT_DIR}/runs" \
    --model-type ae_tessera_gated \
    --test-embeddings-dir "${ALPHA_TRAIN}" \
    --secondary-test-embeddings-dir "${TESSERA_TRAIN}" \
    --test-targets-dir "${LABELS_DIR}" \
    2>&1 | tee "runs/${EXP_NAME}/predict_val.log"

echo ""
echo ">>> [${EXP_NAME}] evaluate validation at $(date)"
python evaluate.py \
    --runs-dir "${PROJECT_DIR}/runs" \
    --labels-dir "${LABELS_DIR}" \
    --only "${EXP_NAME}" \
    --val-only \
    --split-file "${SPLIT_FILE}" \
    2>&1 | tee "runs/${EXP_NAME}/evaluate_val.txt"

echo ""
echo ">>> [${EXP_NAME}] threshold sweep at $(date)"
python tools/sweep_thresholds.py \
    --pred-dir "runs/${EXP_NAME}/predictions" \
    --labels-dir "${LABELS_DIR}" \
    --split-file "${SPLIT_FILE}" \
    --grid-start 0.05 \
    --grid-stop 0.95 \
    --grid-step 0.01 \
    --water-k-grid 0,4,8,12,16,24,32 \
    --output-json "runs/${EXP_NAME}/threshold_sweep.json" \
    2>&1 | tee "runs/${EXP_NAME}/threshold_sweep.txt"

if [[ "${RUN_TEST_SUBMISSION}" == "1" ]]; then
    echo ""
    echo ">>> [${EXP_NAME}] predict test at $(date)"
    python predict.py \
        --experiment-name "${EXP_NAME}" \
        --base-dir "${PROJECT_DIR}/runs" \
        --model-type ae_tessera_gated \
        --test-embeddings-dir "${ALPHA_TEST}" \
        --secondary-test-embeddings-dir "${TESSERA_TEST}" \
        --predictions-dir "runs/${EXP_NAME}/test_predictions" \
        2>&1 | tee "runs/${EXP_NAME}/predict_test.log"

    echo ""
    echo ">>> [${EXP_NAME}] make submission at $(date)"
    python tools/make_submission.py \
        --pred-dir "runs/${EXP_NAME}/test_predictions" \
        --sweep-pred-dir "runs/${EXP_NAME}/predictions" \
        --labels-dir "${LABELS_DIR}" \
        --split-file "${SPLIT_FILE}" \
        --grid-start 0.05 \
        --grid-stop 0.95 \
        --grid-step 0.01 \
        --threshold-report "runs/${EXP_NAME}/submission_thresholds.json" \
        --binarized-output-dir "runs/${EXP_NAME}/test_predictions_binary" \
        --output "submission/${EXP_NAME}.zip" \
        2>&1 | tee "runs/${EXP_NAME}/make_submission.log"
fi

echo ">>> [${EXP_NAME}] done at $(date)"
echo "End at: $(date)"
