#!/bin/bash

#SBATCH --account=bcrm-tgirails
#SBATCH --job-name=exp-N-base48
#SBATCH --output=slurm-%j-exp-N-base48.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_a100

set -euo pipefail

echo "Start at: $(date)"
echo "Running node: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emb2heights

PROJECT_DIR=/u/dingqi2/workspace/esa/emb2heights-backbone
DATA_DIR=/u/dingqi2/workspace/esa/data/train
ALPHA_DIR=${DATA_DIR}/alphaearth_emb
TESSERA_DIR=${DATA_DIR}/tessera_emb
LABELS_DIR=${DATA_DIR}/labels
SPLIT_FILE=${PROJECT_DIR}/splits/split.json

cd "${PROJECT_DIR}"

# Experiment N — J with widened LightUNet (base_ch 32 -> 48).
#
# Motivation: HRNet-W18/W32 swaps under-performed vs LightUNet under the same
# recipe, suggesting the capacity gap wasn't what held LightUNet back. But
# LightUNet itself is tiny (~2.7M params in J). Before concluding "capacity
# doesn't matter," scale the existing architecture (which already has the
# right inductive bias for pixel-aligned embeddings) instead of swapping for
# a different one.
#
# Base channels (b, 2b, 4b, 8b): 32->64->128->256 becomes 48->96->192->384.
# Params go from 2.75M -> 5.44M (~2x). Head in_ch tracks base.
#
# All other knobs copied from J (current best on record):
#   - presence_centered loss, aux_weight 1.0, presence_tversky 1.0
#   - tessera_hidden_ch 96 / depth 2 (proven helpful)
#   - height_specialist_depth 2 (E: +0.0102)
#   - veg_height_boost 1.5 (J: linear-regime compromise of G)
EXP_NAME="alphaearth_tessera_iou_fusion_N_base48"

echo ""
echo "=========================================================="
echo ">>> [${EXP_NAME}] train start at $(date)"
echo "=========================================================="
mkdir -p "runs/${EXP_NAME}"

python train.py \
    --model-type tessera_iou_fusion \
    --train-embeddings-dir "${ALPHA_DIR}" \
    --secondary-train-embeddings-dir "${TESSERA_DIR}" \
    --train-targets-dir "${LABELS_DIR}" \
    --experiment-name "${EXP_NAME}" \
    --split-file "${SPLIT_FILE}" \
    --loss-preset presence_centered \
    --aux-weight 1.0 \
    --presence-tversky-weight 1.0 \
    --fraction-mae-weight 0.1 \
    --tessera-presence-ch 16 \
    --tessera-hidden-ch 96 \
    --tessera-hidden-depth 2 \
    --height-specialist-depth 2 \
    --veg-height-boost 1.5 \
    --lightunet-base-ch 48 \
    --num-workers 4 \
    2>&1 | tee "runs/${EXP_NAME}/train.log"

echo ""
echo ">>> [${EXP_NAME}] predict validation at $(date)"
python predict.py \
    --experiment-name "${EXP_NAME}" \
    --model-type tessera_iou_fusion \
    --test-embeddings-dir "${ALPHA_DIR}" \
    --secondary-test-embeddings-dir "${TESSERA_DIR}" \
    --test-targets-dir "${LABELS_DIR}" \
    2>&1 | tee "runs/${EXP_NAME}/predict_val.log"

echo ""
echo ">>> [${EXP_NAME}] evaluate validation at $(date)"
python evaluate.py \
    --only "${EXP_NAME}" \
    --val-only \
    --split-file "${SPLIT_FILE}" \
    2>&1 | tee "runs/${EXP_NAME}/evaluate_val.txt"

echo ""
echo ">>> [${EXP_NAME}] threshold sweep at $(date)"
python tools/sweep_thresholds.py \
    --pred-dir "runs/${EXP_NAME}/predictions" \
    --split-file "${SPLIT_FILE}" \
    --grid-start 0.05 \
    --grid-stop 0.95 \
    --grid-step 0.01 \
    2>&1 | tee "runs/${EXP_NAME}/threshold_sweep.txt"

echo ">>> [${EXP_NAME}] done at $(date)"
echo "End at: $(date)"
