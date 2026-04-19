#!/bin/bash
# Predict + evaluate all 6 baselines trained by run.sbatch.
# Usage:
#   bash predict_and_evaluate.sh
# Run on a GPU node (e.g. srun --gres=gpu:1 ... bash predict_and_evaluate.sh),
# or wrap in its own sbatch if you want to queue it after training.

set -euo pipefail

PROJECT_DIR=/u/dingqi2/workspace/esa/emb2heights-backbone
DATA_DIR=/u/dingqi2/workspace/esa/data/train
LABELS_DIR=${DATA_DIR}/labels
SPLIT_FILE=${PROJECT_DIR}/splits/split.json
TAG=baseline_v35

cd "${PROJECT_DIR}"

# (embedding_subdir, model_type, short_name)
BASELINES=(
    # "alphaearth_emb   lightunet         alphaearth"
    # "tessera_emb      lightunet         tessera"
    "terramind_s1_emb decoder_residual  terramind_s1"
    "terramind_s2_emb decoder_residual  terramind_s2"
    "thor_s1_emb      decoder_residual  thor_s1"
    "thor_s2_emb      decoder_residual  thor_s2"
)

EXP_NAMES=()

# --- 1. Predict each baseline on the train embeddings (val IDs filtered later) ---
for entry in "${BASELINES[@]}"; do
    read -r EMB_SUBDIR MODEL_TYPE SHORT_NAME <<< "${entry}"
    EXP_NAME="${TAG}_${SHORT_NAME}"
    EXP_NAMES+=("${EXP_NAME}")
    EMB_DIR="${DATA_DIR}/${EMB_SUBDIR}"

    echo ""
    echo "=========================================================="
    echo ">>> Predict ${EXP_NAME}  (model=${MODEL_TYPE})"
    echo ">>> $(date)"
    echo "=========================================================="

    python predict.py \
        --experiment-name "${EXP_NAME}" \
        --model-type "${MODEL_TYPE}" \
        --test-embeddings-dir "${EMB_DIR}" \
        --test-targets-dir "${LABELS_DIR}"
done

# --- 2. Evaluate all baselines on the val split in a single call ---
echo ""
echo "=========================================================="
echo ">>> Evaluate on val split"
echo ">>> $(date)"
echo "=========================================================="

python evaluate.py \
    --only "${EXP_NAMES[@]}" \
    --val-only \
    --split-file "${SPLIT_FILE}"
