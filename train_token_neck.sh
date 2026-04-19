#!/bin/bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emb2heights

PROJECT_DIR=/u/dingqi2/workspace/esa/emb2heights-backbone
DATA_DIR=/u/dingqi2/workspace/esa/data/train
TEST_DIR=/u/dingqi2/workspace/esa/data/test
LABELS_DIR=${DATA_DIR}/labels
SPLIT_FILE=${PROJECT_DIR}/splits/split.json
TAG=neck_v1

cd "${PROJECT_DIR}"

# (train_emb_subdir, test_emb_subdir, short_name)
RUNS=(
    "terramind_s2_emb  terramind_test_s2_emb  terramind_s2"
    "thor_s2_emb       thor_test_s2_emb       thor_s2"
)

for ROW in "${RUNS[@]}"; do
    read -r TRAIN_SUB TEST_SUB SHORT <<< "${ROW}"
    EXP_NAME="${TAG}_${SHORT}"
    TRAIN_EMB="${DATA_DIR}/${TRAIN_SUB}"
    TEST_EMB="${TEST_DIR}/${TEST_SUB}"
    RUN_DIR="${PROJECT_DIR}/runs/${EXP_NAME}"

    echo ">>> [${EXP_NAME}] TRAIN  $(date)"
    python train.py \
        --model-type token_neck \
        --train-embeddings-dir "${TRAIN_EMB}" \
        --train-targets-dir   "${LABELS_DIR}" \
        --experiment-name     "${EXP_NAME}" \
        --split-file          "${SPLIT_FILE}" \
        --num-workers 4

    echo ">>> [${EXP_NAME}] VAL-PREDICT  $(date)"
    python predict.py \
        --model-type token_neck \
        --experiment-name   "${EXP_NAME}" \
        --test-embeddings-dir "${TRAIN_EMB}" \
        --test-targets-dir    "${LABELS_DIR}" \
        --predictions-dir     "${RUN_DIR}/predictions_val"

    echo ">>> [${EXP_NAME}] EVAL  $(date)"
    python evaluate.py \
        --predictions-dir "${RUN_DIR}/predictions_val" \
        --targets-dir     "${LABELS_DIR}" \
        --split-file      "${SPLIT_FILE}" \
        | tee "${RUN_DIR}/eval_val.txt"

    echo ">>> [${EXP_NAME}] TEST-PREDICT (submission)  $(date)"
    python predict.py \
        --model-type token_neck \
        --experiment-name     "${EXP_NAME}" \
        --test-embeddings-dir "${TEST_EMB}" \
        --predictions-dir     "${RUN_DIR}/predictions_test"

    echo ">>> [${EXP_NAME}] DONE  $(date)"
done

echo "All runs finished at $(date)"
