#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --job-name=xf095_2stage
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu,gpu_a100

# Two-stage single-job pipeline (see slurm-cluster-practices: ONE submission
# per experiment — train, purify, predict and evaluate all run here).
#
#   stage 1 (80 ep, coupled)   -> runs/<EXP>/model_best.pth        = IoU channels 0-2
#   stage 2 (20 ep, purified)  -> runs/<EXP>_purify/model_best.pth = height channel 3
#   predict+evaluate both checkpoints on the fold's val split.
#
# Usage:  sbatch two_stage_train_eval.sbatch [FOLD] [SEED]
#   FOLD (default 0) selects splits/group_code_5fold_seed42/fold_<FOLD>
#   SEED (default 0) sets the training seed.
# Walltime budget: ~6h stage 1 (80ep) + ~1.3h stage 2 + ~0.5h dual predict/eval.

set -euo pipefail

FOLD=${1:-0}
SEED=${2:-0}

# SLURM runs the script from a spool copy, so $0 is not the worktree.
# SLURM_SUBMIT_DIR is the dir sbatch was invoked from (the worktree); fall
# back to $0's dir for direct (non-SLURM) execution.
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
CONFIG=${PROJECT_DIR}/configs/active/xfusion_095_p3_dualtrunk_2stage.yml
EXP=xf095_p3_2stage_s${SEED}_f${FOLD}
RUNS_DIR=${PROJECT_DIR}/runs
DATA_DIR=/u/dingqi2/workspace/esa/data/train
SPLIT_FILE=${PROJECT_DIR}/splits/group_code_5fold_seed42/fold_${FOLD}/split.json

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emb2heights

cd "${PROJECT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ---- stage 1: coupled training (IoU checkpoint) ----
mkdir -p "runs/${EXP}"
python train.py --config "${CONFIG}" \
  --experiment-name "${EXP}" \
  --split-file "${SPLIT_FILE}" \
  --seed "${SEED}" \
  2>&1 | tee "runs/${EXP}/train.log"

# ---- stage 2: backbone purification (height checkpoint) ----
mkdir -p "runs/${EXP}_purify"
python train.py --config "${CONFIG}" \
  --experiment-name "${EXP}_purify" \
  --split-file "${SPLIT_FILE}" \
  --seed "${SEED}" \
  --init-checkpoint "runs/${EXP}/model_best.pth" \
  --presence-trunk-grad-scale 0.0 \
  --epochs 20 \
  --lr 0.00015 \
  2>&1 | tee "runs/${EXP}_purify/train.log"

# ---- predict + evaluate both checkpoints on the val split ----
for E in "${EXP}" "${EXP}_purify"; do
  python predict.py \
    --experiment-name "${E}" \
    --base-dir "${RUNS_DIR}" \
    --test-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
    --secondary-test-embeddings-dir "${DATA_DIR}/tessera_emb" \
    --token-test-embeddings-dir "${DATA_DIR}/terramind_s1_emb" \
    --secondary-token-test-embeddings-dir "${DATA_DIR}/terramind_s2_emb" \
    --third-token-test-embeddings-dir "${DATA_DIR}/thor_s1_emb" \
    --fourth-token-test-embeddings-dir "${DATA_DIR}/thor_s2_emb" \
    --test-targets-dir "${DATA_DIR}/labels" \
    --patch-size 256 \
    --max-samples 0 \
    2>&1 | tee "${RUNS_DIR}/${E}/predict_val.log"

  python evaluate.py \
    --runs-dir "${RUNS_DIR}" \
    --labels-dir "${DATA_DIR}/labels" \
    --only "${E}" \
    --val-only \
    --split-file "${SPLIT_FILE}" \
    2>&1 | tee "${RUNS_DIR}/${E}/evaluate_val.txt"
done

echo "two-stage done: IoU ckpt runs/${EXP}/model_best.pth, height ckpt runs/${EXP}_purify/model_best.pth"
