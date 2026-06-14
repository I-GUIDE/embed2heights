#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --job-name=2stage_screen
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu,gpu_a100

# Generalized 2-stage screen: sbatch screen_2stage.sbatch CONFIG_BASENAME [FOLD] [SEED]
# Runs stage1 (CONFIG) -> stage2 purify (warm-start, grad-scale 0, 20ep lr1.5e-4)
# -> predict+eval both checkpoints on the fold val split. One job per the
# slurm-cluster-practices rule.
set -euo pipefail
CFG=${1:?need config basename}
FOLD=${2:-0}
SEED=${3:-0}
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
CONFIG=${PROJECT_DIR}/configs/active/${CFG}.yml
EXP=${CFG}_s${SEED}_f${FOLD}
RUNS_DIR=${PROJECT_DIR}/runs
DATA_DIR=/u/dingqi2/workspace/esa/data/train
SPLIT_FILE=${PROJECT_DIR}/splits/group_code_5fold_seed42/fold_${FOLD}/split.json

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emb2heights
cd "${PROJECT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "runs/${EXP}"
python train.py --config "${CONFIG}" --experiment-name "${EXP}" \
  --split-file "${SPLIT_FILE}" --seed "${SEED}" 2>&1 | tee "runs/${EXP}/train.log"

mkdir -p "runs/${EXP}_purify"
python train.py --config "${CONFIG}" --experiment-name "${EXP}_purify" \
  --split-file "${SPLIT_FILE}" --seed "${SEED}" \
  --init-checkpoint "runs/${EXP}/model_best.pth" \
  --presence-trunk-grad-scale 0.0 --epochs 20 --lr 0.00015 \
  2>&1 | tee "runs/${EXP}_purify/train.log"

for E in "${EXP}" "${EXP}_purify"; do
  python predict.py --experiment-name "${E}" --base-dir "${RUNS_DIR}" \
    --test-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
    --secondary-test-embeddings-dir "${DATA_DIR}/tessera_emb" \
    --token-test-embeddings-dir "${DATA_DIR}/terramind_s1_emb" \
    --secondary-token-test-embeddings-dir "${DATA_DIR}/terramind_s2_emb" \
    --third-token-test-embeddings-dir "${DATA_DIR}/thor_s1_emb" \
    --fourth-token-test-embeddings-dir "${DATA_DIR}/thor_s2_emb" \
    --test-targets-dir "${DATA_DIR}/labels" --patch-size 256 --max-samples 0 \
    2>&1 | tee "${RUNS_DIR}/${E}/predict_val.log"
  python evaluate.py --runs-dir "${RUNS_DIR}" --labels-dir "${DATA_DIR}/labels" \
    --only "${E}" --val-only --split-file "${SPLIT_FILE}" \
    2>&1 | tee "${RUNS_DIR}/${E}/evaluate_val.txt"
done
echo "screen ${EXP} done"
