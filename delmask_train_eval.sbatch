#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --job-name=covgt10_delmask
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --partition=gpu,gpu_a100
set -euo pipefail
# Loss-masking A/B: stage-1 covgt10 with presence/seg loss DROPPED on v2 missing-building
# regions (suspected human-deleted footprints). Baseline (no mask) fold0 seed0
# cov>0.10 bld IoU = 0.4795. Compare runs/<EXP>/evaluate_cov0p10.txt.
FOLD=${1:-0}; SEED=${2:-0}
CFG=xf095_2stage_softbin_covgt10_delmask
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
CONFIG=${PROJECT_DIR}/configs/active/${CFG}.yml
EXP=${CFG}_s${SEED}_f${FOLD}
RUNS_DIR=${PROJECT_DIR}/runs; DATA_DIR=/u/dingqi2/workspace/esa/data/train
SPLIT_FILE=${PROJECT_DIR}/splits/group_code_5fold_seed42/fold_${FOLD}/split.json
source ~/miniconda3/etc/profile.d/conda.sh; conda activate emb2heights
cd "${PROJECT_DIR}"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "EXP=${EXP} (fold ${FOLD}, seed ${SEED}) mask=runs/missing_masks"
mkdir -p "runs/${EXP}"
python train.py --config "${CONFIG}" --experiment-name "${EXP}" --split-file "${SPLIT_FILE}" --seed "${SEED}" 2>&1 | tee "runs/${EXP}/train.log"
python predict.py --experiment-name "${EXP}" --base-dir "${RUNS_DIR}" \
  --test-embeddings-dir "${DATA_DIR}/alphaearth_emb" --secondary-test-embeddings-dir "${DATA_DIR}/tessera_emb" \
  --token-test-embeddings-dir "${DATA_DIR}/terramind_s1_emb" --secondary-token-test-embeddings-dir "${DATA_DIR}/terramind_s2_emb" \
  --third-token-test-embeddings-dir "${DATA_DIR}/thor_s1_emb" --fourth-token-test-embeddings-dir "${DATA_DIR}/thor_s2_emb" \
  --test-targets-dir "${DATA_DIR}/labels" --patch-size 256 --max-samples 0 2>&1 | tee "${RUNS_DIR}/${EXP}/predict_val.log"
python evaluate.py --runs-dir "${RUNS_DIR}" --labels-dir "${DATA_DIR}/labels" --only "${EXP}" --val-only --split-file "${SPLIT_FILE}" 2>&1 | tee "${RUNS_DIR}/${EXP}/evaluate_val.txt"
python cov0p10_eval.py "${EXP}" "${FOLD}" 2>&1 | tee "${RUNS_DIR}/${EXP}/evaluate_cov0p10.txt"
echo "delmask ${EXP} done"
