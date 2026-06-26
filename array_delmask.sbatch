#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --job-name=delmask
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --partition=gpu,gpu_a100
#SBATCH --array=0-13
# Full 5fold x 3seed delmask sweep MINUS f0s0 (done as job 115333). No %limit:
# launch all at once. I/O safety = stage the 105GB dataset to NODE-LOCAL NVMe (/tmp,
# 3.5TB) ONCE per node (flock-guarded), then train/predict from local disk so the
# 80-epoch run never touches the shared network FS (no cross-job I/O contention).
set -euo pipefail
CFG=xf095_2stage_softbin_covgt10_delmask
COMBOS=("1 0" "2 0" "3 0" "4 0" "0 1" "1 1" "2 1" "3 1" "4 1" "0 2" "1 2" "2 2" "3 2" "4 2")
read FOLD SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
CONFIG=${PROJECT_DIR}/configs/active/${CFG}.yml
EXP=${CFG}_s${SEED}_f${FOLD}
RUNS_DIR=${PROJECT_DIR}/runs
SRC=/u/dingqi2/workspace/esa/data/train
SPLIT_FILE=${PROJECT_DIR}/splits/group_code_5fold_seed42/fold_${FOLD}/split.json
source ~/miniconda3/etc/profile.d/conda.sh; conda activate emb2heights
cd "${PROJECT_DIR}"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ---- stage dataset to node-local NVMe (shared across co-located jobs via flock) ----
STAGE=/tmp/esa_train_data
mkdir -p "${STAGE}"
exec 200>/tmp/esa_stage.lock
echo "[stage] $(date) waiting for stage lock on $(hostname)"
flock -x 200
if [ ! -f "${STAGE}/.complete" ]; then
  echo "[stage] $(date) copying 105GB dataset to ${STAGE} on $(hostname)"
  for d in alphaearth_emb tessera_emb terramind_s1_emb terramind_s2_emb thor_s1_emb thor_s2_emb labels; do
    rsync -a --no-compress "${SRC}/${d}" "${STAGE}/"
  done
  touch "${STAGE}/.complete"
  echo "[stage] $(date) staging complete"
else
  echo "[stage] $(date) reuse existing node-local copy"
fi
flock -u 200
AE=${STAGE}/alphaearth_emb; TE=${STAGE}/tessera_emb
T1=${STAGE}/terramind_s1_emb; T2=${STAGE}/terramind_s2_emb
H1=${STAGE}/thor_s1_emb; H2=${STAGE}/thor_s2_emb; LB=${STAGE}/labels

echo "array task ${SLURM_ARRAY_TASK_ID} -> EXP=${EXP} (fold ${FOLD}, seed ${SEED}) local-data=${STAGE}"
mkdir -p "runs/${EXP}"
# ---- train (reads from node-local NVMe) ----
python train.py --config "${CONFIG}" --experiment-name "${EXP}" --split-file "${SPLIT_FILE}" --seed "${SEED}" \
  --train-embeddings-dir "${AE}" --secondary-train-embeddings-dir "${TE}" \
  --token-train-embeddings-dir "${T1}" --secondary-token-train-embeddings-dir "${T2}" \
  --third-token-train-embeddings-dir "${H1}" --fourth-token-train-embeddings-dir "${H2}" \
  --train-targets-dir "${LB}" 2>&1 | tee "runs/${EXP}/train.log"
# ---- predict val + eval (also from node-local) ----
python predict.py --experiment-name "${EXP}" --base-dir "${RUNS_DIR}" \
  --test-embeddings-dir "${AE}" --secondary-test-embeddings-dir "${TE}" \
  --token-test-embeddings-dir "${T1}" --secondary-token-test-embeddings-dir "${T2}" \
  --third-token-test-embeddings-dir "${H1}" --fourth-token-test-embeddings-dir "${H2}" \
  --test-targets-dir "${LB}" --patch-size 256 --max-samples 0 2>&1 | tee "${RUNS_DIR}/${EXP}/predict_val.log"
python evaluate.py --runs-dir "${RUNS_DIR}" --labels-dir "${SRC}/labels" --only "${EXP}" --val-only --split-file "${SPLIT_FILE}" 2>&1 | tee "${RUNS_DIR}/${EXP}/evaluate_val.txt"
python cov0p10_eval.py "${EXP}" "${FOLD}" 2>&1 | tee "${RUNS_DIR}/${EXP}/evaluate_cov0p10.txt"
echo "delmask array ${EXP} done"
