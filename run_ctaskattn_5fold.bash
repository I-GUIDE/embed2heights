#!/bin/bash
#SBATCH --job-name=emb2h_ctaFfold
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --array=1-4
# fold 0 is reused from ctaskattn_v1 (identical recipe). See README/log.

# 5-fold group-stratified training of the cross-task-attention recipe.
# Mirror of run_gated_F_5fold.bash, identical recipe except for one
# extra flag: --cross-task-attention.
#
# ctaskattn_v1 (single fold-0 run) scored 0.4810 vs gated_F_fold0's
# 0.4766 (+0.0044). This sweep confirms that delta across the 5 group
# folds and produces 5 checkpoints for ensemble submission.
#
# Each task trains on one fold; their per-fold scores form a 5-element
# bag whose mean is the right estimate of expected leaderboard score.

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
DATA_DIR="/projects/bcrm/emb2height/data/train"

if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
fi
__conda_setup="$("${CONDA_BASE}/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate emb2heights

cd "$SCRIPT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FOLD=$SLURM_ARRAY_TASK_ID
EXP_NAME="ctaskattn_fold${FOLD}"
SPLIT_FILE="${SCRIPT_DIR}/splits/group_code_5fold_seed42/fold_${FOLD}/split.json"

echo "========================================"
echo "fold=$FOLD  exp=$EXP_NAME"
echo "split=$SPLIT_FILE"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

python train.py \
    --experiment-name "$EXP_NAME" \
    --model-type tessera_iou_fusion \
    --train-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
    --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb" \
    --train-targets-dir "${DATA_DIR}/labels" \
    --split-file "$SPLIT_FILE" \
    --batch-size 32 \
    --patch-size 256 \
    --epochs 30 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --loss-preset presence_centered \
    --aux-weight 1.0 \
    --presence-tversky-weight 1.0 \
    --fraction-mae-weight 0.1 \
    --tessera-presence-ch 16 \
    --tessera-hidden-ch 96 \
    --tessera-hidden-depth 2 \
    --height-specialist-depth 2 \
    --lightunet-base-ch 48 \
    --height-loss-kind l1 \
    --huber-delta 1.0 \
    --build-height-boost 5.0 \
    --veg-height-boost 1.5 \
    --aux-veg-weight 1.0 \
    --iou-loss-kind tversky \
    --focal-gamma 2.0 \
    --focal-alpha 0.25 \
    --structure-weight 2.0 \
    --no-augment \
    --scheduler plateau \
    --compile \
    --fusion-mode gated_feature \
    --gate-mode simple \
    --no-gate-untied \
    --modality-dropout 0.0 \
    --cross-task-attention \
    --no-uncertainty-weighting \
    --seed 42
