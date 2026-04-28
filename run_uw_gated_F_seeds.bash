#!/bin/bash
#SBATCH --job-name=emb2h_uwgF_seeds
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-1

# Multi-seed validation of the new champion recipe (uw_gated_F, val 0.5072,
# +0.0428 over N at seed 42). Confirms the breakthrough isn't a seed lottery.
#
#   0  uw_gated_F_s0 : seed 0
#   1  uw_gated_F_s1 : seed 1
#
# Combined with the original uw_gated_F (seed 42), this gives a 3-seed
# bag for the new recipe. Mean ± std across the three seeds is the right
# headline number to report.
#
# All other knobs locked to the recipe that produced 0.5072 — no aug,
# presence_centered, base_ch=48, specialist_d=2, gated_feature fusion.

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

case $SLURM_ARRAY_TASK_ID in
    0) EXP_NAME="uw_gated_F_s0"; SEED=0 ;;
    1) EXP_NAME="uw_gated_F_s1"; SEED=1 ;;
    *) echo "Unknown array task id: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "========================================"
echo "task=$SLURM_ARRAY_TASK_ID  exp=$EXP_NAME  seed=$SEED"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

python train.py \
    --experiment-name "$EXP_NAME" \
    --model-type tessera_iou_fusion \
    --train-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
    --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb" \
    --train-targets-dir "${DATA_DIR}/labels" \
    --split-file "${SCRIPT_DIR}/splits/split.json" \
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
    --no-uncertainty-weighting \
    --seed $SEED
