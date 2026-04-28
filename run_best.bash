#!/bin/bash
#SBATCH --job-name=emb2h_best
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1

# Runs the tessera_iou_fusion config from logs/current_best_training_params.json
# on this repo's local data tree. Pass extra flags after the script name.

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
DATA_DIR="/projects/bcrm/emb2height/data/train"
EXTRA_ARGS="$@"

if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "ERROR: conda not found." >&2
    exit 1
fi

__conda_setup="$("${CONDA_BASE}/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
unset __conda_setup
conda activate emb2heights

mkdir -p "${SCRIPT_DIR}/slurm_logs"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SCRIPT_DIR"

echo "========================================"
echo "emb2heights BEST config run"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Data: $DATA_DIR"
echo "Args: $EXTRA_ARGS"
echo "========================================"

python train.py \
    --experiment-name alphaearth_tessera_iou_fusion_N_base48 \
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
    --seed 42 \
    $EXTRA_ARGS

echo ""
echo "Done: $(date)"
