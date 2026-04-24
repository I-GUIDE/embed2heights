#!/bin/bash
#SBATCH --job-name=emb2h_ab
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-1

# A/B test: does the modernization bundle (aux-Tversky + flip_rot180 aug +
# EMA + cosine + longer epochs) beat the current recipe?
#
#   A (task_id=0, control):   30 ep, d4 aug,         plateau, no ema, aux_tversky=0
#   B (task_id=1, treatment): 60 ep, flip_rot180,    cosine,  ema,    aux_tversky=0.5
#
# Submit with:  sbatch run_ab_modernize.bash
# Each array task gets its own H100. A100 fallback: change partition to
# gpu_a100 and gres to gpu:a100:1.

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
DATA_DIR="${SCRIPT_DIR}/tools/data/embed2heights/data/train"

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

# Shared config — matches runs/iou_fusion_v48_base (the current best-known recipe).
COMMON_ARGS=(
    --model-type tessera_iou_fusion
    --train-embeddings-dir "${DATA_DIR}/alphaearth_emb"
    --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb"
    --train-targets-dir "${DATA_DIR}/labels"
    --split-file "${SCRIPT_DIR}/splits/split.json"
    --batch-size 32
    --patch-size 256
    --lr 2e-4
    --weight-decay 1e-4
    --loss-preset presence_centered
    --aux-weight 1.0
    --presence-tversky-weight 1.0
    --fraction-mae-weight 0.1
    --tessera-presence-ch 16
    --tessera-hidden-ch 96
    --tessera-hidden-depth 2
    --height-specialist-depth 2
    --lightunet-base-ch 48
    --height-loss-kind l1
    --huber-delta 1.0
    --build-height-boost 5.0
    --veg-height-boost 1.5
    --aux-veg-weight 1.0
    --iou-loss-kind tversky
    --focal-gamma 2.0
    --focal-alpha 0.25
    --structure-weight 2.0
    --seed 42
)

case $SLURM_ARRAY_TASK_ID in
    0)
        EXP_NAME="ab_A_control"
        EXTRA_ARGS=(
            --epochs 30
            --augment --augment-mode d4
            --scheduler plateau
            --no-ema
            --aux-tversky-weight 0.0
        )
        ;;
    1)
        EXP_NAME="ab_B_modernize"
        EXTRA_ARGS=(
            --epochs 60
            --augment --augment-mode flip_rot180
            --scheduler cosine
            --ema --ema-decay 0.9995
            --aux-tversky-weight 0.5
        )
        ;;
    *)
        echo "Unknown array task id: $SLURM_ARRAY_TASK_ID" >&2
        exit 1
        ;;
esac

echo "========================================"
echo "A/B task=$SLURM_ARRAY_TASK_ID  exp=$EXP_NAME"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Extra: ${EXTRA_ARGS[@]}"
echo "========================================"

python train.py \
    --experiment-name "$EXP_NAME" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
