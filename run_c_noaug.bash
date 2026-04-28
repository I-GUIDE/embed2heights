#!/bin/bash
#SBATCH --job-name=emb2h_c_noaug
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1

# Decisive comparison: is the modernization bundle (EMA + cosine +
# aux-Tversky) worth anything on top of the no-aug champion recipe?
#
# B beat A by +0.0035 (inside noise) but both underperformed champion N
# (0.4644) by ~0.028, because both had augmentation on and N didn't.
# This run takes B's bundle and strips augmentation — head-to-head vs N.
#
# Promotion rule: val score ≥ 0.4644 + 0.006 (= 0.4704) with no axis
# dropping > 0.02. Otherwise modernization bundle is closed.

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

python train.py \
    --experiment-name ab_C_noaug_modernize \
    --model-type tessera_iou_fusion \
    --train-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
    --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb" \
    --train-targets-dir "${DATA_DIR}/labels" \
    --split-file "${SCRIPT_DIR}/splits/split.json" \
    --batch-size 32 \
    --patch-size 256 \
    --epochs 60 \
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
    --no-augment \
    --scheduler cosine \
    --ema --ema-decay 0.9995 \
    --aux-tversky-weight 0.5
