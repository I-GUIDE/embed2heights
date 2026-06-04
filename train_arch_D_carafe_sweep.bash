#!/bin/bash
#SBATCH --job-name=emb2h_Dcsw
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --array=0-8          # 3 LR x 3 boundary-weight grid, fold 0
#
# Hyperparameter sweep on the best config (A+C+B+D CARAFE, "archDc"), fold 0,
# 30 epochs (enough for relative ranking). Sweeps LR x building-boundary-weight;
# pinball_tau fixed at 0.75. Pick the best EXP by val_weighted_score, then run
# train_arch_D_carafe_5fold.bash with those values for the final 5-fold model.

set -euo pipefail

SCRIPT_DIR="/u/wz53/emb2height_warehouse/embed2heights_max"
DATA_DIR="/projects/bcrm/emb2height/data/train"
SPLITS_ROOT="${SCRIPT_DIR}/splits/group_code_5fold_seed42"

cd "$SCRIPT_DIR"
source /u/wz53/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Grid: index = 3*lr_i + bw_i
LRS=(1e-4 2e-4 3e-4)
BWS=(0.3 0.5 1.0)
I=$SLURM_ARRAY_TASK_ID
LR=${LRS[$((I / 3))]}
BW=${BWS[$((I % 3))]}

FOLD=0
SPLIT="${SPLITS_ROOT}/fold_${FOLD}/split.json"
EXP="sweep_archDc_lr${LR}_bw${BW}_fold0"

echo "========================================"
echo "Sweep idx=$I | LR=$LR | boundary_weight=$BW | exp=$EXP"
echo "========================================"

python train.py \
    --experiment-name "$EXP" \
    --model-type ae_tessera_gated \
    --train-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
    --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb" \
    --train-targets-dir "${DATA_DIR}/labels" \
    --split-file "$SPLIT" \
    --batch-size 32 \
    --patch-size 256 \
    --epochs 30 \
    --lr "$LR" \
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
    --gate-mode simple \
    --upsample-kind carafe \
    --height-head-kind softbin \
    --height-n-bins 64 \
    --height-bin-max-m 80.0 \
    --height-bin-aux-weight 0.5 \
    --height-loss-kind pinball \
    --pinball-tau 0.75 \
    --build-height-boost 5.0 \
    --veg-height-boost 1.5 \
    --aux-veg-weight 1.0 \
    --height-independent-branches \
    --uncertainty-weighting \
    --building-boundary-weight "$BW" \
    --structure-weight 2.0 \
    --compile \
    --seed 42
