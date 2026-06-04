#!/bin/bash
#SBATCH --job-name=emb2h_archD
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --array=0          # single fold for the 4-way comparison; set 0-4 for all folds
#
# SoTA-shortcut architecture experiment, layered on the uw_gated_F champion
# (ae_tessera_gated, presence_centered, no-aug). Differences vs train.bash:
#
#   Stage A  --upsample-kind dysample            sharper decoder boundaries
#                                              -> building IoU + edge height RMSE
#   Stage C  --height-head-kind softbin        ordinal/binned height regression
#            --height-bin-aux-weight 0.5        bin-CE aux forces bin commitment
#            --height-loss-kind pinball         quantile height loss
#            --pinball-tau 0.75                 penalize under-prediction (DSM)
#   Stage B  --height-independent-branches      decouple base/build/veg heights
#            --uncertainty-weighting            auto-balance presence vs height
#
# Ablate one stage at a time to attribute leaderboard movement. Start from
# Stage A alone (lowest risk), then add C, then B.

SCRIPT_DIR="/u/wz53/emb2height_warehouse/embed2heights_max"
DATA_DIR="/projects/bcrm/emb2height/data/train"
SPLITS_ROOT="${SCRIPT_DIR}/splits/group_code_5fold_seed42"

cd "$SCRIPT_DIR"
source /u/wz53/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FOLD=$SLURM_ARRAY_TASK_ID
SPLIT="${SPLITS_ROOT}/fold_${FOLD}/split.json"
EXP="arch_dysample_softbin_pinball_fold${FOLD}"

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
    --gate-mode simple \
    --upsample-kind dysample \
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
    --structure-weight 2.0 \
    --compile \
    --seed 42
