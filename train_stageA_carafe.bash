#!/bin/bash
#SBATCH --job-name=emb2h_carafe
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --array=0          # single fold for a cheap A/B check; set 0-4 for all folds

# Stage-A ablation: uw_gated_F champion recipe with the ONLY change being the
# decoder upsampler (bilinear -> CARAFE). Everything else is byte-identical to
# train.bash, so any leaderboard movement is attributable to Stage A alone.
#
#   Baseline : uw_gated_F_fold0        (train.bash, --upsample-kind bilinear default)
#   This run : uw_gated_F_carafe_fold0 (only adds --upsample-kind carafe)
#
# Compare with: python evaluate.py over both runs' predictions, or the
# in-training val leaderboard score (topk_pool / training logs).
#
# To try DySample instead, change carafe -> dysample below (cheaper, ~0 params).

SCRIPT_DIR="/projects/bcrm/akhot2/embed2heights_max"
DATA_DIR="/projects/bcrm/emb2height/data/train"
SPLITS_ROOT="${SCRIPT_DIR}/splits/group_code_5fold_seed42"

cd "$SCRIPT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FOLD=$SLURM_ARRAY_TASK_ID
EXP="uw_gated_F_carafe_fold${FOLD}"
SPLIT="${SPLITS_ROOT}/fold_${FOLD}/split.json"

echo "========================================"
echo "Stage-A ablation (CARAFE) | fold=$FOLD  exp=$EXP"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
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
    --upsample-kind carafe \
    --height-loss-kind l1 \
    --huber-delta 1.0 \
    --build-height-boost 5.0 \
    --veg-height-boost 1.5 \
    --aux-veg-weight 1.0 \
    --iou-loss-kind tversky \
    --focal-gamma 2.0 \
    --focal-alpha 0.25 \
    --structure-weight 2.0 \
    --compile \
    --seed 42
