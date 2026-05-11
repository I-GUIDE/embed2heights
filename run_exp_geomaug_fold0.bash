#!/bin/bash
#SBATCH --job-name=emb2h_geomaug_f0
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=03:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gres=gpu:1
# Generalization-gap test: add D4 (h-flip/v-flip/transpose) augmentation on top
# of the smooth baseline. Hypothesis: closes train→test iou_bld gap on the held-
# out test set (LB is at 0.4417 vs val 0.50+).
EXP="exp_geomaug_fold0"
EXTRA_TRAIN_ARGS=(
    --building-smooth-weight 0.5 --building-smooth-erode-px 2 --building-smooth-thr 0.5
    --geom-aug
)
source /u/dkiv2/group_dkiv2/active/embed2heights/exp_runner.sh
