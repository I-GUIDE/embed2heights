#!/bin/bash
#SBATCH --job-name=emb2h_wd1e3_f0
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=03:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gres=gpu:1
# Generalization-gap test: 10x weight decay (1e-4 → 1e-3) on smooth baseline.
# Pure regularization probe — if buildings overfit training distribution,
# stronger L2 should narrow train→test gap.
EXP="exp_wd1e3_fold0"
EXTRA_TRAIN_ARGS=(
    --building-smooth-weight 0.5 --building-smooth-erode-px 2 --building-smooth-thr 0.5
    --weight-decay 1e-3
)
source /u/dkiv2/group_dkiv2/active/embed2heights/exp_runner.sh
