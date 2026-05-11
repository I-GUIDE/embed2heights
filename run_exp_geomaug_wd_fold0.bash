#!/bin/bash
#SBATCH --job-name=emb2h_gaugwd_f0
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=03:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gres=gpu:1
# Generalization-gap test: D4 aug + 10x weight decay stacked on smooth baseline.
# If both individual probes help, this should help more — diagnoses whether they
# attack the same overfit (additive ~= one) or orthogonal modes (additive > one).
EXP="exp_geomaug_wd_fold0"
EXTRA_TRAIN_ARGS=(
    --building-smooth-weight 0.5 --building-smooth-erode-px 2 --building-smooth-thr 0.5
    --geom-aug
    --weight-decay 1e-3
)
source /u/dkiv2/group_dkiv2/active/embed2heights/exp_runner.sh
