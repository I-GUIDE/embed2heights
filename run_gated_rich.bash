#!/bin/bash
#SBATCH --job-name=emb2h_gated_rich
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# Stack the three gate upgrades on top of the single-scale gated_feature
# champion (uw_gated_F = 0.5072):
#
#   --gate-mode rich         : 2-layer MLP gate (GMU paper §5.1 nonlinearity)
#   --gate-untied            : independent G_AE and G_TES (multimodal GMU form)
#   --modality-dropout 0.15  : per-sample Tessera zeroing for regularization
#
# Everything else identical to uw_gated_F's recipe: seed 42, 30 epochs,
# no aug, plateau LR, presence_centered loss, base_ch=48, specialist_d=2.
# Promotion bar: ≥ 0.513 (uw_gated_F + 0.006), no axis dropping > 0.02.

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

echo "========================================"
echo "exp=gated_rich  Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

python train.py \
    --experiment-name gated_rich \
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
    --gate-mode rich \
    --gate-untied \
    --modality-dropout 0.15 \
    --no-uncertainty-weighting \
    --seed 42
