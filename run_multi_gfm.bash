#!/bin/bash
#SBATCH --job-name=emb2h_multi
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# 4-modality GMU: AE + Tessera + TerraMind(S1+S2) + THOR(S1+S2).
#
# Why this run: AE+Tessera fusion (uw_gated_F = 0.5072 random val,
# 0.4999 ± 0.0147 group-fold mean) saturated knob-tuning. The actual
# missing capacity is the other two GeoFMs sitting unused on disk.
# This activates them via the k=4 multimodal-GMU form (Arevalo §3.1
# Figure 2a) with the same zero-init residual warm-start trick: AE
# gate opens at init, the other 3 gates start closed, model is
# bit-equivalent to AE-only at t=0 and learns each new modality as
# a residual contribution.
#
# Modern stability fix on top of vanilla GMU: per-modality GroupNorm
# before the gate concat input. Prevents whichever stream has the
# largest feature magnitudes from dominating the gate logit even
# when the gate weights are well-calibrated.
#
# Recipe pinned to uw_gated_F's known-good knobs (presence_centered,
# base_ch=48, specialist_d=2, no aug, plateau, seed 42).

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
echo "exp=multi_gfm_v1  Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

python train.py \
    --experiment-name multi_gfm_v1 \
    --model-type multi_gfm \
    --train-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
    --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb" \
    --terramind-s1-train-emb-dir "${DATA_DIR}/terramind_s1_emb" \
    --terramind-s2-train-emb-dir "${DATA_DIR}/terramind_s2_emb" \
    --thor-s1-train-emb-dir "${DATA_DIR}/thor_s1_emb" \
    --thor-s2-train-emb-dir "${DATA_DIR}/thor_s2_emb" \
    --train-targets-dir "${DATA_DIR}/labels" \
    --split-file "${SCRIPT_DIR}/splits/group_code_5fold_seed42/fold_0/split.json" \
    --batch-size 32 \
    --patch-size 256 \
    --epochs 30 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --loss-preset presence_centered \
    --aux-weight 1.0 \
    --presence-tversky-weight 1.0 \
    --fraction-mae-weight 0.1 \
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
    --modality-dropout 0.15 \
    --no-uncertainty-weighting \
    --seed 42
