#!/bin/bash
#SBATCH --job-name=emb2h_quad_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# Predict + evaluate the 4 new runs against champion (uw_gated_F = 0.5072):
#   - uw_gated_F_s0 / uw_gated_F_s1  : seed-bag confirmation of the champion
#   - gated_rich                     : rich + untied + dropout knobs
#   - pyramid_gated                  : multi-scale GMU at 4 U-Net levels
# Promotion bar for arch upgrades: ≥ 0.513 (uw_gated_F + 0.006).

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

for exp in uw_gated_F_s0 uw_gated_F_s1 gated_rich pyramid_gated; do
    echo "============================================"
    echo "[PREDICT] $exp"
    echo "============================================"
    python predict.py \
        --experiment-name "$exp" \
        --model-type tessera_iou_fusion \
        --test-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
        --secondary-test-embeddings-dir "${DATA_DIR}/tessera_emb" \
        --test-targets-dir "${DATA_DIR}/labels"
done

echo "============================================"
echo "[EVALUATE] val split — including champion uw_gated_F for reference"
echo "============================================"
python evaluate.py \
    --only uw_gated_F uw_gated_F_s0 uw_gated_F_s1 gated_rich pyramid_gated \
    --val-only \
    --labels-dir "${DATA_DIR}/labels"

echo ""
echo "Done: $(date)"
