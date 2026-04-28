#!/bin/bash
#SBATCH --job-name=emb2h_gFfold_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# Predict + evaluate the 5 group-fold models on their respective val
# subsets. Each fold's val groups are disjoint from training groups
# (no spatial leakage). The 5 per-fold scores form an honest bag for
# expected leaderboard score.

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

# Predict once per fold model on the full embedding set (predict.py
# doesn't filter by split — evaluate.py applies the val mask).
for FOLD in 0 1 2 3 4; do
    EXP="gated_F_fold${FOLD}"
    echo "============================================"
    echo "[PREDICT] $EXP"
    echo "============================================"
    python predict.py \
        --experiment-name "$EXP" \
        --model-type tessera_iou_fusion \
        --test-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
        --secondary-test-embeddings-dir "${DATA_DIR}/tessera_emb" \
        --test-targets-dir "${DATA_DIR}/labels"
done

# Evaluate each fold against ITS OWN val groups (the disjoint set).
# We can't use a single --split-file across all 5 — each needs its own.
for FOLD in 0 1 2 3 4; do
    EXP="gated_F_fold${FOLD}"
    SPLIT="${SCRIPT_DIR}/splits/group_code_5fold_seed42/fold_${FOLD}/split.json"
    echo "============================================"
    echo "[EVALUATE fold $FOLD] $EXP  split=$SPLIT"
    echo "============================================"
    python evaluate.py \
        --only "$EXP" \
        --val-only \
        --split-file "$SPLIT" \
        --labels-dir "${DATA_DIR}/labels"
done

echo ""
echo "Done: $(date)"
