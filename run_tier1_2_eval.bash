#!/bin/bash
#SBATCH --job-name=emb2h_t12_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# Predict + evaluate the Tier 1+2 fold-0 experiments against the
# gated_feature champion (gated_F_fold0 = 0.4766).
# Promotion bar: ≥ 0.4826 (champion + 0.006), no axis dropping > 0.02.

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
DATA_DIR="/projects/bcrm/emb2height/data/train"
SPLIT="${SCRIPT_DIR}/splits/group_code_5fold_seed42/fold_0/split.json"

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

# multi_gfm_v1 + hierarchical_v1 use the multi_gfm dataset (4 GFMs).
for EXP in multi_gfm_v1 hierarchical_v1; do
    echo "============================================"
    echo "[PREDICT] $EXP"
    echo "============================================"
    python predict.py \
        --experiment-name "$EXP" \
        --model-type multi_gfm \
        --test-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
        --secondary-test-embeddings-dir "${DATA_DIR}/tessera_emb" \
        --terramind-s1-test-emb-dir "${DATA_DIR}/terramind_s1_emb" \
        --terramind-s2-test-emb-dir "${DATA_DIR}/terramind_s2_emb" \
        --thor-s1-test-emb-dir "${DATA_DIR}/thor_s1_emb" \
        --thor-s2-test-emb-dir "${DATA_DIR}/thor_s2_emb" \
        --test-targets-dir "${DATA_DIR}/labels"
done

# dirichlet_v1 + ctaskattn_v1 use the 2-modality champion path.
for EXP in dirichlet_v1 ctaskattn_v1; do
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

# Evaluate all four against fold-0's val groups.
echo "============================================"
echo "[EVALUATE fold 0] split=$SPLIT"
echo "============================================"
python evaluate.py \
    --only multi_gfm_v1 hierarchical_v1 dirichlet_v1 ctaskattn_v1 gated_F_fold0 \
    --val-only \
    --split-file "$SPLIT" \
    --labels-dir "${DATA_DIR}/labels"

echo ""
echo "Done: $(date)"
