#!/bin/bash
#SBATCH --job-name=emb2h_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1

# Runs predict.py on the val split + evaluate.py for each of the 4 ablation
# experiments, then prints the 5 leaderboard metrics side by side.

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
DATA_DIR="/projects/bcrm/emb2height/data/train"

if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
fi
__conda_setup="$("${CONDA_BASE}/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate emb2heights

cd "$SCRIPT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

EXPERIMENTS=(
    iou_fusion_v48_base
    iou_fusion_v48_aug
    iou_fusion_v48_attn
    iou_fusion_v48_attn_aug
)

for exp in "${EXPERIMENTS[@]}"; do
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
echo "[EVALUATE] all ablation runs, val split only"
echo "============================================"
python evaluate.py --only "${EXPERIMENTS[@]}" --val-only

echo ""
echo "Done: $(date)"
