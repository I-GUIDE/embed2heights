#!/bin/bash
#SBATCH --job-name=emb2h
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1

# Usage:
#   sbatch run_baselines.bash                          # run all 4 baselines
#   sbatch run_baselines.bash --only alphaearth        # run one baseline
#   sbatch run_baselines.bash --only alphaearth --epochs 2  # quick smoke test

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
DATA_DIR="/projects/bcrm/emb2height/data"
EXTRA_ARGS="$@"

# --- Conda setup ---
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "ERROR: conda not found." >&2
    exit 1
fi

__conda_setup="$("${CONDA_BASE}/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
unset __conda_setup

conda activate emb2heights

# --- Run ---
mkdir -p "${SCRIPT_DIR}/slurm_logs"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SCRIPT_DIR"

echo "========================================"
echo "embed2heights baselines"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Data: $DATA_DIR"
echo "Args: $EXTRA_ARGS"
echo "========================================"

python run_all_baselines.py \
    --data-dir "${DATA_DIR}/train" \
    --test-data-dir "${DATA_DIR}/test" \
    $EXTRA_ARGS

echo ""
echo "Done: $(date)"
