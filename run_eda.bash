#!/bin/bash
#SBATCH --job-name=emb2h_eda
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu

# Usage:
#   sbatch run_eda.bash                                   # full pipeline
#   sbatch run_eda.bash --only labels embeddings_alphaearth
#   sbatch run_eda.bash --sample 100                      # smoke test
#   sbatch run_eda.bash --overwrite

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd "$SCRIPT_DIR"

echo "========================================"
echo "embed2heights EDA"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?}  Mem: ${SLURM_MEM_PER_NODE:-?}"
echo "Args: $EXTRA_ARGS"
echo "========================================"

python tools/run_eda.py $EXTRA_ARGS

echo ""
echo "Done: $(date)"
