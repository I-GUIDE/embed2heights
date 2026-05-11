#!/bin/bash
#SBATCH --job-name=emb2h_pfsweep
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu,gpu,gpu_a100
# Per-fold sweep on gated_F_fold0..4 to expose iou_bld variance across folds.
# Writes one log per fold under runs/submission/_sweep/gated_F_perfold_fold{N}.log.

set -e

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
LABELS="/projects/bcrm/emb2height/data/train/labels"
SPLITS_BASE="${SCRIPT_DIR}/splits/group_code_5fold_seed42"
OUT_BASE="${SCRIPT_DIR}/runs/submission/_sweep"

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
mkdir -p "$OUT_BASE"
echo "Node: $(hostname)"

for f in 0 1 2 3 4; do
    PRED_DIR="${SCRIPT_DIR}/runs/gated_F_fold${f}/predictions"
    SPLIT="${SPLITS_BASE}/fold_${f}/split.json"
    OUT_LOG="${OUT_BASE}/gated_F_perfold_fold${f}.log"
    echo "=== gated_F fold ${f} ==="
    python tools/sweep_thresholds.py \
        --pred-dir "$PRED_DIR" \
        --labels-dir "$LABELS" \
        --split-file "$SPLIT" \
        | tee "$OUT_LOG"
    echo
done

echo "Done: $(date)"
