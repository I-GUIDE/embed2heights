#!/bin/bash
#SBATCH --job-name=emb2h_ctaSubmit
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# Build the leaderboard submission package by ensembling the 5
# cross-task-attention fold checkpoints (ctaskattn_fold0..4) on the
# test set. Mirrors run_submission_ensemble.bash but for the new
# recipe (gated_feature + cross-task-attention).
#
# Honest local estimate of leaderboard score (assuming the +0.0044
# fold-0 delta holds across folds):
#   gated_feature 5-fold mean: 0.4999 ± 0.0147
#   ctaskattn      expected:   ~0.504 (mean) + bagging gain of ~0.005
#                              → submission ≈ 0.51

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
TEST_DIR="/projects/bcrm/emb2height/data/test"

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

SUBMIT_BASE="${SCRIPT_DIR}/runs/submission"
mkdir -p "$SUBMIT_BASE"

# 1) Predict per fold on TEST data (label-free → submission filenames).
for FOLD in 0 1 2 3 4; do
    EXP="ctaskattn_fold${FOLD}"
    PRED_DIR="${SUBMIT_BASE}/preds_ctaskattn_fold${FOLD}"
    echo "============================================"
    echo "[PREDICT-TEST fold $FOLD]  out=$PRED_DIR"
    echo "============================================"
    python predict.py \
        --experiment-name "$EXP" \
        --model-type tessera_iou_fusion \
        --test-embeddings-dir "${TEST_DIR}/alphaearth_test_emb" \
        --secondary-test-embeddings-dir "${TEST_DIR}/tessera_test_emb" \
        --predictions-dir "$PRED_DIR"
done

# 2) Average the 5 fold predictions into the ensemble dir.
ENSEMBLE_DIR="${SUBMIT_BASE}/ctaskattn_5fold_ensemble"
echo "============================================"
echo "[ENSEMBLE]  in=preds_ctaskattn_fold{0..4}  out=$ENSEMBLE_DIR"
echo "============================================"
python tools/ensemble_predictions.py \
    --input-dirs ${SUBMIT_BASE}/preds_ctaskattn_fold0 \
                 ${SUBMIT_BASE}/preds_ctaskattn_fold1 \
                 ${SUBMIT_BASE}/preds_ctaskattn_fold2 \
                 ${SUBMIT_BASE}/preds_ctaskattn_fold3 \
                 ${SUBMIT_BASE}/preds_ctaskattn_fold4 \
    --output-dir "$ENSEMBLE_DIR" \
    --expected-count 946

# 3) Zip the ensemble dir for upload.
ZIP_PATH="${SUBMIT_BASE}/ctaskattn_5fold_ensemble.zip"
rm -f "$ZIP_PATH"
echo "============================================"
echo "[ZIP]  $ZIP_PATH"
echo "============================================"
( cd "$SUBMIT_BASE" && zip -r -q "ctaskattn_5fold_ensemble.zip" "ctaskattn_5fold_ensemble/" )
ls -lh "$ZIP_PATH"

# Quick sanity peek.
python - <<'PY'
import os, glob, numpy as np
SUBMIT="/u/dkiv2/group_dkiv2/active/embed2heights/runs/submission/ctaskattn_5fold_ensemble"
fns = sorted(glob.glob(os.path.join(SUBMIT, "*.npy")))
print(f"Files: {len(fns)}")
print(f"Example: {os.path.basename(fns[0])}")
a = np.load(fns[0])
print(f"  shape: {a.shape}  dtype: {a.dtype}")
for i, name in enumerate(("bld%", "veg%", "wat%", "h_m")):
    print(f"  ch{i} ({name}):  min={a[i].min():.3f}  max={a[i].max():.3f}  mean={a[i].mean():.3f}")
PY

echo ""
echo "Done: $(date)"
echo "Submission archive: $ZIP_PATH"
