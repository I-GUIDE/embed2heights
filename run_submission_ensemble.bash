#!/bin/bash
#SBATCH --job-name=emb2h_submit
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# Build the leaderboard submission package by ensembling the 5 group-fold
# gated_feature models on the test set.
#
# Steps:
#   1. predict.py for each fold model in label-free mode on the test set
#      (filenames keep the year suffix, as the leaderboard expects).
#   2. Average the 5 sets of predictions per file → ensemble dir.
#   3. Zip ensemble dir to runs/submission/gated_F_5fold_ensemble.zip.
#
# Honest local estimate of leaderboard score for this ensemble:
#   5-fold mean = 0.4999 ± 0.0147 → with bagging variance reduction
#   expect ~0.50–0.51 on submission.

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
    EXP="gated_F_fold${FOLD}"
    PRED_DIR="${SUBMIT_BASE}/preds_fold${FOLD}"
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
ENSEMBLE_DIR="${SUBMIT_BASE}/gated_F_5fold_ensemble"
echo "============================================"
echo "[ENSEMBLE]  in=preds_fold{0..4}  out=$ENSEMBLE_DIR"
echo "============================================"
python tools/ensemble_predictions.py \
    --input-dirs ${SUBMIT_BASE}/preds_fold0 \
                 ${SUBMIT_BASE}/preds_fold1 \
                 ${SUBMIT_BASE}/preds_fold2 \
                 ${SUBMIT_BASE}/preds_fold3 \
                 ${SUBMIT_BASE}/preds_fold4 \
    --output-dir "$ENSEMBLE_DIR" \
    --expected-count 946

# 3) Zip the ensemble dir for upload.
ZIP_PATH="${SUBMIT_BASE}/gated_F_5fold_ensemble.zip"
rm -f "$ZIP_PATH"
echo "============================================"
echo "[ZIP]  $ZIP_PATH"
echo "============================================"
( cd "$SUBMIT_BASE" && zip -r -q "gated_F_5fold_ensemble.zip" "gated_F_5fold_ensemble/" )
ls -lh "$ZIP_PATH"

# Quick sanity: peek at one prediction's shape and stats.
python - <<'PY'
import os, glob, numpy as np
SUBMIT="/u/dkiv2/group_dkiv2/active/embed2heights/runs/submission/gated_F_5fold_ensemble"
fns = sorted(glob.glob(os.path.join(SUBMIT, "*.npy")))
print(f"Files in ensemble dir: {len(fns)}")
print(f"Example: {os.path.basename(fns[0])}")
a = np.load(fns[0])
print(f"  shape: {a.shape}  dtype: {a.dtype}")
print(f"  ch0 (bld%):  min={a[0].min():.3f}  max={a[0].max():.3f}  mean={a[0].mean():.3f}")
print(f"  ch1 (veg%):  min={a[1].min():.3f}  max={a[1].max():.3f}  mean={a[1].mean():.3f}")
print(f"  ch2 (wat%):  min={a[2].min():.3f}  max={a[2].max():.3f}  mean={a[2].mean():.3f}")
print(f"  ch3 (h_m):   min={a[3].min():.3f}  max={a[3].max():.3f}  mean={a[3].mean():.3f}")
PY

echo ""
echo "Done: $(date)"
echo "Submission archive: $ZIP_PATH"
