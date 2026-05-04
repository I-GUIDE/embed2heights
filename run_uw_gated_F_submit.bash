#!/bin/bash
#SBATCH --job-name=emb2h_uwgFsub
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gres=gpu:1

# Build a binarized leaderboard submission from the 5-fold uw_gated_F ensemble.
#
# Steps:
#   1. Predict each fold on TEST (label-free, submission filenames).
#   2. Predict each fold on VAL (paired mode, for threshold sweep).
#   3. Ensemble test predictions → continuous zip.
#   4. Sweep thresholds on val ensemble → optimal (t_bld, t_veg, t_wat).
#   5. Binarize test ensemble at swept thresholds → binarized dir + zip (~220 MB).
#
# Expects: runs/uw_gated_F_fold{0..4}/model_best.pth to exist.
# Produces: runs/submission/uw_gated_F_5fold_ensemble_bin.zip

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
DATA_DIR="/projects/bcrm/emb2height/data/train"
TEST_DIR="/projects/bcrm/emb2height/data/test"
SPLITS_ROOT="${SCRIPT_DIR}/splits/group_code_5fold_seed42"
SUBMIT_BASE="${SCRIPT_DIR}/runs/submission"

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
mkdir -p "$SUBMIT_BASE/_sweep" "$SUBMIT_BASE"

# ── 1. Predict test set (label-free) per fold ──────────────────────────────
for FOLD in 0 1 2 3 4; do
    EXP="uw_gated_F_fold${FOLD}"
    PRED_DIR="${SUBMIT_BASE}/preds_uwgF_fold${FOLD}"
    echo "=== [TEST fold $FOLD] ==="
    python predict.py \
        --experiment-name "$EXP" \
        --test-embeddings-dir "${TEST_DIR}/alphaearth_test_emb" \
        --secondary-test-embeddings-dir "${TEST_DIR}/tessera_test_emb" \
        --predictions-dir "$PRED_DIR"
done

# ── 2. Ensemble test predictions ───────────────────────────────────────────
ENSEMBLE_DIR="${SUBMIT_BASE}/uw_gated_F_5fold_ensemble"
echo "=== [ENSEMBLE test] ==="
python tools/ensemble.py mean \
    --inputs \
        "${SUBMIT_BASE}/preds_uwgF_fold0" \
        "${SUBMIT_BASE}/preds_uwgF_fold1" \
        "${SUBMIT_BASE}/preds_uwgF_fold2" \
        "${SUBMIT_BASE}/preds_uwgF_fold3" \
        "${SUBMIT_BASE}/preds_uwgF_fold4" \
    --output-dir "$ENSEMBLE_DIR"

# ── 3. Predict val (OOF, paired) per fold for threshold sweep ──────────────
for FOLD in 0 1 2 3 4; do
    EXP="uw_gated_F_fold${FOLD}"
    VAL_PRED_DIR="${SCRIPT_DIR}/runs/${EXP}/predictions"
    if [ -d "$VAL_PRED_DIR" ] && [ "$(ls -A $VAL_PRED_DIR)" ]; then
        echo "=== [VAL fold $FOLD] skipping — predictions already exist ==="
    else
        echo "=== [VAL fold $FOLD] predicting paired ==="
        python predict.py \
            --experiment-name "$EXP" \
            --test-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
            --secondary-test-embeddings-dir "${DATA_DIR}/tessera_emb" \
            --test-targets-dir "${DATA_DIR}/labels"
    fi
done

# ── 4. Sweep thresholds (per-fold OOF, then average) ──────────────────────
echo "=== [SWEEP thresholds] ==="
python - <<'PY'
import subprocess, sys, re, numpy as np
from pathlib import Path

SCRIPT_DIR = Path("/u/dkiv2/group_dkiv2/active/embed2heights")
LABELS_DIR = Path("/projects/bcrm/emb2height/data/train/labels")
SPLITS_ROOT = SCRIPT_DIR / "splits/group_code_5fold_seed42"
SWEEP_DIR = SCRIPT_DIR / "runs/submission/_sweep"

triples = []
for fold in range(5):
    pred_dir = SCRIPT_DIR / f"runs/uw_gated_F_fold{fold}/predictions"
    split_file = SPLITS_ROOT / f"fold_{fold}/split.json"
    log_path = SWEEP_DIR / f"uwgF_fold{fold}.log"
    print(f"[fold {fold}] sweeping {pred_dir} ...", flush=True)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "tools/sweep_thresholds.py"),
        "--pred-dir", str(pred_dir),
        "--labels-dir", str(LABELS_DIR),
        "--split-file", str(split_file),
    ]
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"  WARN: sweep failed for fold {fold}, check {log_path}", flush=True)
        continue
    text = log_path.read_text()
    m = re.search(r"per-class \(([0-9.]+),([0-9.]+),([0-9.]+)\)", text)
    if m:
        t = tuple(float(x) for x in m.groups())
        triples.append(t)
        print(f"  → ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})", flush=True)
    else:
        print(f"  WARN: could not parse threshold from {log_path}", flush=True)

if not triples:
    print("ERROR: no threshold triples found — using (0.5, 0.5, 0.5)", flush=True)
    triples = [(0.5, 0.5, 0.5)]

avg = tuple(float(np.mean([t[i] for t in triples])) for i in range(3))
print(f"\nAveraged optimal thresholds: (bld={avg[0]:.3f}, veg={avg[1]:.3f}, wat={avg[2]:.3f})", flush=True)

# Write for the shell to pick up
(SWEEP_DIR / "uwgF_avg_thresholds.txt").write_text(f"{avg[0]:.3f} {avg[1]:.3f} {avg[2]:.3f}\n")
PY

read T_BLD T_VEG T_WAT < "${SUBMIT_BASE}/_sweep/uwgF_avg_thresholds.txt"
echo "Using thresholds: bld=$T_BLD  veg=$T_VEG  wat=$T_WAT"

# ── 5. Binarize test ensemble + zip ────────────────────────────────────────
BIN_DIR="${SUBMIT_BASE}/uw_gated_F_5fold_ensemble_bin"
ZIP_PATH="${SUBMIT_BASE}/uw_gated_F_5fold_ensemble_bin.zip"

echo "=== [BINARIZE] ==="
python tools/binarize_ensemble.py \
    --input-dir  "$ENSEMBLE_DIR" \
    --output-dir "$BIN_DIR" \
    --thresholds "$T_BLD" "$T_VEG" "$T_WAT"

echo "=== [ZIP] ==="
rm -f "$ZIP_PATH"
( cd "$SUBMIT_BASE" && zip -r -q "uw_gated_F_5fold_ensemble_bin.zip" "uw_gated_F_5fold_ensemble_bin/" )
ls -lh "$ZIP_PATH"

echo ""
echo "Done: $(date)"
echo "Submit: $ZIP_PATH"
echo "Thresholds baked in: bld=$T_BLD  veg=$T_VEG  wat=$T_WAT"
