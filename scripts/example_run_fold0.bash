#!/bin/bash
#SBATCH --job-name=emb2h_example_f0
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gres=gpu:1
#
# Canonical fold-0 launcher (self-contained example).
#
# Architecture: ae_tessera_gated (LightUNet + gated Tessera fusion), with
# the "tess128" trunk (tessera_presence_ch=32, tessera_hidden_ch=128),
# softbin height head, building-interior smoothness, and bidirectional
# cross-task attention. Trains 70 epochs (val loss converges around
# epoch 60-66; e70 is the validated sweet spot).
#
# Single seed scores ~0.499–0.509 on fold0 val. A 3-seed ensemble of this
# config lifts to ~0.517 (single → ensemble += ~0.016). K-water
# postprocessing at inference adds another ~+0.003.
#
# Set REPO_DIR to wherever you cloned embed2heights, then:
#   sbatch scripts/example_run_fold0.bash

set -e

# ── Repo root (defaults to script's parent; override via env var) ─────────
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/train}"
TEST_DIR="${TEST_DIR:-${REPO_DIR}/data/test}"
SPLIT="${SPLIT:-${REPO_DIR}/splits/group_code_5fold_seed42/fold_0/split.json}"
SUBMIT_BASE="${REPO_DIR}/runs/submission"

# ── Conda activation ──────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
fi
__conda_setup="$("${CONDA_BASE}/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate emb2heights

cd "$REPO_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p slurm_logs "${SUBMIT_BASE}/_sweep" "${SUBMIT_BASE}"

EXP="${EXP:-example_tess128_e70_fold0}"
echo "Node: $(hostname) | EXP: $EXP | REPO_DIR: $REPO_DIR"

# ── 1. Train ──────────────────────────────────────────────────────────────
python train.py \
    --experiment-name "$EXP" \
    --model-type ae_tessera_gated \
    --train-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
    --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb" \
    --train-targets-dir "${DATA_DIR}/labels" \
    --split-file "$SPLIT" \
    --batch-size 32 \
    --patch-size 256 \
    --epochs 70 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --loss-preset presence_centered \
    --aux-weight 1.0 \
    --presence-tversky-weight 1.0 \
    --fraction-mae-weight 0.1 \
    --tessera-presence-ch 32 \
    --tessera-hidden-ch 128 \
    --tessera-hidden-depth 2 \
    --height-specialist-depth 2 \
    --lightunet-base-ch 48 \
    --height-loss-kind berhu \
    --build-height-boost 5.0 \
    --veg-height-boost 1.5 \
    --aux-veg-weight 1.0 \
    --iou-loss-kind tversky \
    --bidirectional-ctask \
    --building-smooth-weight 0.5 --building-smooth-erode-px 2 --building-smooth-thr 0.5 \
    --height-head-kind softbin --height-bin-aux-weight 0.5 \
    --compile \
    --seed 42

# ── 2. Predict val ────────────────────────────────────────────────────────
python predict.py \
    --experiment-name "$EXP" \
    --test-embeddings-dir "${DATA_DIR}/alphaearth_emb" \
    --secondary-test-embeddings-dir "${DATA_DIR}/tessera_emb" \
    --test-targets-dir "${DATA_DIR}/labels"

# ── 3. Predict test ───────────────────────────────────────────────────────
PRED_TEST_DIR="${SUBMIT_BASE}/preds_${EXP}"
python predict.py \
    --experiment-name "$EXP" \
    --test-embeddings-dir "${TEST_DIR}/alphaearth_test_emb" \
    --secondary-test-embeddings-dir "${TEST_DIR}/tessera_test_emb" \
    --predictions-dir "$PRED_TEST_DIR"

# ── 4. Sweep thresholds (with optional K-water filter for free iou_wat lift)
python tools/sweep_thresholds.py \
    --pred-dir "${REPO_DIR}/runs/${EXP}/predictions" \
    --labels-dir "${DATA_DIR}/labels" \
    --split-file "$SPLIT" \
    --water-k-grid "0,8,12,14,16" \
    | tee "${SUBMIT_BASE}/_sweep/${EXP}.log"

echo "Done $EXP: $(date)"
echo "Inspect: runs/${EXP}/  and  ${SUBMIT_BASE}/_sweep/${EXP}.log"

