#!/usr/bin/env bash
# Predict this fold's held-out VALIDATION tiles (out-of-fold) for ONE member-fold.
# These OOF predictions are what assemble_final.py tunes the per-class seg
# thresholds on, and what evaluate.py scores per fold.
#
# Runs predict.py on the TRAIN embeddings + labels, restricted to the fold's val
# split (--restrict-val-split), for the same 3 checkpoints as the test pass.
# Writes runs/<exp>_{cldice,segpurify,purify}/predictions/*.npy
# Resumable: a stage whose predictions dir is already populated is skipped.
#
# Usage:  MEMBER (0-4)  FOLD (0-4)
#   scripts/predict_val_member_fold.sh 0 0
# Env:  DATA_ROOT (default <repo>/data) — must contain train/{alphaearth_emb,...,labels}
set -euo pipefail
MEMBER=${1:?member idx 0-4}; FOLD=${2:?fold 0-4}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO/data}"; TR="$DATA_ROOT/train"
export REPO_DIR="$REPO"; cd "$REPO"

CFGS=(xfusion_095_unetpp \
      xfusion_095_unetpp \
      xfusion_095_unetpp \
      xfusion_095_unet3plus \
      xfusion_095_unetpp_trans)
SEEDS=(0 1 2 0 0)
CFG=${CFGS[$MEMBER]}; SEED=${SEEDS[$MEMBER]}
EXP="${CFG}_s${SEED}_f${FOLD}"
SPLIT="$REPO/splits/group_code_5fold_seed42/fold_${FOLD}/split.json"

PARGS="--test-embeddings-dir $TR/alphaearth_emb --secondary-test-embeddings-dir $TR/tessera_emb \
  --token-test-embeddings-dir $TR/terramind_s1_emb --secondary-token-test-embeddings-dir $TR/terramind_s2_emb \
  --third-token-test-embeddings-dir $TR/thor_s1_emb --fourth-token-test-embeddings-dir $TR/thor_s2_emb \
  --test-targets-dir $TR/labels --restrict-val-split $SPLIT"

for STAGE in cldice segpurify purify; do
  E="${EXP}_${STAGE}"
  [ -f "runs/$E/model_best.pth" ] || { echo "skip $E (no ckpt)"; continue; }
  OUT="$REPO/runs/$E/predictions"
  if ls "$OUT"/*.npy >/dev/null 2>&1; then echo "[skip] $E predictions (populated)"; continue; fi
  python predict.py --experiment-name "$E" --base-dir "$REPO/runs" $PARGS \
    --predictions-dir "$OUT" --patch-size 256 --max-samples 0
done
echo "[predict-val] done: ${EXP} (cldice/segpurify/purify OOF predictions)"
