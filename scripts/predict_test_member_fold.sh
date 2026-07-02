#!/usr/bin/env bash
# Predict the 946 TEST tiles for ONE member-fold, for the 3 checkpoints the final
# submission consumes: _cldice + _segpurify (seg ch0-2) and _purify (height ch3).
# Writes runs/<exp>_{cldice,segpurify,purify}/test_predictions/*.npy
# Resumable: a stage whose test_predictions dir is already populated is skipped.
#
# Usage:  MEMBER (0-4)  FOLD (0-4)
#   scripts/predict_test_member_fold.sh 0 0
# Env:  DATA_ROOT (default <repo>/data) — must contain test/{alphaearth_test_emb,...}
set -euo pipefail
MEMBER=${1:?member idx 0-4}; FOLD=${2:?fold 0-4}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO/data}"; T="$DATA_ROOT/test"
export REPO_DIR="$REPO"; cd "$REPO"

CFGS=(xfusion_095_unetpp \
      xfusion_095_unetpp \
      xfusion_095_unetpp \
      xfusion_095_unet3plus \
      xfusion_095_unetpp_trans)
SEEDS=(0 1 2 0 0)
CFG=${CFGS[$MEMBER]}; SEED=${SEEDS[$MEMBER]}
EXP="${CFG}_s${SEED}_f${FOLD}"

PARGS="--test-embeddings-dir $T/alphaearth_test_emb --secondary-test-embeddings-dir $T/tessera_test_emb \
  --token-test-embeddings-dir $T/terramind_test_s1_emb --secondary-token-test-embeddings-dir $T/terramind_test_s2_emb \
  --third-token-test-embeddings-dir $T/thor_test_s1_emb --fourth-token-test-embeddings-dir $T/thor_test_s2_emb"

for STAGE in cldice segpurify purify; do
  E="${EXP}_${STAGE}"
  [ -f "runs/$E/model_best.pth" ] || { echo "skip $E (no ckpt)"; continue; }
  OUT="$REPO/runs/$E/test_predictions"
  if ls "$OUT"/*.npy >/dev/null 2>&1; then echo "[skip] $E test_predictions (populated)"; continue; fi
  python predict.py --experiment-name "$E" --base-dir "$REPO/runs" $PARGS \
    --predictions-dir "$OUT" --patch-size 256 --max-samples 0
done
echo "[predict-test] done: ${EXP} (cldice/segpurify/purify test_predictions)"
