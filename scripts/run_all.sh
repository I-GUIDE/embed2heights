#!/usr/bin/env bash
# ============================================================================
# SINGLE ENTRY POINT — full reproduction of the final submission.
#
# For every ensemble member x fold it runs, in order:
#   1. train      — 4 stages (stage-1 coupled + height/seg/clDice purify)
#   2. predict-val — out-of-fold VAL predictions (for threshold tuning + eval)
#   3. predict-test— the 946 test tiles
# then assembles the 50-seg + height ensemble into submission/FINAL_*.zip.
#
# 5 members (U-Net++ s0/s1/s2, UNet3+ s0, TransUNet s0) x 5 folds = 25 jobs.
# Runs SEQUENTIALLY and is VERY long on one GPU; on a cluster submit each
# (member,fold) as its own job (see README) — every pair is independent.
#
# RESUMABLE: completed train stages / prediction dirs are skipped, so re-running
# after an interruption picks up where it stopped.
#
# Usage:
#   scripts/run_all.sh                 # all 25 member-folds, then assemble
#   MEMBERS="0" FOLDS="0" scripts/run_all.sh   # a single member-fold (smoke test)
#   SKIP_ASSEMBLE=1 scripts/run_all.sh # train + predict only, no zip
# Env:
#   DATA_ROOT   default <repo>/data
#   MEMBERS     default "0 1 2 3 4"
#   FOLDS       default "0 1 2 3 4"
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MEMBERS="${MEMBERS:-0 1 2 3 4}"
FOLDS="${FOLDS:-0 1 2 3 4}"
LOGDIR="$REPO/logs/run_all"; mkdir -p "$LOGDIR"

for MEMBER in $MEMBERS; do
  for FOLD in $FOLDS; do
    LOG="$LOGDIR/m${MEMBER}_f${FOLD}.log"
    echo "==== member $MEMBER fold $FOLD  ($(date '+%F %T'))  -> $LOG ===="
    {
      echo "#### member $MEMBER fold $FOLD  $(date '+%F %T')"
      bash "$REPO/scripts/train_member_fold.sh"        "$MEMBER" "$FOLD"
      bash "$REPO/scripts/predict_val_member_fold.sh"  "$MEMBER" "$FOLD"
      bash "$REPO/scripts/predict_test_member_fold.sh" "$MEMBER" "$FOLD"
    } 2>&1 | tee "$LOG"
  done
done

if [ "${SKIP_ASSEMBLE:-0}" = "1" ]; then
  echo "[run_all] training + prediction done; SKIP_ASSEMBLE=1, not assembling."
  exit 0
fi
echo "==== assembling final submission  ($(date '+%F %T')) ===="
python "$REPO/assemble_final.py"
echo "[run_all] done. Submission zip in $REPO/submission/"
