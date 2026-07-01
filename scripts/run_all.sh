#!/usr/bin/env bash
# End-to-end reproduction: train + test-predict all 25 member-folds, then assemble.
# 5 members (U-Net++ s0/s1/s2, UNet3+ s0, TransUNet s0) x 5 folds.
# Each member-fold trains 4 stages then predicts the test set for 3 of them.
#
# This runs SEQUENTIALLY (very long). On a cluster, submit each member-fold as a
# separate job instead (see README) — every (member,fold) is independent.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
for MEMBER in 0 1 2 3 4; do
  for FOLD in 0 1 2 3 4; do
    bash "$REPO/scripts/train_member_fold.sh"        "$MEMBER" "$FOLD"
    bash "$REPO/scripts/predict_test_member_fold.sh" "$MEMBER" "$FOLD"
  done
done
python "$REPO/assemble_final.py"
