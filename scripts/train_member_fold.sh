#!/usr/bin/env bash
# Train ONE ensemble member on ONE fold, through all 4 stages.
#   stage 1       : coupled seg+height (50 ep)                       -> <exp>/model_best.pth
#   height-purify : freeze seg side, height owns backbone (20 ep)    -> <exp>_purify     (ch 3)
#   seg-purify    : freeze height side, seg owns backbone (20 ep)    -> <exp>_segpurify  (ch 0-2)
#   cldice-purify : seg-purify + clDice topology loss (20 ep)        -> <exp>_cldice     (ch 0-2)
#
# Usage:  MEMBER (0-4)  FOLD (0-4)
#   scripts/train_member_fold.sh 0 0
# Env:  DATA_ROOT (default <repo>/data)  — must contain train/{alphaearth_emb,...,labels}
set -euo pipefail
MEMBER=${1:?member idx 0-4}; FOLD=${2:?fold 0-4}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO/data}"; TR="$DATA_ROOT/train"
export REPO_DIR="$REPO"; cd "$REPO"

# member -> (config, seed):  U-Net++ x3 seeds, UNet3+, TransUNet
CFGS=(xfusion_095_p3_2stage_softbin_covgt10_delmask_unetpp \
      xfusion_095_p3_2stage_softbin_covgt10_delmask_unetpp \
      xfusion_095_p3_2stage_softbin_covgt10_delmask_unetpp \
      xfusion_095_p3_2stage_softbin_covgt10_delmask_unet3plus \
      xfusion_095_p3_2stage_softbin_covgt10_delmask_unetpp_trans)
SEEDS=(0 1 2 0 0)
CFG=${CFGS[$MEMBER]}; SEED=${SEEDS[$MEMBER]}
CONFIG="configs/active/${CFG}.yml"
SPLIT="splits/group_code_5fold_seed42/fold_${FOLD}/split.json"
EXP="${CFG}_s${SEED}_f${FOLD}"

DARGS="--train-embeddings-dir $TR/alphaearth_emb --secondary-train-embeddings-dir $TR/tessera_emb \
  --token-train-embeddings-dir $TR/terramind_s1_emb --secondary-token-train-embeddings-dir $TR/terramind_s2_emb \
  --third-token-train-embeddings-dir $TR/thor_s1_emb --fourth-token-train-embeddings-dir $TR/thor_s2_emb \
  --train-targets-dir $TR/labels"

echo "[train] member $MEMBER ($CFG seed $SEED) fold $FOLD"
# ---- stage 1 ----
python train.py --config "$CONFIG" --experiment-name "$EXP" --split-file "$SPLIT" --seed "$SEED" \
  --epochs 50 $DARGS
# ---- height-purify (ch 3) ----
python train.py --config "$CONFIG" --experiment-name "${EXP}_purify" --split-file "$SPLIT" --seed "$SEED" \
  --init-checkpoint "runs/${EXP}/model_best.pth" --presence-trunk-grad-scale 0.0 --epochs 20 --lr 0.00015 $DARGS
# ---- seg-purify (ch 0-2, source 1) ----
python train.py --config "$CONFIG" --experiment-name "${EXP}_segpurify" --split-file "$SPLIT" --seed "$SEED" \
  --init-checkpoint "runs/${EXP}/model_best.pth" --height-trunk-grad-scale 0.0 --epochs 20 --lr 0.00015 $DARGS
# ---- clDice-purify (ch 0-2, source 2) ----
python train.py --config "$CONFIG" --experiment-name "${EXP}_cldice" --split-file "$SPLIT" --seed "$SEED" \
  --init-checkpoint "runs/${EXP}_segpurify/model_best.pth" --height-trunk-grad-scale 0.0 \
  --cl-dice-weight 1.0 --epochs 20 --lr 0.00015 $DARGS
echo "[train] done: ${EXP} (+_purify/_segpurify/_cldice)"
