
SCRIPT_DIR="/projects/bcrm/akhot2/embed2heights_max"
DATA_DIR="/projects/bcrm/emb2height/data/train"
SPLITS_ROOT="${SCRIPT_DIR}/splits/group_code_5fold_seed42"

cd "$SCRIPT_DIR"

EXP="uw_gated_F_fold"
SPLIT="${SPLITS_ROOT}/fold_${FOLD}/split.json"


# 1. Ensemble test predictions
python tools/ensemble.py mean \
    --inputs "submission/${EXP}0" "submission/${EXP}1" "submission/${EXP}2" "submission/${EXP}3" "submission/${EXP}4" \
    --output-dir "submission/ens-${EXP}"




