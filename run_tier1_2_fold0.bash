#!/bin/bash
#SBATCH --job-name=emb2h_t12
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-2

# Three parallel experiments on fold 0 of group_code_5fold_seed42 — the
# hardest, most leakage-honest fold (gated_feature champion got 0.4766
# there, baseline 0.42). Each isolates one architectural change against
# the gated_feature champion recipe so deltas are interpretable.
#
#   0  hierarchical_v1 : 4-modality hierarchical bipartite GMU
#                        (annual ↔ epoch fusion grouped by temporal scope)
#                        — Tier 1 from blueprints
#   1  dirichlet_v1    : 2-modality champion + Dirichlet aux loss on the
#                        4-class fractional simplex (B/V/W/Background)
#                        — Tier 1 from blueprints
#   2  ctaskattn_v1    : 2-modality champion + cross-task spatial attention
#                        from land-cover fractions onto height_trunk
#                        — Tier 2 from blueprints
#
# Promotion bar (each, vs gated_F_fold0 = 0.4766):
#   ≥ 0.4826 (+0.006 noise margin), no axis dropping > 0.02.

SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/embed2heights"
DATA_DIR="/projects/bcrm/emb2height/data/train"
SPLIT="${SCRIPT_DIR}/splits/group_code_5fold_seed42/fold_0/split.json"

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

# Shared core args (matches uw_gated_F's recipe).
COMMON_ARGS=(
    --train-targets-dir "${DATA_DIR}/labels"
    --split-file "$SPLIT"
    --batch-size 32
    --patch-size 256
    --epochs 30
    --lr 2e-4
    --weight-decay 1e-4
    --loss-preset presence_centered
    --aux-weight 1.0
    --presence-tversky-weight 1.0
    --fraction-mae-weight 0.1
    --height-specialist-depth 2
    --lightunet-base-ch 48
    --height-loss-kind l1
    --huber-delta 1.0
    --build-height-boost 5.0
    --veg-height-boost 1.5
    --aux-veg-weight 1.0
    --iou-loss-kind tversky
    --focal-gamma 2.0
    --focal-alpha 0.25
    --structure-weight 2.0
    --no-augment
    --scheduler plateau
    --compile
    --no-uncertainty-weighting
    --seed 42
)

case $SLURM_ARRAY_TASK_ID in
    0)
        # 4-modality hierarchical bipartite (annual ↔ epoch).
        EXP_NAME="hierarchical_v1"
        EXTRA=(
            --experiment-name "$EXP_NAME"
            --model-type multi_gfm
            --train-embeddings-dir "${DATA_DIR}/alphaearth_emb"
            --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb"
            --terramind-s1-train-emb-dir "${DATA_DIR}/terramind_s1_emb"
            --terramind-s2-train-emb-dir "${DATA_DIR}/terramind_s2_emb"
            --thor-s1-train-emb-dir "${DATA_DIR}/thor_s1_emb"
            --thor-s2-train-emb-dir "${DATA_DIR}/thor_s2_emb"
            --fusion-mode hierarchical
            --modality-dropout 0.15
        )
        ;;
    1)
        # 2-modality champion + Dirichlet aux loss.
        EXP_NAME="dirichlet_v1"
        EXTRA=(
            --experiment-name "$EXP_NAME"
            --model-type tessera_iou_fusion
            --train-embeddings-dir "${DATA_DIR}/alphaearth_emb"
            --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb"
            --tessera-presence-ch 16
            --tessera-hidden-ch 96
            --tessera-hidden-depth 2
            --fusion-mode gated_feature
            --gate-mode simple
            --no-gate-untied
            --modality-dropout 0.0
            --dirichlet-weight 0.5
        )
        ;;
    2)
        # 2-modality champion + cross-task spatial attention.
        EXP_NAME="ctaskattn_v1"
        EXTRA=(
            --experiment-name "$EXP_NAME"
            --model-type tessera_iou_fusion
            --train-embeddings-dir "${DATA_DIR}/alphaearth_emb"
            --secondary-train-embeddings-dir "${DATA_DIR}/tessera_emb"
            --tessera-presence-ch 16
            --tessera-hidden-ch 96
            --tessera-hidden-depth 2
            --fusion-mode gated_feature
            --gate-mode simple
            --no-gate-untied
            --modality-dropout 0.0
            --cross-task-attention
        )
        ;;
    *)
        echo "Unknown array task id: $SLURM_ARRAY_TASK_ID" >&2
        exit 1
        ;;
esac

echo "========================================"
echo "task=$SLURM_ARRAY_TASK_ID  exp=$EXP_NAME"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

python train.py "${EXTRA[@]}" "${COMMON_ARGS[@]}"
