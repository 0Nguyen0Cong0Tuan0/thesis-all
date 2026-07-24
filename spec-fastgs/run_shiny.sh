#!/bin/bash

# ============================================================
# SPEC-FASTGS SHINY / REF-NERF RUN SCRIPT
# Minimal synthetic config: train -> render -> metrics
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/Ref-NeRF/refnerf
OUTPUT_ROOT=./output
SCENE=toaster

# Important final knobs
ASG_DEGREE=64

# Reflection prior method:
#   tan    = Tan-Ikeuchi
#   shafer = Shafer/Klinker
#   hybrid = combined heuristic
REF_PRIOR_METHOD=tan
EXTRACT_REF_PRIOR=True
BACKUP_REF_PRIOR=True
USE_REF_SCORE=True
USE_ADAPTIVE_PRIOR=True
USE_SH_SPEC_MASK=True

# Representation Capacity: light SH suppression on high-confidence specular areas.
SH_SPEC_GRAD_SCALE=0.75
SH_SPEC_MASK_START=8000
SH_SPEC_MASK_THRESHOLD=0.75
SH_SPEC_MIN_METRIC_COUNT=3

REF_SCORE_FLAG=""
if [ "$USE_REF_SCORE" = "True" ]; then
    REF_SCORE_FLAG="--use_ref_score"
fi

ADAPTIVE_PRIOR_FLAG=""
if [ "$USE_ADAPTIVE_PRIOR" = "True" ]; then
    ADAPTIVE_PRIOR_FLAG="--use_adaptive_prior"
fi

SH_SPEC_MASK_FLAG=""
if [ "$USE_SH_SPEC_MASK" = "True" ]; then
    SH_SPEC_MASK_FLAG="--use_sh_spec_mask"
fi

echo "========================================================================"
echo " Starting spec-fastgs Training Pipeline for Synthetic Dataset"
echo "========================================================================"
echo "Dataset Path : ${DATA_ROOT}/${SCENE}"
echo "Output Path  : ${OUTPUT_ROOT}/${SCENE}"
echo "ASG Degree   : ${ASG_DEGREE}"
echo "Prior Method : ${REF_PRIOR_METHOD}"
echo "Extract Prior: ${EXTRACT_REF_PRIOR}"
echo "Use RefScore : ${USE_REF_SCORE}"
echo "AdaptivePrior: ${USE_ADAPTIVE_PRIOR}"
echo "SH Spec Mask : ${USE_SH_SPEC_MASK}"
echo "========================================================================"

# 0. EXTRACT REFLECTION PRIOR
if [ "$USE_REF_SCORE" = "True" ] && [ "$EXTRACT_REF_PRIOR" = "True" ]; then
    PRIOR_DIR=${DATA_ROOT}/${SCENE}/reflection_prior
    if [ "$BACKUP_REF_PRIOR" = "True" ] && [ -d "$PRIOR_DIR" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BACKUP_ROOT=${DATA_ROOT}/${SCENE}/backups
        BACKUP_DIR=${BACKUP_ROOT}/reflection_prior_${TIMESTAMP}
        echo "Backing up existing reflection_prior to:"
        echo "  ${BACKUP_DIR}"
        mkdir -p "$BACKUP_ROOT"
        mv "$PRIOR_DIR" "$BACKUP_DIR"
    fi

    python extract_reflection_prior.py \
        -s ${DATA_ROOT}/${SCENE} \
        --ref_prior_method ${REF_PRIOR_METHOD}
fi

# 1. TRAIN
# python train.py \
#     -s ${DATA_ROOT}/${SCENE} \
#     -m ${OUTPUT_ROOT}/${SCENE} \
#     --eval \
#     --white_background \
#     --iterations 30000 \
#     --densification_interval 500 \
#     --optimizer_type default \
#     --asg_degree ${ASG_DEGREE} \
#     --sh_degree 3 \
#     --specular_start_iter 3000 \
#     --ref_prior_method ${REF_PRIOR_METHOD} \
#     --sh_spec_grad_scale ${SH_SPEC_GRAD_SCALE} \
#     --sh_spec_mask_start ${SH_SPEC_MASK_START} \
#     --sh_spec_mask_threshold ${SH_SPEC_MASK_THRESHOLD} \
#     --sh_spec_min_metric_count ${SH_SPEC_MIN_METRIC_COUNT} \
#     ${REF_SCORE_FLAG} \
#     ${ADAPTIVE_PRIOR_FLAG} \
#     ${SH_SPEC_MASK_FLAG}

# 2. RENDER
python render.py \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}

echo "========================================================================"
echo " Pipeline Completed Successfully!"
echo " Results and train_info.json are saved in ${OUTPUT_ROOT}/${SCENE}"
echo "========================================================================"
