#!/bin/bash

# ============================================================
# SPEC-FASTGS MIP-NERF 360 BATCH SCRIPT
# Runs all available Mip-NeRF 360 scenes with images_4
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output/mip360_images4
IMAGES=images_4

SCENES=(
    bicycle
    flowers
    garden
    stump
    treehill
    room
    counter
    kitchen
    bonsai
)

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
STOP_ON_ERROR=True

# Representation Capacity: light SH suppression on high-confidence specular areas.
SH_SPEC_GRAD_SCALE=0.75
SH_SPEC_MASK_START=8000
SH_SPEC_MASK_THRESHOLD=0.75
SH_SPEC_MIN_METRIC_COUNT=2

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

run_or_stop() {
    "$@"
    STATUS=$?
    if [ "$STATUS" -ne 0 ] && [ "$STOP_ON_ERROR" = "True" ]; then
        echo "ERROR: command failed with status ${STATUS}"
        exit "$STATUS"
    fi
    return "$STATUS"
}

for SCENE in "${SCENES[@]}"; do
    SOURCE_PATH=${DATA_ROOT}/${SCENE}
    MODEL_PATH=${OUTPUT_ROOT}/${SCENE}

    if [ ! -d "$SOURCE_PATH" ]; then
        echo "Skipping ${SCENE}: ${SOURCE_PATH} not found."
        continue
    fi
    if [ ! -d "${SOURCE_PATH}/${IMAGES}" ]; then
        echo "Skipping ${SCENE}: ${SOURCE_PATH}/${IMAGES} not found."
        continue
    fi

    echo "========================================================================"
    echo " Starting spec-fastgs Mip-NeRF 360 scene"
    echo "========================================================================"
    echo "Dataset Path : ${SOURCE_PATH}"
    echo "Images       : ${IMAGES}"
    echo "Output Path  : ${MODEL_PATH}"
    echo "ASG Degree   : ${ASG_DEGREE}"
    echo "Prior Method : ${REF_PRIOR_METHOD}"
    echo "Extract Prior: ${EXTRACT_REF_PRIOR}"
    echo "Use RefScore : ${USE_REF_SCORE}"
    echo "AdaptivePrior: ${USE_ADAPTIVE_PRIOR}"
    echo "SH Spec Mask : ${USE_SH_SPEC_MASK}"
    echo "========================================================================"

    if [ "$USE_REF_SCORE" = "True" ] && [ "$EXTRACT_REF_PRIOR" = "True" ]; then
        PRIOR_DIR=${SOURCE_PATH}/reflection_prior
        if [ "$BACKUP_REF_PRIOR" = "True" ] && [ -d "$PRIOR_DIR" ]; then
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            BACKUP_ROOT=${SOURCE_PATH}/backups
            BACKUP_DIR=${BACKUP_ROOT}/reflection_prior_${TIMESTAMP}
            echo "Backing up existing reflection_prior to:"
            echo "  ${BACKUP_DIR}"
            mkdir -p "$BACKUP_ROOT"
            mv "$PRIOR_DIR" "$BACKUP_DIR"
        fi

        run_or_stop python extract_reflection_prior.py \
            -s ${SOURCE_PATH} \
            -i ${IMAGES} \
            --ref_prior_method ${REF_PRIOR_METHOD}
    fi

    run_or_stop python train.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        -i ${IMAGES} \
        --eval \
        --iterations 30000 \
        --densification_interval 100 \
        --optimizer_type default \
        --asg_degree ${ASG_DEGREE} \
        --is_real \
        --is_indoor \
        --sh_degree 3 \
        --highfeature_lr 0.02 \
        --grad_abs_thresh 0.0004 \
        --specular_start_iter 3000 \
        --ref_prior_method ${REF_PRIOR_METHOD} \
        --sh_spec_grad_scale ${SH_SPEC_GRAD_SCALE} \
        --sh_spec_mask_start ${SH_SPEC_MASK_START} \
        --sh_spec_mask_threshold ${SH_SPEC_MASK_THRESHOLD} \
        --sh_spec_min_metric_count ${SH_SPEC_MIN_METRIC_COUNT} \
        ${REF_SCORE_FLAG} \
        ${ADAPTIVE_PRIOR_FLAG} \
        ${SH_SPEC_MASK_FLAG}

    run_or_stop python render.py \
        -m ${MODEL_PATH} \
        --skip_train

    run_or_stop python metrics.py \
        -m ${MODEL_PATH}
done
