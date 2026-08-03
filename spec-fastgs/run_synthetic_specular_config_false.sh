#!/bin/bash

# ============================================================
# SPEC-FASTGS (CONFIG=FALSE) SYNTHETIC-SPECULAR BATCH SCRIPT
# Runs baseline Spec-FastGS (without reflection score/adaptive prior/SH spec mask)
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=${DATA_ROOT:-/home/ghp4hc/datasets/datasets/synthetic_specular}
OUTPUT_ROOT=./output/synthetic_specular

SCENES=(
    ashtray
    dishes
    headphone
    jupyter
    lock
    plane
    record
    teapot
)

# Important final knobs
ASG_DEGREE=64

# Config = False settings (no prior extraction, no ref score, no adaptive prior, no SH spec mask)
REF_PRIOR_METHOD=tan
EXTRACT_REF_PRIOR=False
BACKUP_REF_PRIOR=False
USE_REF_SCORE=False
USE_ADAPTIVE_PRIOR=False
USE_SH_SPEC_MASK=False
STOP_ON_ERROR=True

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
    if [ ! -f "${SOURCE_PATH}/transforms_train.json" ] && [ ! -f "${SOURCE_PATH}/transforms.json" ]; then
        echo "Skipping ${SCENE}: transforms_train.json or transforms.json not found."
        continue
    fi

    echo "========================================================================"
    echo " Starting Spec-FastGS (config=False) Synthetic-Specular scene: ${SCENE}"
    echo "========================================================================"
    echo "Dataset Path : ${SOURCE_PATH}"
    echo "Output Path  : ${MODEL_PATH}"
    echo "ASG Degree   : ${ASG_DEGREE}"
    echo "Use RefScore : ${USE_REF_SCORE}"
    echo "AdaptivePrior: ${USE_ADAPTIVE_PRIOR}"
    echo "SH Spec Mask : ${USE_SH_SPEC_MASK}"
    echo "========================================================================"

    run_or_stop python train.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        --eval \
        --white_background \
        --iterations 30000 \
        --densification_interval 500 \
        --optimizer_type default \
        --asg_degree ${ASG_DEGREE} \
        --sh_degree 3 \
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
