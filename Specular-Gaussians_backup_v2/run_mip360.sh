#!/bin/bash

# ============================================================
# SPECULAR-GAUSSIANS MIP-NERF 360 BATCH SCRIPT
# Runs all 9 Mip-NeRF 360 scenes with configurable resolution
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=${DATA_ROOT:-/home/ghp4hc/datasets/datasets/mipneft360}
IMAGES=${IMAGES:-images_4}
OUTPUT_ROOT=./output/mip360_${IMAGES}
STOP_ON_ERROR=True

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

INDOOR_SCENES=(room counter kitchen bonsai)

is_indoor() {
    local scene=$1
    for s in "${INDOOR_SCENES[@]}"; do
        [ "$s" = "$scene" ] && return 0
    done
    return 1
}

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
    if [ -d "${DATA_ROOT}/360_v2/${SCENE}" ]; then
        SOURCE_PATH="${DATA_ROOT}/360_v2/${SCENE}"
    elif [ -d "${DATA_ROOT}/360_extra_scenes/${SCENE}" ]; then
        SOURCE_PATH="${DATA_ROOT}/360_extra_scenes/${SCENE}"
    else
        SOURCE_PATH="${DATA_ROOT}/${SCENE}"
    fi
    MODEL_PATH=${OUTPUT_ROOT}/${SCENE}

    if [ ! -d "$SOURCE_PATH" ]; then
        echo "Skipping ${SCENE}: ${SOURCE_PATH} not found."
        continue
    fi
    if [ ! -d "${SOURCE_PATH}/${IMAGES}" ]; then
        echo "Skipping ${SCENE}: ${SOURCE_PATH}/${IMAGES} not found."
        continue
    fi

    INDOOR_FLAG=""
    if is_indoor "$SCENE"; then
        INDOOR_FLAG="--is_indoor"
    fi

    echo "========================================================================"
    echo " Starting Specular-Gaussians Mip-NeRF 360 scene: ${SCENE}"
    echo "========================================================================"
    echo "Dataset Path : ${SOURCE_PATH}"
    echo "Images       : ${IMAGES}"
    echo "Output Path  : ${MODEL_PATH}"
    echo "========================================================================"

    run_or_stop python train.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        -i ${IMAGES} \
        --eval \
        --is_real \
        --asg_degree 12 \
        ${INDOOR_FLAG}

    run_or_stop python render.py \
        -m ${MODEL_PATH} \
        --skip_train

    run_or_stop python metrics.py \
        -m ${MODEL_PATH}
done
