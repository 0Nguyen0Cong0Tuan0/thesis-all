#!/bin/bash

# ============================================================
# SPECULAR-GAUSSIANS SYNTHETIC-SPECULAR BATCH SCRIPT
# Runs all 8 synthetic specular scenes
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=${DATA_ROOT:-/home/ghp4hc/datasets/datasets/synthetic_specular}
OUTPUT_ROOT=./output/synthetic_specular
STOP_ON_ERROR=True

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
    echo " Starting Specular-Gaussians Synthetic-Specular scene: ${SCENE}"
    echo "========================================================================"
    echo "Dataset Path : ${SOURCE_PATH}"
    echo "Output Path  : ${MODEL_PATH}"
    echo "========================================================================"

    run_or_stop python train.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        --white_background \
        --eval \
        --asg_degree 64

    run_or_stop python render.py \
        -m ${MODEL_PATH} \
        --skip_train

    run_or_stop python metrics.py \
        -m ${MODEL_PATH}
done
