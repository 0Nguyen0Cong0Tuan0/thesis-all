#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN SCRIPT
# ============================================================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_ROOT=${DATA_ROOT:-./datasets/mipnerf360}
OUTPUT_ROOT=${OUTPUT_ROOT:-./output}
SCENE=${SCENE:-counter}
IMAGES=${IMAGES:-images}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-""}

# 1. TRAIN
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_ROOT}/${SCENE}${OUTPUT_SUFFIX} \
    -i ${IMAGES} \
    --eval \
    --iterations 30000 \
    --densification_interval 100 \
    --optimizer_type default \
    --asg_degree 24 \
    --is_real \
    --is_indoor \
    --sh_degree 3 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0004

# 2. RENDER
python render.py \
    -m ${OUTPUT_ROOT}/${SCENE}${OUTPUT_SUFFIX} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}${OUTPUT_SUFFIX}
