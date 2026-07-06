#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN SCRIPT
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images_8
ASG_DEGREE=24
USE_REF_SCORE=False
NUM_SCORE_CAMERAS=10
FULL_ASG_INTERVAL=0
F_REST_WARMUP_UNTIL=3000
F_REST_INTERVAL_EARLY=16
F_REST_INTERVAL_MID=32
F_REST_INTERVAL_LATE=64
SK_INTENSITY=0.7
SK_SATURATION=0.2

REF_SCORE_FLAG=""
if [ "$USE_REF_SCORE" = "True" ]; then
    REF_SCORE_FLAG="--use_ref_score"
fi

# 0. EXTRACT REFLECTION PRIOR
if [ "$USE_REF_SCORE" = "True" ]; then
    echo "[0/4] Running extract_reflection_prior.py..."
    python extract_reflection_prior.py \
        -s ${DATA_ROOT}/${SCENE} \
        -i ${IMAGES} \
        --sk_intensity ${SK_INTENSITY} \
        --sk_saturation ${SK_SATURATION}
else
    echo "[0/4] Skipping extract_reflection_prior.py because USE_REF_SCORE=False"
fi

# 1. TRAIN
echo "[1/4] Running train.py..."
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_ROOT}/${SCENE} \
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
    --num_score_cameras ${NUM_SCORE_CAMERAS} \
    --full_asg_interval ${FULL_ASG_INTERVAL} \
    --f_rest_warmup_until ${F_REST_WARMUP_UNTIL} \
    --f_rest_interval_early ${F_REST_INTERVAL_EARLY} \
    --f_rest_interval_mid ${F_REST_INTERVAL_MID} \
    --f_rest_interval_late ${F_REST_INTERVAL_LATE} \
    --sk_intensity ${SK_INTENSITY} \
    --sk_saturation ${SK_SATURATION} \
    ${REF_SCORE_FLAG}

# 2. RENDER
echo "[2/4] Running render.py..."
python render.py \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --skip_train

# 3. METRICS
echo "[3/4] Running metrics.py..."
python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}
