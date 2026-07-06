#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN SCRIPT
# ============================================================
# All tunables below respect an already-exported environment variable of the same
# name (the "${VAR:-default}" pattern) and only fall back to the hardcoded default
# otherwise -- this lets a caller (e.g. a notebook launching several configs, one per
# GPU) override ASG_DEGREE/USE_REF_SCORE/CUDA_VISIBLE_DEVICES/OUTPUT_SUFFIX per
# invocation without editing this file, while `bash run_spec-fastgs_big.sh` alone
# still behaves exactly as before.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE="${SCENE:-counter}"
IMAGES="${IMAGES:-images_8}"
ASG_DEGREE="${ASG_DEGREE:-24}"
USE_REF_SCORE="${USE_REF_SCORE:-False}"
NUM_SCORE_CAMERAS="${NUM_SCORE_CAMERAS:-10}"
FULL_ASG_INTERVAL="${FULL_ASG_INTERVAL:-0}"
F_REST_WARMUP_UNTIL="${F_REST_WARMUP_UNTIL:-3000}"
F_REST_INTERVAL_EARLY="${F_REST_INTERVAL_EARLY:-16}"
F_REST_INTERVAL_MID="${F_REST_INTERVAL_MID:-32}"
F_REST_INTERVAL_LATE="${F_REST_INTERVAL_LATE:-64}"
SK_INTENSITY="${SK_INTENSITY:-0.7}"
SK_SATURATION="${SK_SATURATION:-0.2}"
# Distinguishes output dirs across multiple configs/GPUs run against the same SCENE
# (e.g. an ASG_DEGREE sweep) -- default "" reproduces the original ./output/<scene>
# layout exactly for a single run.
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE}${OUTPUT_SUFFIX}"

echo "[config] GPU=${CUDA_VISIBLE_DEVICES} scene=${SCENE} images=${IMAGES} asg_degree=${ASG_DEGREE} use_ref_score=${USE_REF_SCORE} output=${OUTPUT_DIR}"

REF_SCORE_FLAG=""
if [ "$USE_REF_SCORE" = "True" ]; then
    REF_SCORE_FLAG="--use_ref_score"
fi

# 0. EXTRACT REFLECTION PRIOR
# Idempotent: only depends on SCENE/IMAGES (not ASG_DEGREE), so if two configs for the
# SAME scene already produced it, skip recomputing -- matters when this script is
# invoked several times back-to-back for an ASG_DEGREE sweep on one scene.
REF_PRIOR_DIR="${DATA_ROOT}/${SCENE}/reflection_prior"
if [ "$USE_REF_SCORE" = "True" ]; then
    if [ -d "$REF_PRIOR_DIR" ] && [ -n "$(ls -A "$REF_PRIOR_DIR" 2>/dev/null)" ]; then
        echo "[0/4] Reflection prior already present at ${REF_PRIOR_DIR}, skipping extraction"
    else
        echo "[0/4] Running extract_reflection_prior.py..."
        python extract_reflection_prior.py \
            -s ${DATA_ROOT}/${SCENE} \
            -i ${IMAGES} \
            --sk_intensity ${SK_INTENSITY} \
            --sk_saturation ${SK_SATURATION}
    fi
else
    echo "[0/4] Skipping extract_reflection_prior.py because USE_REF_SCORE=False"
fi

# 1. TRAIN
echo "[1/4] Running train.py..."
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_DIR} \
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
    -m ${OUTPUT_DIR} \
    --skip_train

# 3. METRICS
echo "[3/4] Running metrics.py..."
python metrics.py \
    -m ${OUTPUT_DIR}
