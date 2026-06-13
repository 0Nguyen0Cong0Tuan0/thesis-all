#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN — R1
# Hypothesis: root cause A (supervision deficit).
# Change vs v2.4: ONLY add the specular-weighted L1 loss on the brightest
# GT pixels (--spec_loss_weight 0.5). Architecture unchanged (ReLU, dense 24-d
# latent). Isolates whether highlight-targeted supervision lifts the specular
# energy (predicted: energyRatio up, gain a* -> 1).
# Verify the startup banner prints "code: v2.5-..." and "spec_loss_weight=0.5".
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images
RUN=r1                      # keep R1/R2 outputs separate for comparison
MODEL=${OUTPUT_ROOT}/${SCENE}_${RUN}

# 1. TRAIN
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${MODEL} \
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
    --grad_abs_thresh 0.0004 \
    --spec_loss_weight 0.5 \
    --spec_loss_quantile 0.97

# 2. RENDER
python render.py \
    -m ${MODEL} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${MODEL}
