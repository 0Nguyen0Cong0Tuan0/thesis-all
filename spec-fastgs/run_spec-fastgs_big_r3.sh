#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN — R3  (v2.6)
# Hypothesis: v2.5-R1 failed only because its highlight mask was MIS-TARGETED
# (top GT-luminance => caught bright DIFFUSE: white counter/walls), which drove
# +31% Gaussians and regressed every metric.
#
# Change vs R1: re-target the specular loss mask to the |GT - diffuse| residual
# (--spec_loss_mode residual, the SAME locator the offline diagnostic uses) and
# lower the weight (0.15) on the top-5% (quantile 0.95). Architecture unchanged
# (ReLU, dense 24-d latent) to isolate the supervision fix.
#
# Predicted if the hypothesis holds: energyRatio UP, NCC UP, gain a* -> 1, WITHOUT
# the Gaussian blow-up (count should stay near v2.4's ~588k, not 768k).
# Verify the startup banner prints "code: v2.6-..." and
# "spec_loss_weight=0.15 | spec_loss_mode=residual".
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images
RUN=r3                      # keep run outputs separate for comparison
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
    --spec_loss_weight 0.15 \
    --spec_loss_quantile 0.95 \
    --spec_loss_mode residual

# 2. RENDER
python render.py \
    -m ${MODEL} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${MODEL}
