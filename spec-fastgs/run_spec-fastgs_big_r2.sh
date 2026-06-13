#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN — R2
# Hypothesis: root cause A (supervision) + the architecture-side win.
# Change vs R1: ALSO swap the specular branch to the low-rank / LoRA ASG latent
# (--spec_arch '{"activation":"relu","latent_mode":"lowrank","rank":8}'). The
# ablation (results/MLP_LATENT_ABLATION_2026-06-13.md) found this gives the best
# placement (NCC) with fewer params; ReLU is kept (SIREN/WIRE were refuted).
# Run AFTER R1 so the loss-only vs loss+latent effect can be attributed.
# Verify the banner prints "spec_arch={...}" and "spec_loss_weight=0.5".
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images
RUN=r2                      # keep R1/R2 outputs separate for comparison
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
    --spec_loss_quantile 0.97 \
    --spec_arch '{"activation":"relu","latent_mode":"lowrank","rank":8}'

# 2. RENDER
python render.py \
    -m ${MODEL} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${MODEL}
