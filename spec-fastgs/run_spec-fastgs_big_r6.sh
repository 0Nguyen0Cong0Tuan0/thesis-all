#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN — R6  (v2.9)  — DISK-FREE root cause B
# Hypothesis: the specular PLACEMENT deficit (NCC/sigma flat since v2.6) is caused
# by the noisy geometric min-axis normal that sets reflect_dir in ASGRender. The
# monocular-normal-prior route (R4/R5) was disk-heavy (~1GB of .npy maps -> Kaggle
# "No space left on device") and only weak-positive.
#
# R6 instead refines the normal INTERNALLY: a tiny BOUNDED MLP inside ASGRender
# predicts a per-Gaussian normal correction from the existing ASG latent
# (--normal_refine), and the working multi-view specular loss (root cause A)
# identifies the true normal (Ref-NeRF / 3DGS-DR principle). NO external files,
# NO extra disk, no densification changes. Zero-init + 0.3*tanh bound => starts as
# a no-op, can only gently refine (low-risk).
#
# Config = R3 (residual specular loss, the confirmed root-cause-A win) + normal_refine,
# so this isolates the disk-free normal refinement as the single change vs R3.
#
# Predicted if root cause B is fixable this way: NCC UP, sigma DOWN, energyRatio
# holds/improves — WITHOUT any disk cost.
# Verify the banner prints 'code: v2.9-...', 'normal_refine=True', and
# '[Specular] normal_refine ENABLED ...'.
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images
RUN=r6                      # keep run outputs separate for comparison
MODEL=${OUTPUT_ROOT}/${SCENE}_${RUN}

# 0. STALE-CODE GUARD — abort if this checkout lacks the normal-refine feature.
if ! grep -q "normal_refine" arguments/__init__.py; then
    echo "❌ STALE CODE: 'normal_refine' missing from arguments/__init__.py."
    echo "   This checkout is older than v2.9. Run 'git pull' before R6."
    exit 1
fi
echo "🔖 CODE_VERSION: $(grep -m1 '^CODE_VERSION' train.py | cut -d'\"' -f2)"
echo "🔖 RUN tag: ${RUN}"

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
    --spec_loss_mode residual \
    --normal_refine \
    --run_tag ${RUN}

# 2. RENDER
python render.py \
    -m ${MODEL} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${MODEL}
