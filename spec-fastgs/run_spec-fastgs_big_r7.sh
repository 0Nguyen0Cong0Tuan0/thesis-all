#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN — R7  (v3.0)  — SPECULAR-AWARE DENSIFICATION (CVPR contribution)
# FastGS's multi-view consistency vote is Lambertian-biased: specular highlights are
# view-inconsistent, so the vote either fakes them with diffuse geometry (Gaussian
# bloat) or under-resolves them. R7 makes the vote specular-aware via a diffuse-vs-full
# residual decomposition in fast_utils.compute_gaussian_score_fastgs:
#   (1) residual-gated vote   -> geometric densify only on error NOT explained by the
#       ASG branch (keeps bright-diffuse under-fit; drops specular the branch handles),
#   (2) specular-deficit alloc-> ALSO clone where specular is present but under-resolved
#       (weight spec_densify_weight), anchoring sharper highlights.
# This reframes specular PLACEMENT (root cause B, which normal supervision R4-R6 could
# not fix) as an ALLOCATION problem.
#
# Config = R3 (residual specular loss, the confirmed root-cause-A win) + --spec_densify,
# isolating specular-aware densification as the single change vs R3.
# Active window: densification iters 7000-15000 (where the specular branch is live).
#
# Predicted: at EQUAL or LOWER #Gaussians, higher specular fidelity (NCC up, sigma down,
# energyRatio up) and no highlight bloat. NOTE: counter is diffuse-dominant (~5% specular
# px) so the win may be small here — the real test is the specular benchmark sweep (P3).
# Verify the banner prints 'code: v3.0-...' and 'spec_densify=True (w=0.5)'.
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images
RUN=r7
MODEL=${OUTPUT_ROOT}/${SCENE}_${RUN}

# 0. STALE-CODE GUARD — abort if this checkout lacks specular-aware densification.
if ! grep -q "spec_densify" arguments/__init__.py; then
    echo "❌ STALE CODE: 'spec_densify' missing from arguments/__init__.py."
    echo "   This checkout is older than v3.0. Run 'git pull' before R7."
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
    --spec_densify \
    --spec_densify_weight 0.5 \
    --run_tag ${RUN}

# 2. RENDER
python render.py \
    -m ${MODEL} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${MODEL}
