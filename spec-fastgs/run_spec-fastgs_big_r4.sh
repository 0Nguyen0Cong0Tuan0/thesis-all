#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN — R4  (v2.7)
# Hypothesis: root cause B (noisy min-axis normals). v2.6-R3 proved the residual
# specular loss fixes MAGNITUDE (energyRatio 0.388->0.461, gain a*->1.03) but left
# PLACEMENT flat (NCC 0.526, structural% 98.9%, sigma 4.78). That residual is the
# fingerprint of wrong per-Gaussian normals feeding the specular MLP.
#
# Change vs R3: keep R3's residual specular loss, ADD the monocular normal prior
# (DN-Splatter style) — a cosine loss between the rendered Gaussian normal and a
# precomputed monocular normal map, which reshapes geometry toward true surface
# orientation. This isolates whether better normals lift NCC/sigma (placement).
#
# Predicted if the hypothesis holds: NCC UP, sigma DOWN (sharper highlights),
# energyRatio holds or improves.
#
# PREREQUISITE — run ONCE before this script (generates ./datasets/.../counter/normals):
#   pip install -q diffusers transformers accelerate
#   python tools/gen_normal_priors.py -s ./datasets/mipnerf360/counter -i images --model marigold
#
# Verify the startup banner prints "code: v2.7-..." and
# "normal_prior_weight=0.05", plus the one-time "[normal-prior] alignment check:
# mean cos(render,prior)=+0.xxx" (must be POSITIVE; if strongly negative, add
# --normal_prior_flip to the train command and rerun).
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images
RUN=r4                      # keep run outputs separate for comparison
MODEL=${OUTPUT_ROOT}/${SCENE}_${RUN}

# 0. STALE-CODE GUARD — abort if this checkout lacks the normal-prior feature.
if ! grep -q "normal_prior_weight" arguments/__init__.py; then
    echo "❌ STALE CODE: 'normal_prior_weight' missing from arguments/__init__.py."
    echo "   This checkout is older than v2.7. Run 'git pull' before R4."
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
    --normal_prior_weight 0.05 \
    --normal_prior_dir normals \
    --run_tag ${RUN}

# 2. RENDER
python render.py \
    -m ${MODEL} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${MODEL}
