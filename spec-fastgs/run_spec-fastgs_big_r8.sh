#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN — R8  (v3.1)  — PRECOMPUTED SHAFER SPECULAR LOCATOR
# R7 made densification specular-aware, but its locator (spec_frac, a MODEL residual
# comparing diffuse-vs-full renders) can't reliably isolate true specular from diffuse
# texture/geometry clutter on real Mip-NeRF scenes (counter A/B was neutral: NCC/energy-
# Ratio/sigma all flat, see results/MIPNERF_BOTTLENECK_PLAN_2026-07-01.md). R8 swaps that
# locator for a PRECOMPUTED, MODEL-INDEPENDENT one: Shafer's dichromatic reflection model
# (bright + desaturated) gated by a morphological white-top-hat blob filter (rejects flat-
# background / bright-diffuse false positives — validated: teapot 78.6%->2.0% flagged,
# no silhouette-edge artifact; counter 3.5%->1.7%, still lights up true glints). See
# tools/classical_specular_mask.py for the full derivation and tools/gen_shafer_priors.py
# for the offline sweep (pure CPU/scipy, no GPU, ~0.6s/image).
#
# Both mechanisms that make Gaussians "carry the specular load harder" now read the SAME
# precomputed prior:
#   --spec_loss_mode shafer         -> the specular LOSS mask (root cause A) uses it
#   --spec_densify_locator shafer   -> the DENSIFICATION vote (allocation) uses it
# Config = R7 (spec_densify on) + swap BOTH locators to shafer, isolating the locator
# source as the single change vs R7.
#
# PREREQUISITE — run the "Shafer priors" notebook cell (or manually) BEFORE this script:
#   python tools/gen_shafer_priors.py -s ./datasets/mipnerf360/counter -i images
# This needs NO GPU and works in the stock Kaggle python (numpy/scipy/PIL only) —
# unlike the monocular normal prior, it does NOT need to run before the conda teardown.
#
# Predicted: same or better than R7 (which was neutral on counter) — if NCC/sigma/
# energyRatio move now, the locator (not the mechanism) was the bottleneck; if still flat,
# the classical prior itself lacks signal on this scene and the geometry-gated Reflection
# Score (tools/extract_reflection_score.py) is the next thing to try.
# Verify the banner prints 'code: v3.1-...' and
# 'spec_densify=True (w=0.5, locator=shafer)'.
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images
RUN=r8
MODEL=${OUTPUT_ROOT}/${SCENE}_${RUN}

# 0. STALE-CODE GUARD — abort if this checkout lacks the shafer locator (v3.1).
if ! grep -q "spec_densify_locator" arguments/__init__.py; then
    echo "❌ STALE CODE: 'spec_densify_locator' missing from arguments/__init__.py."
    echo "   This checkout is older than v3.1. Run 'git pull' before R8."
    exit 1
fi
echo "🔖 CODE_VERSION: $(grep -m1 '^CODE_VERSION' train.py | cut -d'\"' -f2)"
echo "🔖 RUN tag: ${RUN}"

# 0.5 SAFETY CHECK — the priors must already exist (from the prereq sweep).
if ! ls ${DATA_ROOT}/${SCENE}/shafer_priors/*.png >/dev/null 2>&1; then
    echo "❌ No Shafer priors found at ${DATA_ROOT}/${SCENE}/shafer_priors/"
    echo "   Run: python tools/gen_shafer_priors.py -s ${DATA_ROOT}/${SCENE} -i ${IMAGES}"
    exit 1
fi

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
    --spec_loss_mode shafer \
    --shafer_prior_dir shafer_priors \
    --shafer_prior_thresh 0.3 \
    --spec_densify \
    --spec_densify_weight 0.5 \
    --spec_densify_locator shafer \
    --run_tag ${RUN}

# 2. RENDER
python render.py \
    -m ${MODEL} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${MODEL}
