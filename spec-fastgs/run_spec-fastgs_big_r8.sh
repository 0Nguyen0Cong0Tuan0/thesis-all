#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN — R8  (v3.2)  — PRECOMPUTED TAN-IKEUCHI SPECULAR LOCATOR
# R7 made densification specular-aware, but its locator (spec_frac, a MODEL residual
# comparing diffuse-vs-full renders) can't reliably isolate true specular from diffuse
# texture/geometry clutter on real Mip-NeRF scenes (counter A/B was neutral: NCC/energy-
# Ratio/sigma all flat, see results/MIPNERF_BOTTLENECK_PLAN_2026-07-01.md). R8 swaps that
# locator for a PRECOMPUTED, MODEL-INDEPENDENT one: Tan & Ikeuchi's (PAMI 2005) specular-
# pedestal score (per-pixel minimum channel), gated by a morphological white-top-hat blob
# filter PLUS a near-saturation recovery condition (Algorithm 2b from
# test_specular_algorithms_comparison.ipynb — fixes a top-hat false-negative on spatially
# broad highlights that a v3.1-era Shafer-based locator did not need). See
# tools/classical_specular_mask.py for the full derivation, tools/gen_tanikeuchi_priors.py
# for the offline sweep (pure CPU/scipy, no GPU, well under 1s/image).
#
# KNOWN TRADEOFF (validated via full 240-image production-resolution sweeps on counter):
# this locator flags substantially MORE of the image as candidate-specular than the prior
# Shafer implementation did (mean 7.89% vs Shafer's 1.44%, max 17.96% vs 4.22%) — Imin
# alone lacks Shafer's saturation condition, which independently helped reject colored-
# but-bright diffuse pixels. Switched anyway per project decision after visually reviewing
# the comparison notebook. WATCH Gaussian count / PSNR-SSIM for the same kind of
# regression the v2.5-R1 luminance mask caused (a looser locator can dilute the specular
# loss/densification signal with non-specular pixels) — if this run regresses, that is the
# most likely cause, and the fix (a stricter tanikeuchi_prior_thresh) is a one-line change.
#
# Both mechanisms that make Gaussians "carry the specular load harder" now read the SAME
# precomputed prior:
#   --spec_loss_mode tanikeuchi         -> the specular LOSS mask (root cause A) uses it
#   --spec_densify_locator tanikeuchi   -> the DENSIFICATION vote (allocation) uses it
# Config = R7 (spec_densify on) + swap BOTH locators to tanikeuchi, isolating the locator
# source as the single change vs R7.
#
# PREREQUISITE — run the "Tan-Ikeuchi priors" notebook cell (or manually) BEFORE this script:
#   python tools/gen_tanikeuchi_priors.py -s ./datasets/mipnerf360/counter -i images
# This needs NO GPU and works in the stock Kaggle python (numpy/scipy/PIL only) —
# unlike the monocular normal prior, it does NOT need to run before the conda teardown.
#
# Predicted: same or better than R7 (which was neutral on counter) — if NCC/sigma/
# energyRatio move now, the locator (not the mechanism) was the bottleneck; if still flat,
# the classical prior itself lacks signal on this scene and the geometry-gated Reflection
# Score (tools/extract_reflection_score.py) is the next thing to try.
# Verify the banner prints 'code: v3.2-...' and
# 'spec_densify=True (w=0.5, locator=tanikeuchi)'.
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images
RUN=r8
MODEL=${OUTPUT_ROOT}/${SCENE}_${RUN}

# 0. STALE-CODE GUARD — abort if this checkout lacks the tanikeuchi locator (v3.2).
if ! grep -q "tanikeuchi_prior_dir" arguments/__init__.py; then
    echo "❌ STALE CODE: 'tanikeuchi_prior_dir' missing from arguments/__init__.py."
    echo "   This checkout is older than v3.2. Run 'git pull' before R8."
    exit 1
fi
echo "🔖 CODE_VERSION: $(grep -m1 '^CODE_VERSION' train.py | cut -d'\"' -f2)"
echo "🔖 RUN tag: ${RUN}"

# 0.5 SAFETY CHECK — the priors must already exist (from the prereq sweep).
if ! ls ${DATA_ROOT}/${SCENE}/tanikeuchi_priors/*.png >/dev/null 2>&1; then
    echo "❌ No Tan-Ikeuchi priors found at ${DATA_ROOT}/${SCENE}/tanikeuchi_priors/"
    echo "   Run: python tools/gen_tanikeuchi_priors.py -s ${DATA_ROOT}/${SCENE} -i ${IMAGES}"
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
    --spec_loss_mode tanikeuchi \
    --tanikeuchi_prior_dir tanikeuchi_priors \
    --tanikeuchi_prior_thresh 0.3 \
    --spec_densify \
    --spec_densify_weight 0.5 \
    --spec_densify_locator tanikeuchi \
    --run_tag ${RUN}

# 2. RENDER
python render.py \
    -m ${MODEL} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${MODEL}
