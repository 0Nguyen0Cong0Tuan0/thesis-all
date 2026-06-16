#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN — R5  (v2.8)
# Hypothesis: the normal prior (root cause B) was UNDERPOWERED in v2.7-R4, not
# wrong. R4 (weight 0.05, gated at iter 7000) moved every placement metric the
# right way but only barely (NCC 0.524->0.532, sigma 4.82->4.80) — because by
# iter 7000 the geometry is already converged, so the cosine prior has little
# room to reshape it.
#
# Change vs R4: (1) raise the weight 0.05 -> 0.15, and (2) apply the prior EARLY
# (--normal_prior_start_iter 500) so it shapes geometry DURING densification.
# Everything else = R4 (= R3 residual specular loss). Isolates "stronger + earlier
# normal prior".
#
# Predicted if root cause B is fixable this way: NCC clearly UP, sigma DOWN,
# energyRatio holds/improves. If still flat, the Marigold prior signal itself is
# the limiter (switch estimator / rethink) rather than the weight/timing.
#
# PREREQUISITE — the normal priors must already exist (run once, see the notebook
# "R4 PREREQUISITE" cell): ./datasets/mipnerf360/counter/normals/*.npy
#
# NOTE: applying the prior from iter 500 means the extra normal render runs for
# ~29.5k iters (vs ~23k in R4), so expect this run to be SLOWER than R4's 117m.
# Verify the banner prints 'code: v2.8-...', 'normal_prior_weight=0.15',
# 'normal_prior_start_iter=500', and the one-time alignment-check cosine (>0).
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images
RUN=r5                      # keep run outputs separate for comparison
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
    --spec_loss_mode residual \
    --normal_prior_weight 0.15 \
    --normal_prior_dir normals \
    --normal_prior_start_iter 500

# 2. RENDER
python render.py \
    -m ${MODEL} \
    --skip_train

# 3. METRICS
python metrics.py \
    -m ${MODEL}
