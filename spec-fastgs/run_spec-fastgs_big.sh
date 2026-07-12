#!/bin/bash

# ============================================================
# SPEC-FASTGS BIG RUN SCRIPT
# ============================================================

export CUDA_VISIBLE_DEVICES=0

# ------------------------------------------------------------
# Dataset / Output
# ------------------------------------------------------------
DATA_ROOT=./datasets/mipnerf360
OUTPUT_ROOT=./output
SCENE=counter
IMAGES=images_8

# ------------------------------------------------------------
# Core Training / Representation
# ------------------------------------------------------------
ASG_DEGREE=64
ASG_NUM_THETA=-1
ASG_NUM_PHI=-1
SPECULAR_HIDDEN=-1
SPECULAR_LAYERS=-1
REAL_USE_REFLECTION_DIR=False
NUM_SCORE_CAMERAS=10

# ------------------------------------------------------------
# Reflection Prior Extraction
# ------------------------------------------------------------
EXTRACT_REF_PRIOR=True
BACKUP_REF_PRIOR=True
REF_PRIOR_METHOD=tan
TI_THRESH=0.35
TI_BRIGHT=0.6
SK_INTENSITY=0.65
SK_SATURATION=0.3

# ------------------------------------------------------------
# Geometry Coverage Ablation - Shared Switches
# ------------------------------------------------------------
USE_REF_SCORE=True
USE_ADAPTIVE_PRIOR=True

# ------------------------------------------------------------
# Geometry Coverage B1 - Scene-Relative RefScore Budget
#   -1 means auto budget from initial Gaussian count.
# ------------------------------------------------------------
MAX_REFSCORE_GAUSSIANS=-1
REFSCORE_BUDGET_MULTIPLIER=10.0
REFSCORE_BUDGET_MIN=200000
REFSCORE_BUDGET_MAX=1000000

# ------------------------------------------------------------
# Geometry Coverage B2 - Soft RefScore Decay
# ------------------------------------------------------------
REFSCORE_DECAY_POWER=1.0
REFSCORE_MIN_STRENGTH=0.15
REFSCORE_THRESHOLD_MIN=0.5
REFSCORE_THRESHOLD_MAX=0.9

# ------------------------------------------------------------
# Geometry Coverage A - Residual-Adaptive Prior
#   Active only when USE_ADAPTIVE_PRIOR=True.
# ------------------------------------------------------------
ADAPTIVE_PRIOR_START=5000
ADAPTIVE_PRIOR_INTERVAL=3000
ADAPTIVE_PRIOR_NUM_CAMERAS=20
ADAPTIVE_PRIOR_EMA=0.7

# ------------------------------------------------------------
# ASG / SH Scheduling Ablation
# ------------------------------------------------------------
FULL_ASG_INTERVAL=0
F_REST_WARMUP_UNTIL=0
F_REST_INTERVAL_EARLY=16
F_REST_INTERVAL_MID=32
F_REST_INTERVAL_LATE=64

# ------------------------------------------------------------
# Representation Capacity R1/R2 - SH/ASG Role Separation
# ------------------------------------------------------------
USE_SH_SPEC_MASK=False
SH_SPEC_GRAD_SCALE=0.75
SH_SPEC_MASK_START=8000
SH_SPEC_MASK_THRESHOLD=0.75
SH_SPEC_MIN_METRIC_COUNT=2

USE_ASG_RESIDUAL_SUPERVISION=False
LAMBDA_ASG_RESIDUAL=0.0
LAMBDA_ASG_LEAK=0.0
ASG_RESIDUAL_START=8000
ASG_RESIDUAL_INTERVAL=16
ASG_RESIDUAL_REF_THRESHOLD=0.75

# ------------------------------------------------------------
# Supervision Signal S - Specular-weighted loss / ASG regularizer
# ------------------------------------------------------------
LAMBDA_SPEC_L1_WEIGHT=0.0
LAMBDA_SPEC_REG=0.0

# ------------------------------------------------------------
# Normal Quality N - Learned delta / Gaussian-space smoothness
# ------------------------------------------------------------
USE_NORMAL_DELTA=False
NORMAL_DELTA_LR=0.00005
NORMAL_DELTA_START_ITER=3000
NORMAL_DELTA_MAX_NORM=0.05
LAMBDA_NORMAL_DELTA_REG=0.001
LAMBDA_NORMAL_SMOOTH=0.0001
NORMAL_SMOOTH_START_ITER=8000
NORMAL_SMOOTH_INTERVAL=16
NORMAL_SMOOTH_MAX_POINTS=2048
NORMAL_SMOOTH_K=8
NORMAL_SMOOTH_USE_REF_MASK=False

REF_SCORE_FLAG=""
if [ "$USE_REF_SCORE" = "True" ]; then
    REF_SCORE_FLAG="--use_ref_score"
fi

ADAPTIVE_PRIOR_FLAG=""
if [ "$USE_ADAPTIVE_PRIOR" = "True" ]; then
    ADAPTIVE_PRIOR_FLAG="--use_adaptive_prior"
fi

SH_SPEC_MASK_FLAG=""
if [ "$USE_SH_SPEC_MASK" = "True" ]; then
    SH_SPEC_MASK_FLAG="--use_sh_spec_mask"
fi

ASG_RESIDUAL_FLAG=""
if [ "$USE_ASG_RESIDUAL_SUPERVISION" = "True" ]; then
    ASG_RESIDUAL_FLAG="--use_asg_residual_supervision"
fi

REAL_REFLECTION_FLAG=""
if [ "$REAL_USE_REFLECTION_DIR" = "True" ]; then
    REAL_REFLECTION_FLAG="--real_use_reflection_dir"
fi

NORMAL_DELTA_FLAG=""
if [ "$USE_NORMAL_DELTA" = "True" ]; then
    NORMAL_DELTA_FLAG="--use_normal_delta"
fi

NORMAL_SMOOTH_REF_FLAG=""
if [ "$NORMAL_SMOOTH_USE_REF_MASK" = "True" ]; then
    NORMAL_SMOOTH_REF_FLAG="--normal_smooth_use_ref_mask"
fi

echo "========================================================================"
echo " Starting spec-fastgs BIG Training Pipeline"
echo "========================================================================"
echo "Dataset Path : ${DATA_ROOT}/${SCENE}"
echo "Scene Name   : $SCENE"
echo "Output Path  : ${OUTPUT_ROOT}/${SCENE}"
echo "Use RefScore : ${USE_REF_SCORE}"
echo "AdaptivePrior: ${USE_ADAPTIVE_PRIOR}"
echo "SH Spec Mask : ${USE_SH_SPEC_MASK}"
echo "Normal Delta : ${USE_NORMAL_DELTA}"
echo "========================================================================"

# 0. EXTRACT REFLECTION PRIOR
if [ "$USE_REF_SCORE" = "True" ] && [ "$EXTRACT_REF_PRIOR" = "True" ]; then
    echo "[0/4] Running extract_reflection_prior.py..."
    PRIOR_DIR=${DATA_ROOT}/${SCENE}/reflection_prior
    if [ "$BACKUP_REF_PRIOR" = "True" ] && [ -d "$PRIOR_DIR" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BACKUP_ROOT=${DATA_ROOT}/${SCENE}/backups
        BACKUP_DIR=${BACKUP_ROOT}/reflection_prior_${TIMESTAMP}
        echo "Backing up existing reflection_prior to:"
        echo "  ${BACKUP_DIR}"
        mkdir -p "$BACKUP_ROOT"
        mv "$PRIOR_DIR" "$BACKUP_DIR"
    fi
    python extract_reflection_prior.py \
        -s ${DATA_ROOT}/${SCENE} \
        -i ${IMAGES} \
        --ref_prior_method ${REF_PRIOR_METHOD} \
        --ti_thresh ${TI_THRESH} \
        --ti_bright ${TI_BRIGHT} \
        --sk_intensity ${SK_INTENSITY} \
        --sk_saturation ${SK_SATURATION}
else
    echo "[0/4] Skipping extract_reflection_prior.py"
fi

# 1. TRAIN
echo "[1/4] Running train.py..."
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_ROOT}/${SCENE} \
    -i ${IMAGES} \
    --eval \
    --iterations 30000 \
    --densification_interval 100 \
    --optimizer_type default \
    --asg_degree ${ASG_DEGREE} \
    --asg_num_theta ${ASG_NUM_THETA} \
    --asg_num_phi ${ASG_NUM_PHI} \
    --specular_hidden ${SPECULAR_HIDDEN} \
    --specular_layers ${SPECULAR_LAYERS} \
    --is_real \
    --is_indoor \
    --sh_degree 3 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0004 \
    --specular_start_iter 3000 \
    --num_score_cameras ${NUM_SCORE_CAMERAS} \
    --max_refscore_gaussians ${MAX_REFSCORE_GAUSSIANS} \
    --refscore_budget_multiplier ${REFSCORE_BUDGET_MULTIPLIER} \
    --refscore_budget_min ${REFSCORE_BUDGET_MIN} \
    --refscore_budget_max ${REFSCORE_BUDGET_MAX} \
    --refscore_decay_power ${REFSCORE_DECAY_POWER} \
    --refscore_min_strength ${REFSCORE_MIN_STRENGTH} \
    --refscore_threshold_min ${REFSCORE_THRESHOLD_MIN} \
    --refscore_threshold_max ${REFSCORE_THRESHOLD_MAX} \
    --adaptive_prior_start ${ADAPTIVE_PRIOR_START} \
    --adaptive_prior_interval ${ADAPTIVE_PRIOR_INTERVAL} \
    --adaptive_prior_num_cameras ${ADAPTIVE_PRIOR_NUM_CAMERAS} \
    --adaptive_prior_ema ${ADAPTIVE_PRIOR_EMA} \
    --full_asg_interval ${FULL_ASG_INTERVAL} \
    --f_rest_warmup_until ${F_REST_WARMUP_UNTIL} \
    --f_rest_interval_early ${F_REST_INTERVAL_EARLY} \
    --f_rest_interval_mid ${F_REST_INTERVAL_MID} \
    --f_rest_interval_late ${F_REST_INTERVAL_LATE} \
    --sh_spec_mask_threshold ${SH_SPEC_MASK_THRESHOLD} \
    --sh_spec_grad_scale ${SH_SPEC_GRAD_SCALE} \
    --sh_spec_mask_start ${SH_SPEC_MASK_START} \
    --sh_spec_min_metric_count ${SH_SPEC_MIN_METRIC_COUNT} \
    --lambda_asg_residual ${LAMBDA_ASG_RESIDUAL} \
    --lambda_asg_leak ${LAMBDA_ASG_LEAK} \
    --asg_residual_start ${ASG_RESIDUAL_START} \
    --asg_residual_interval ${ASG_RESIDUAL_INTERVAL} \
    --asg_residual_ref_threshold ${ASG_RESIDUAL_REF_THRESHOLD} \
    --lambda_spec_l1_weight ${LAMBDA_SPEC_L1_WEIGHT} \
    --lambda_spec_reg ${LAMBDA_SPEC_REG} \
    --normal_delta_lr ${NORMAL_DELTA_LR} \
    --normal_delta_start_iter ${NORMAL_DELTA_START_ITER} \
    --normal_delta_max_norm ${NORMAL_DELTA_MAX_NORM} \
    --lambda_normal_delta_reg ${LAMBDA_NORMAL_DELTA_REG} \
    --lambda_normal_smooth ${LAMBDA_NORMAL_SMOOTH} \
    --normal_smooth_start_iter ${NORMAL_SMOOTH_START_ITER} \
    --normal_smooth_interval ${NORMAL_SMOOTH_INTERVAL} \
    --normal_smooth_max_points ${NORMAL_SMOOTH_MAX_POINTS} \
    --normal_smooth_k ${NORMAL_SMOOTH_K} \
    --ref_prior_method ${REF_PRIOR_METHOD} \
    --ti_thresh ${TI_THRESH} \
    --ti_bright ${TI_BRIGHT} \
    --sk_intensity ${SK_INTENSITY} \
    --sk_saturation ${SK_SATURATION} \
    ${REF_SCORE_FLAG} \
    ${ADAPTIVE_PRIOR_FLAG} \
    ${SH_SPEC_MASK_FLAG} \
    ${ASG_RESIDUAL_FLAG} \
    ${REAL_REFLECTION_FLAG} \
    ${NORMAL_DELTA_FLAG} \
    ${NORMAL_SMOOTH_REF_FLAG}

# 2. RENDER
echo "[2/4] Running render.py..."
python render.py \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --skip_train

# 3. METRICS
echo "[3/4] Running metrics.py..."
python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}
