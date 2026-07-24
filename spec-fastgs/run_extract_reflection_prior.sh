#!/bin/bash

# ============================================================
# REFLECTION PRIOR EXTRACTION SCRIPT
# ============================================================
# Edit the variables below, then run:
#   bash run_extract_reflection_prior.sh
#
# Output is written to:
#   ${DATA_ROOT}/${SCENE}/reflection_prior
# ============================================================

export CUDA_VISIBLE_DEVICES=0

# Dataset config
DATA_ROOT=./datasets/mipnerf360
SCENE=counter
IMAGES=images_8

# Prior method:
#   tan    = Tan-Ikeuchi-style prior, older spec-fastgs behavior
#   shafer = Shafer/Klinker-style prior
#   hybrid = softer confidence prior combining several highlight cues
# REF_PRIOR_METHOD=shafer
REF_PRIOR_METHOD=tan

# Tan-Ikeuchi parameters
TI_THRESH=0.35
TI_BRIGHT=0.6

# Shafer/Klinker parameters
SK_INTENSITY=0.7
SK_SATURATION=0.2

# Conservative confidence post-processing.
# Keep these as identity values to reproduce older extracted priors.
REF_CONF_GAMMA=1.0
REF_CONF_QUANTILE=0.0
REF_CONF_SMOOTH_RADIUS=0

# Backup existing reflection_prior before overwrite.
# Set to False if you intentionally want direct overwrite.
BACKUP_EXISTING=True

SOURCE_PATH=${DATA_ROOT}/${SCENE}
PRIOR_DIR=${SOURCE_PATH}/reflection_prior

echo "========================================================================"
echo " Extract Reflection Prior"
echo "========================================================================"
echo "Dataset Path : ${SOURCE_PATH}"
echo "Images       : ${IMAGES}"
echo "Output Path  : ${PRIOR_DIR}"
echo "Method       : ${REF_PRIOR_METHOD}"
if [ "$REF_PRIOR_METHOD" = "tan" ]; then
    echo "ti_thresh    : ${TI_THRESH}"
    echo "ti_bright    : ${TI_BRIGHT}"
elif [ "$REF_PRIOR_METHOD" = "shafer" ]; then
    echo "sk_intensity : ${SK_INTENSITY}"
    echo "sk_saturation: ${SK_SATURATION}"
elif [ "$REF_PRIOR_METHOD" = "hybrid" ]; then
    echo "ti_thresh    : ${TI_THRESH}"
    echo "ti_bright    : ${TI_BRIGHT}"
    echo "sk_intensity : ${SK_INTENSITY}"
    echo "sk_saturation: ${SK_SATURATION}"
else
    echo "ERROR: REF_PRIOR_METHOD must be 'tan', 'shafer', or 'hybrid'."
    exit 1
fi
echo "conf_gamma   : ${REF_CONF_GAMMA}"
echo "conf_quantile: ${REF_CONF_QUANTILE}"
echo "smooth_radius: ${REF_CONF_SMOOTH_RADIUS}"
echo "========================================================================"

if [ "$BACKUP_EXISTING" = "True" ] && [ -d "$PRIOR_DIR" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_ROOT=${SOURCE_PATH}/backups
    BACKUP_DIR=${BACKUP_ROOT}/reflection_prior_${TIMESTAMP}
    echo "Backing up existing reflection_prior to:"
    echo "  ${BACKUP_DIR}"
    mkdir -p "$BACKUP_ROOT"
    mv "$PRIOR_DIR" "$BACKUP_DIR"
fi

python extract_reflection_prior.py \
    -s ${SOURCE_PATH} \
    -i ${IMAGES} \
    --ref_prior_method ${REF_PRIOR_METHOD} \
    --ti_thresh ${TI_THRESH} \
    --ti_bright ${TI_BRIGHT} \
    --sk_intensity ${SK_INTENSITY} \
    --sk_saturation ${SK_SATURATION} \
    --ref_conf_gamma ${REF_CONF_GAMMA} \
    --ref_conf_quantile ${REF_CONF_QUANTILE} \
    --ref_conf_smooth_radius ${REF_CONF_SMOOTH_RADIUS}

echo "========================================================================"
echo " Reflection prior extraction completed."
echo " Saved to: ${PRIOR_DIR}"
echo "========================================================================"
