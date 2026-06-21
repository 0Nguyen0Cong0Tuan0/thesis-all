#!/bin/bash

# ============================================================
# SPEC-FASTGS — SYNTHETIC SCENE RUNNER (Spec-Gaussian eval suite)
# Generalised, scene-path-agnostic runner for the Anisotropic-Synthetic and
# Synthetic_NSVF datasets. Format (Blender transforms.json / NSVF pose+rgb+
# intrinsics) is auto-detected by Scene (scene/__init__.py).
#
# Synthetic config differs from the counter (real) runs:
#   * --white_background   (synthetic alpha-composited on white)
#   * NO --is_real/--is_indoor  (synthetic uses SpecularNetwork, not the Real variant)
#   * --specular_start_iter 3000 (Spec-Gaussian's synthetic schedule; gives the
#     specular branch + spec_densify a proper window inside densification 500-15000)
#
# IMPORTANT: <scene_path> must be on a WRITABLE fs (e.g. /kaggle/working) because the
# loader writes points3d.ply into it. Do NOT point at read-only /kaggle/input.
#
# Usage:
#   bash run_spec-fastgs_synthetic.sh <scene_path> <run_tag> [extra train args...]
# Examples (A/B for the spec_densify ablation):
#   bash run_spec-fastgs_synthetic.sh ./datasets/Anisotropic-Synthetic-Dataset/teapot base
#   bash run_spec-fastgs_synthetic.sh ./datasets/Anisotropic-Synthetic-Dataset/teapot r7 --spec_densify
# ============================================================

export CUDA_VISIBLE_DEVICES=0
set -e

SCENE_PATH="$1"
RUN="$2"
shift 2
EXTRA="$@"

if [ -z "$SCENE_PATH" ] || [ -z "$RUN" ]; then
    echo "Usage: bash run_spec-fastgs_synthetic.sh <scene_path> <run_tag> [extra args]"
    exit 1
fi

SCENE_NAME=$(basename "$SCENE_PATH")
MODEL=./output/${SCENE_NAME}_${RUN}

# stale-code guard — spec_densify is the v3.0 token (and the loaders ship with it)
if ! grep -q "spec_densify" arguments/__init__.py; then
    echo "❌ STALE CODE: 'spec_densify' missing — git pull (need v3.0)."
    exit 1
fi
echo "🔖 CODE_VERSION: $(grep -m1 '^CODE_VERSION' train.py | cut -d'\"' -f2)"
echo "🔖 scene=${SCENE_NAME} | run=${RUN} | extra='${EXTRA}'"

python train.py \
    -s "${SCENE_PATH}" \
    -m "${MODEL}" \
    --eval \
    --white_background \
    --iterations 30000 \
    --densification_interval 100 \
    --optimizer_type default \
    --asg_degree 24 \
    --sh_degree 3 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0004 \
    --specular_start_iter 3000 \
    --spec_loss_weight 0.15 \
    --spec_loss_quantile 0.95 \
    --spec_loss_mode residual \
    --run_tag "${RUN}" \
    ${EXTRA}

python render.py -m "${MODEL}" --skip_train
python metrics.py -m "${MODEL}"
