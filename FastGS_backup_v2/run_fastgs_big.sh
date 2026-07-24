#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Default values
DEFAULT_DATASET="./datasets/mipnerf360/counter"
DEFAULT_SCALE="images_8"
DEFAULT_GPU="0"

DATASET_PATH=${1:-$DEFAULT_DATASET}
IMAGE_SCALE=${2:-$DEFAULT_SCALE}
GPU_ID=${3:-$DEFAULT_GPU}

# Extract scene name from the dataset path and append _big
BASE_SCENE_NAME=$(basename "$(realpath -m "$DATASET_PATH")")
SCENE_NAME="${BASE_SCENE_NAME}_big"

echo "========================================================================"
echo " Starting FastGS BIG Training Pipeline"
echo "========================================================================"
echo "Dataset Path : $DATASET_PATH"
echo "Image Scale  : $IMAGE_SCALE"
echo "GPU ID       : $GPU_ID"
echo "Scene Name   : $SCENE_NAME"
echo "Output Path  : output/$SCENE_NAME"
echo "========================================================================"

# Step 1: Run Training
# echo "[1/3] Running train.py..."
# CUDA_VISIBLE_DEVICES=$GPU_ID OAR_JOB_ID=$SCENE_NAME python train.py \
#     -s "$DATASET_PATH" \
#     -i "$IMAGE_SCALE" \
#     --eval \
#     --densification_interval 100 \
#     --optimizer_type default \
#     --test_iterations 30000 \
#     --highfeature_lr 0.02 \
#     --grad_abs_thresh 0.0004

# Step 2: Run Rendering
echo "[2/3] Running render.py..."
CUDA_VISIBLE_DEVICES=$GPU_ID python render.py -m "output/$SCENE_NAME" --skip_train

# Step 3: Run Metrics Evaluation
echo "[3/3] Running metrics.py..."
CUDA_VISIBLE_DEVICES=$GPU_ID python metrics.py -m "output/$SCENE_NAME"

echo "========================================================================"
echo " Pipeline Completed Successfully!"
echo " Results and train_info.json are saved in output/$SCENE_NAME"
echo "========================================================================"
