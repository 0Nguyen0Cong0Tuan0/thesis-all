#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Default values
DATASET_PATH="../Specular-Gaussians/data/Ref-NeRF/refnerf/toaster"
SCENE_NAME="toaster"
GPU_ID="0"

echo "========================================================================"
echo " Starting FastGS Training Pipeline for Synthetic Dataset"
echo "========================================================================"
echo "Dataset Path : $DATASET_PATH"
echo "GPU ID       : $GPU_ID"
echo "Scene Name   : $SCENE_NAME"
echo "Output Path  : output/$SCENE_NAME"
echo "========================================================================"

# Step 1: Run Training
echo "[1/3] Running train.py..."
CUDA_VISIBLE_DEVICES=$GPU_ID OAR_JOB_ID=$SCENE_NAME python train.py \
    -s "$DATASET_PATH" \
    --eval \
    --white_background \
    --densification_interval 500 \
    --optimizer_type default \
    --test_iterations 30000 \
    --highfeature_lr 0.02 \
    --grad_abs_thresh 0.0008

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
