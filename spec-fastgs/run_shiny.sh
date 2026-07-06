#!/bin/bash

# ============================================================
# SPEC-FASTGS TEAPOT (SYNTHETIC) RUN SCRIPT
# ============================================================

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./datasets/Ref-NeRF/refnerf
OUTPUT_ROOT=./output
SCENE=toaster

echo "========================================================================"
echo " Starting spec-fastgs Training Pipeline for Synthetic Dataset"
echo "========================================================================"
echo "Dataset Path : ${DATA_ROOT}/${SCENE}"
echo "Scene Name   : $SCENE"
echo "Output Path  : ${OUTPUT_ROOT}/${SCENE}"
echo "========================================================================"

# 1. TRAIN (Giai đoạn 1)
echo "[1/3] Running train.py..."
python train.py \
    -s ${DATA_ROOT}/${SCENE} \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --eval \
    --white_background \
    --iterations 30000 \
    --densification_interval 500 \
    --optimizer_type default \
    --asg_degree 24 \
    --sh_degree 3 \
    --specular_start_iter 3000

# 2. RENDER (Giai đoạn 2)
echo "[2/3] Running render.py..."
python render.py \
    -m ${OUTPUT_ROOT}/${SCENE} \
    --skip_train

# 3. METRICS (Giai đoạn 3)
echo "[3/3] Running metrics.py..."
python metrics.py \
    -m ${OUTPUT_ROOT}/${SCENE}

echo "========================================================================"
echo " Pipeline Completed Successfully!"
echo " Results and train_info.json are saved in ${OUTPUT_ROOT}/${SCENE}"
echo "========================================================================"
