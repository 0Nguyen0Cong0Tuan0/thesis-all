#!/bin/bash

# Kích hoạt môi trường conda (bỏ dấu comment bên dưới nếu cần)
# conda activate spec-gaussian-env

HYBRID=True

# 1. Train tập teapot (Synthetic Dataset)
python train.py \
    -s data/Ref-NeRF/refnerf/toaster \
    -m outputs/refnerf/toaster \
    --eval \
    --white_background \
    --asg_degree 24 \
    --hybrid ${HYBRID}

# 2. Render ảnh sau khi train
python render.py \
    -m outputs/refnerf/toaster \
    --skip_train \
    --hybrid ${HYBRID}

# 3. Đánh giá điểm PSNR, SSIM, LPIPS
python metrics.py \
    -m outputs/refnerf/toaster
