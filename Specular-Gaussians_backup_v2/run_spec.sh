#!/bin/bash

# Kích hoạt môi trường conda (bỏ dấu comment bên dưới nếu cần)
# conda activate spec-gaussian-env

HYBRID=True

# ==============================================================================
# CÁCH 1: Đọc trực tiếp từ thư mục (Khuyên dùng - Nhanh, tiết kiệm RAM/CPU)
# ==============================================================================

# 1.1 Train không dùng Anchor (Chất lượng kết quả tốt nhất)
# python train.py \
#     -s data/mipnerf-360/counter \
#     -m outputs/mip360/counter_images_8 \
#     --images images_8 \
#     -r 1 \
#     --eval \
#     --is_real \
#     --is_indoor \
#     --asg_degree 12 \
#     --hybrid ${HYBRID}

# 1.1.2 Render sau khi train xong để lấy folder "spec"
python render.py \
    -m outputs/mip360/counter_images_8 \
    --skip_train \
    --hybrid ${HYBRID}

# 1.1.3 Đánh giá điểm PSNR, SSIM, LPIPS
python metrics.py \
    -m outputs/mip360/counter_images_8
# 1.2 Train có dùng Anchor (Tốc độ train và render nhanh hơn)
# python train_anchor.py \
#     -s data/mipnerf-360/counter \
#     -m outputs/mip360/counter_images_8_anchor \
#     --images images_8 \
#     -r 1 \
#     --eval \
#     --voxel_size 0.001 \
#     --update_init_factor 16 \
#     --iterations 30000 \
#     --use_c2f


# ==============================================================================
# CÁCH 2: Load từ thư mục images gốc và tự động downscale bằng tham số -r 4
# ==============================================================================

# 2.1 Train không dùng Anchor
# python train.py \
#     -s data/mipnerf-360/counter \
#     -m outputs/mip360/counter_r4 \
#     -r 4 \
#     --eval \
#     --is_real \
#     --is_indoor \
#     --asg_degree 12

# 2.2 Train có dùng Anchor
# python train_anchor.py \
#     -s data/mipnerf-360/counter \
#     -m outputs/mip360/counter_r4_anchor \
#     -r 4 \
#     --eval \
#     --voxel_size 0.001 \
#     --update_init_factor 16 \
#     --iterations 30000 \
#     --use_c2f
