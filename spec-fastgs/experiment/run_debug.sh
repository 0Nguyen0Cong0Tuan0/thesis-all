#!/bin/bash
# Usage: ./run_debug.sh [camera_name]
# Example: ./run_debug.sh train_r_0

CAM_NAME=${1:-train_r_0}
echo "Running debug for camera: $CAM_NAME"

conda run -n fastgs_env python experiment/debug_ref_score.py \
    -s ../Specular-Gaussians/data/Ref-NeRF/refnerf/toaster \
    -m output/toaster \
    --sh_degree 0 \
    --target_cam $CAM_NAME
