#!/bin/bash
conda run -n fastgs_env python train.py -s ./datasets/Ref-NeRF/refnerf/toaster -m ./output/test_asg --iterations 500 --eval
