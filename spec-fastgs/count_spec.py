import torch
from plyfile import PlyData
import numpy as np
import sys

ply_path = "output/toaster/point_cloud/iteration_30000/point_cloud.ply"
try:
    plydata = PlyData.read(ply_path)
    if "is_specular" in plydata.elements[0].data.dtype.names:
        is_specular = np.asarray(plydata.elements[0]["is_specular"])
        print(f"Total Gaussians: {len(is_specular)}")
        print(f"Specular Gaussians (is_specular=True): {np.sum(is_specular)}")
    else:
        print("is_specular attribute not found in PLY file.")
except Exception as e:
    print(e)
