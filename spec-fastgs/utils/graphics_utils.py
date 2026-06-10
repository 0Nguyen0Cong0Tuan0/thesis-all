# ============================================================
# Graphics Utils (Final)
# ============================================================

import torch
import math
import numpy as np
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
	points: np.array
	colors: np.array
	normals: np.array


def geom_transform_points(points, transf_matrix):
	P, _ = points.shape
	ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
	points_hom = torch.cat([points, ones], dim=1)

	points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
	denom = points_out[..., 3:] + 1e-7

	return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
	Rt = np.zeros((4, 4))
	Rt[:3, :3] = R.transpose()
	Rt[:3, 3] = t
	Rt[3, 3] = 1.0
	return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0, 0, 0]), scale=1.0):
	Rt = getWorld2View(R, t)

	C2W = np.linalg.inv(Rt)
	cam_center = (C2W[:3, 3] + translate) * scale

	C2W[:3, 3] = cam_center
	Rt = np.linalg.inv(C2W)

	return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
	tanY = math.tan(fovY / 2)
	tanX = math.tan(fovX / 2)

	P = torch.zeros(4, 4)

	P[0, 0] = 2 * znear / (2 * tanX * znear)
	P[1, 1] = 2 * znear / (2 * tanY * znear)
	P[0, 2] = 0
	P[1, 2] = 0
	P[3, 2] = 1
	P[2, 2] = zfar / (zfar - znear)
	P[2, 3] = -(zfar * znear) / (zfar - znear)

	return P


def fov2focal(fov, pixels):
	return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
	return 2 * math.atan(pixels / (2 * focal))

