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


# ============================================================
# Multi-view geometry helpers (for the model-free Reflection Score / RS locator).
# Adapted from PGSR (arXiv:2406.06521). Conventions match THIS codebase:
#   world_view_transform W is the ROW-vector world->camera transform
#     (p_view = [p_world, 1] @ W), i.e. Rw = W[:3,:3], t = W[3,:3], so
#     p_view = p_world @ Rw + t   and   p_world = (p_view - t) @ Rw^T.
#   Pinhole: u = Fx * x_c/z_c + Cx, v = Fy * y_c/z_c + Cy.
# All functions are pure tensor math (CPU-testable; no rasterizer).
# ============================================================

def cam_intrinsics(FoVx, FoVy, width, height):
	"""Return (Fx, Fy, Cx, Cy) for a centered pinhole from FoV + image size."""
	Fx = fov2focal(FoVx, width)
	Fy = fov2focal(FoVy, height)
	return Fx, Fy, width / 2.0, height / 2.0


def unproject_depth_to_world(depth, Fx, Fy, Cx, Cy, world_view_transform):
	"""depth: [H,W] (z in camera space). Returns world points [H*W, 3]."""
	H, W = depth.shape
	device = depth.device
	v, u = torch.meshgrid(torch.arange(H, device=device, dtype=depth.dtype),
	                       torch.arange(W, device=device, dtype=depth.dtype), indexing="ij")
	z = depth.reshape(-1)
	x_c = (u.reshape(-1) - Cx) / Fx * z
	y_c = (v.reshape(-1) - Cy) / Fy * z
	p_cam = torch.stack([x_c, y_c, z], dim=-1)              # [N,3]
	Rw = world_view_transform[:3, :3]
	t = world_view_transform[3, :3]
	p_world = (p_cam - t) @ Rw.transpose(0, 1)             # inverse of row-vec W2C
	return p_world


def project_world_to_camera(points_world, Fx, Fy, Cx, Cy, world_view_transform):
	"""points_world: [N,3]. Returns (uv [N,2], z [N]) in the target camera."""
	Rw = world_view_transform[:3, :3]
	t = world_view_transform[3, :3]
	p_cam = points_world @ Rw + t                          # [N,3], row-vec W2C
	z = p_cam[:, 2]
	u = Fx * p_cam[:, 0] / (z + 1e-8) + Cx
	v = Fy * p_cam[:, 1] / (z + 1e-8) + Cy
	return torch.stack([u, v], dim=-1), z


def patch_offsets(h, device):
	"""Integer pixel offsets for a (2h+1)x(2h+1) patch -> [(2h+1)^2, 2] as (dx,dy)."""
	off = torch.arange(-h, h + 1, device=device)
	yy, xx = torch.meshgrid(off, off, indexing="ij")
	return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def patch_warp(H, uv):
	"""Apply per-point homographies to pixel patches.
	H: [N,3,3], uv: [N,P,2] -> warped [N,P,2]."""
	N, P, _ = uv.shape
	ones = torch.ones((N, P, 1), device=uv.device, dtype=uv.dtype)
	homo = torch.cat([uv, ones], dim=-1)                   # [N,P,3]
	warped = torch.bmm(homo, H.transpose(1, 2))           # [N,P,3]
	return warped[..., :2] / (warped[..., 2:3] + 1e-8)

