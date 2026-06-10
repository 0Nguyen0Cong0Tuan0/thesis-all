# ============================================================
# Dataset Readers (Final: FastGS + minimal SG)
# ============================================================

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional

import numpy as np
import json
from pathlib import Path

from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB

from scene.gaussian_model import BasicPointCloud


# ------------------------------------------------------------
# DATA STRUCTURES
# ------------------------------------------------------------

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: Optional[np.array] = None  # ✅ SG-compatible


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def getNerfppNorm(cam_info):

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    cam_centers = np.hstack(cam_centers)

    center = np.mean(cam_centers, axis=1, keepdims=True)
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)

    radius = np.max(dist) * 1.1

    return {
        "translate": -center.flatten(),
        "radius": radius
    }


# ------------------------------------------------------------
# CAMERA LOADER (COLMAP)
# ------------------------------------------------------------

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):

        sys.stdout.write(f"\rReading camera {idx+1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        width = intr.width
        height = intr.height

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # ---- intrinsics ----
        if intr.model == "SIMPLE_PINHOLE":
            fx = intr.params[0]
            fy = fx

        elif intr.model in ["PINHOLE", "OPENCV", "SIMPLE_RADIAL"]:
            fx = intr.params[0]
            fy = intr.params[1]

        else:
            raise RuntimeError("Unsupported camera model")

        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)

        # ---- image ----
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image = Image.open(image_path)

        cam_infos.append(
            CameraInfo(
                uid=intr.id,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=Path(image_path).stem,
                width=width,
                height=height,
                depth=None,
            )
        )

    sys.stdout.write("\n")
    return cam_infos


# ------------------------------------------------------------
# POINT CLOUD
# ------------------------------------------------------------

def fetchPly(path):
    ply = PlyData.read(path)

    v = ply["vertex"]

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T / 255.0
    normal = np.vstack([v["nx"], v["ny"], v["nz"]]).T

    return BasicPointCloud(points=xyz, colors=rgb, normals=normal)


def storePly(path, xyz, rgb):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ]

    normals = np.zeros_like(xyz)

    data = np.empty(xyz.shape[0], dtype=dtype)
    data[:] = list(map(tuple, np.concatenate((xyz, normals, rgb), axis=1)))

    PlyData([PlyElement.describe(data, "vertex")]).write(path)


# ------------------------------------------------------------
# MAIN SCENE LOADER
# ------------------------------------------------------------

def readColmapSceneInfo(path, images, eval, llffhold=8):

    try:
        extr = read_extrinsics_binary(os.path.join(path, "sparse/0/images.bin"))
        intr = read_intrinsics_binary(os.path.join(path, "sparse/0/cameras.bin"))
    except:
        extr = read_extrinsics_text(os.path.join(path, "sparse/0/images.txt"))
        intr = read_intrinsics_text(os.path.join(path, "sparse/0/cameras.txt"))

    reading_dir = "images" if images is None else images

    cam_infos = readColmapCameras(
        extr,
        intr,
        os.path.join(path, reading_dir)
    )

    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    # split train/test
    if eval:
        train = [c for i, c in enumerate(cam_infos) if i % llffhold != 0]
        test = [c for i, c in enumerate(cam_infos) if i % llffhold == 0]
    else:
        train = cam_infos
        test = []

    norm = getNerfppNorm(train)

    # ---- point cloud ----
    ply_path = os.path.join(path, "sparse/0/points3D.ply")

    if not os.path.exists(ply_path):
        try:
            xyz, rgb, _ = read_points3D_binary(os.path.join(path, "sparse/0/points3D.bin"))
        except:
            xyz, rgb, _ = read_points3D_text(os.path.join(path, "sparse/0/points3D.txt"))

        storePly(ply_path, xyz, rgb)

    pcd = fetchPly(ply_path)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train,
        test_cameras=test,
        nerf_normalization=norm,
        ply_path=ply_path
    )


# ------------------------------------------------------------
# INTERFACE
# ------------------------------------------------------------

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
}

