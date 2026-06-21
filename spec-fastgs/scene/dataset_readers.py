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
# SYNTHETIC LOADERS (for Spec-Gaussian's eval datasets)
# ------------------------------------------------------------
# The repo shipped Colmap-only, but scene/__init__.py already branches to a
# "Blender" callback for transforms_train.json scenes (it just wasn't registered).
# Spec-Gaussian evaluates on: Anisotropic Synthetic + NeRF Synthetic (Blender json
# format) and NSVF Synthetic (pose/ + rgb/ + intrinsics.txt). Added here so we can
# reproduce their Tables 1/3/4. R/T use the SAME convention as readColmapCameras
# (R = c2w rotation, T = w2c translation); images are alpha-blended to the chosen
# background and returned as RGB PIL (loadCam expects PIL).

def _blend_to_rgb(image, white_background):
    im = np.array(image.convert("RGBA")).astype(np.float32) / 255.0
    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
    rgb = im[:, :, :3] * im[:, :, 3:4] + bg * (1 - im[:, :, 3:4])
    return Image.fromarray((rgb * 255.0).astype(np.uint8), "RGB")


def _random_pcd(ply_path, n=100_000, scale=2.6, offset=1.3):
    if not os.path.exists(ply_path):
        xyz = np.random.random((n, 3)) * scale - offset
        shs = np.random.random((n, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    return fetchPly(ply_path)


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as f:
        contents = json.load(f)
    fovx = contents["camera_angle_x"]
    for idx, frame in enumerate(contents["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        c2w = np.array(frame["transform_matrix"])
        c2w[:3, 1:3] *= -1                      # OpenGL/Blender -> COLMAP
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        image = _blend_to_rgb(Image.open(cam_name), white_background)
        W, H = image.size
        fovy = focal2fov(fov2focal(fovx, W), H)
        cam_infos.append(CameraInfo(
            uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image,
            image_path=cam_name, image_name=Path(cam_name).stem,
            width=W, height=H, depth=None))
    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    train = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    test = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    if not eval:
        train = train + test
        test = []
    norm = getNerfppNorm(train)
    pcd = _random_pcd(os.path.join(path, "points3d.ply"))
    return SceneInfo(point_cloud=pcd, train_cameras=train, test_cameras=test,
                     nerf_normalization=norm, ply_path=os.path.join(path, "points3d.ply"))


def readNSVFSceneInfo(path, white_background, eval):
    """NSVF format: intrinsics.txt (focal on line 1), pose/<split>_*.txt (4x4 c2w),
    rgb/<split>_*.(png|jpg). Split prefix: 0=train, 1=val, 2=test."""
    with open(os.path.join(path, "intrinsics.txt")) as f:
        focal = float(f.readline().split()[0])
    pose_dir, rgb_dir = os.path.join(path, "pose"), os.path.join(path, "rgb")
    rgb_files = {os.path.splitext(n)[0]: os.path.join(rgb_dir, n)
                 for n in os.listdir(rgb_dir)}
    cam_infos = []
    for idx, pf in enumerate(sorted(os.listdir(pose_dir))):
        stem = os.path.splitext(pf)[0]
        if stem not in rgb_files:
            continue
        c2w = np.loadtxt(os.path.join(pose_dir, pf)).reshape(4, 4)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        image = _blend_to_rgb(Image.open(rgb_files[stem]), white_background)
        W, H = image.size
        cam_infos.append(CameraInfo(
            uid=idx, R=R, T=T, FovY=focal2fov(focal, H), FovX=focal2fov(focal, W),
            image=image, image_path=rgb_files[stem], image_name=stem,
            width=W, height=H, depth=None))
    # split by NSVF prefix
    def pref(c): return c.image_name.split("_")[0]
    if eval:
        train = [c for c in cam_infos if pref(c) in ("0", "1")]
        test = [c for c in cam_infos if pref(c) == "2"]
        if not test:                      # fall back if prefixes absent
            train = [c for i, c in enumerate(cam_infos) if i % 8 != 0]
            test = [c for i, c in enumerate(cam_infos) if i % 8 == 0]
    else:
        train, test = cam_infos, []
    norm = getNerfppNorm(train)
    pcd = _random_pcd(os.path.join(path, "points3d.ply"), scale=3.0, offset=1.5)
    return SceneInfo(point_cloud=pcd, train_cameras=train, test_cameras=test,
                     nerf_normalization=norm, ply_path=os.path.join(path, "points3d.ply"))


# ------------------------------------------------------------
# INTERFACE
# ------------------------------------------------------------

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "NSVF": readNSVFSceneInfo,
}

