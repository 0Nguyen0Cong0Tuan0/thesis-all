# ============================================================
# Camera (Final: FastGS + Spec-Gaussian support)
# ============================================================

import torch
from torch import nn
import numpy as np

from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):

    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        depth=None   # ✅ ADD (SG)
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        # --------------------------------------------------------
        # DEVICE
        # --------------------------------------------------------

        try:
            self.data_device = torch.device(data_device)
        except Exception:
            print(f"[Warning] Falling back to cuda")
            self.data_device = torch.device("cuda")

        # --------------------------------------------------------
        # IMAGE
        # --------------------------------------------------------

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # ✅ DEPTH SUPPORT
        self.depth = (
            torch.tensor(depth).to(self.data_device)
            if depth is not None
            else None
        )

        # alpha mask
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width),
                device=self.data_device
            )

        # --------------------------------------------------------
        # PROJECTION
        # --------------------------------------------------------

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1).to(self.data_device)

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx,
                fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(self.data_device)
        )

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )

        # ✅ CRITICAL
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    # ------------------------------------------------------------
    # CAMERA MOTION (SG / VIDEO SUPPORT)
    # ------------------------------------------------------------

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, self.trans, self.scale)
        ).transpose(0, 1).to(self.data_device)

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )

        self.camera_center = self.world_view_transform.inverse()[3, :3]


# ------------------------------------------------------------
# LIGHTWEIGHT CAMERA (RENDER ONLY)
# ------------------------------------------------------------

class MiniCam:

    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform
    ):
        self.image_width = width
        self.image_height = height

        self.FoVy = fovy
        self.FoVx = fovx

        self.znear = znear
        self.zfar = zfar

        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform

        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

