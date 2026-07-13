# ============================================================
# Scene (Spec-FastGS Final)
# ============================================================

import os
import random
import json
import torch

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks

from scene.gaussian_model import GaussianModel
from scene.specular_model import SpecularModel   # ✅ ADD (SG)

from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0]
    ):
        """
        Scene loader (FastGS + Specular support)
        """

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # ✅ SPECULAR MODEL HANDLE (optional use later)
        self.specular = None

        # --------------------------------------------------------
        # LOAD ITERATION
        # --------------------------------------------------------

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration

            print(f"Loading trained model at iteration {self.loaded_iter}")

        # --------------------------------------------------------
        # LOAD DATASET
        # --------------------------------------------------------

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )

        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Detected Blender dataset")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )

        else:
            raise RuntimeError("Unknown dataset format!")

        # --------------------------------------------------------
        # INIT OUTPUT STRUCTURE
        # --------------------------------------------------------

        if not self.loaded_iter:

            # copy ply
            with open(scene_info.ply_path, "rb") as src, \
                 open(os.path.join(self.model_path, "input.ply"), "wb") as dst:
                dst.write(src.read())

            # save camera json
            cams = []
            camlist = []

            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)

            for i, cam in enumerate(camlist):
                cams.append(camera_to_JSON(i, cam))

            with open(os.path.join(self.model_path, "cameras.json"), "w") as f:
                json.dump(cams, f)

        # --------------------------------------------------------
        # SHUFFLE
        # --------------------------------------------------------

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # --------------------------------------------------------
        # LOAD CAMERAS
        # --------------------------------------------------------

        for scale in resolution_scales:

            print("Loading Training Cameras")
            self.train_cameras[scale] = cameraList_from_camInfos(
                scene_info.train_cameras,
                scale,
                args
            )

            print("Loading Test Cameras")
            self.test_cameras[scale] = cameraList_from_camInfos(
                scene_info.test_cameras,
                scale,
                args
            )

        # --------------------------------------------------------
        # LOAD / INIT GAUSSIANS
        # --------------------------------------------------------

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"iteration_{self.loaded_iter}",
                    "point_cloud.ply"
                )
            )
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud,
                self.cameras_extent
            )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    def save(self, iteration):
        path = os.path.join(
            self.model_path,
            f"point_cloud/iteration_{iteration}"
        )

        self.gaussians.save_ply(
            os.path.join(path, "point_cloud.ply")
        )

        torch.save(
            self.gaussians.get_asg_features.detach().cpu(),
            os.path.join(path, "asg.pt")
        )
    # ------------------------------------------------------------
    # GET CAMERAS
    # ------------------------------------------------------------

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
