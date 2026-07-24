# ============================================================
# Camera Utils (Final: FastGS + minimal SG)
# ============================================================

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[INFO] Large image >1.6K, downscaling")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized = PILtoTorch(cam_info.image, resolution)
    gt_image = resized[:3, ...]

    mask = None
    if resized.shape[0] == 4:
        mask = resized[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        depth=getattr(cam_info, "depth", None),  # ✅ SG compat
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    return [loadCam(args, i, c, resolution_scale) for i, c in enumerate(cam_infos)]


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)

    return {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": W2C[:3, 3].tolist(),
        "rotation": [x.tolist() for x in W2C[:3, :3]],
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }

