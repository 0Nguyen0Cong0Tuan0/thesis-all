import os
from typing import List, Dict, Callable, Optional

import torch

try:
    import torchvision
except Exception:
    torchvision = None


def _safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in "-_":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep) if keep else "view"


def save_fixed_view_renders(
    *,
    out_dir: str,
    iteration: int,
    cams: List,
    render_snapshot: Callable[[object, int], torch.Tensor],
    save_gt_once: bool = False,
    manifest: Optional[Dict] = None,
):
    """Render + save PNGs for selected cameras.

    render_snapshot(cam, iteration) should return a CHW tensor in [0, 1].
    """
    if torchvision is None:
        raise RuntimeError("torchvision is required for saving images")

    renders_root = os.path.join(out_dir, "renders")
    gts_root = os.path.join(out_dir, "gts")

    it = int(iteration)

    for cam in cams:
        view_name = _safe_name(getattr(cam, "image_name", f"uid_{getattr(cam, 'uid', -1)}"))
        view_render_dir = os.path.join(renders_root, view_name)
        os.makedirs(view_render_dir, exist_ok=True)

        with torch.no_grad():
            rendering = render_snapshot(cam, it)
            rendering = torch.clamp(rendering, 0.0, 1.0)

        png_path = os.path.join(view_render_dir, f"{it:06d}.png")
        torchvision.utils.save_image(rendering, png_path)

        if manifest is not None:
            manifest.setdefault(str(it), {})[view_name] = os.path.relpath(png_path, out_dir)

        if save_gt_once:
            gt_path = os.path.join(gts_root, view_name + ".png")
            if not os.path.exists(gt_path):
                gt = getattr(cam, "original_image", None)
                if gt is not None:
                    os.makedirs(gts_root, exist_ok=True)
                    gt = gt[0:3, :, :]
                    torchvision.utils.save_image(gt, gt_path)
