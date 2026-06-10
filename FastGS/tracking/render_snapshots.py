import os
from typing import List, Dict

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
    render_func,
    gaussians,
    pipe,
    background,
    mult: float,
    save_gt_once: bool = False,
    manifest: Dict = None,
):
    """Render + save PNGs for selected cameras."""
    if torchvision is None:
        raise RuntimeError("torchvision is required for saving images (used by FastGS\\render.py as well)")

    renders_root = os.path.join(out_dir, "renders")
    gts_root = os.path.join(out_dir, "gts")

    it = int(iteration)

    for cam in cams:
        view_name = _safe_name(getattr(cam, "image_name", f"uid_{getattr(cam, 'uid', -1)}"))
        view_render_dir = os.path.join(renders_root, view_name)
        os.makedirs(view_render_dir, exist_ok=True)

        with torch.no_grad():
            rendering = render_func(cam, gaussians, pipe, background, mult)["render"]
            rendering = torch.clamp(rendering, 0.0, 1.0)

        png_path = os.path.join(view_render_dir, f"{it:06d}.png")
        torchvision.utils.save_image(rendering, png_path)

        if manifest is not None:
            # Prefer a path relative to out_dir, but don't crash if cwd becomes invalid
            # (e.g., network-mounted working dirs disappearing).
            try:
                rel = os.path.relpath(png_path, out_dir)
            except Exception:
                out_norm = os.path.normpath(out_dir)
                png_norm = os.path.normpath(png_path)
                prefix = out_norm + os.sep
                rel = png_norm[len(prefix):] if png_norm.startswith(prefix) else png_norm
            manifest.setdefault(str(it), {})[view_name] = rel

        if save_gt_once:
            gt_path = os.path.join(gts_root, view_name + ".png")
            if not os.path.exists(gt_path):
                os.makedirs(gts_root, exist_ok=True)
                gt = cam.original_image[0:3, :, :]
                torchvision.utils.save_image(gt, gt_path)
