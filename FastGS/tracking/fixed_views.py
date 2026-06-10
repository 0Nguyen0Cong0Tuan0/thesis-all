import os
import json
from typing import List, Dict, Any


def _camera_to_meta(cam) -> Dict[str, Any]:
    return {
        "uid": int(getattr(cam, "uid", -1)),
        "colmap_id": int(getattr(cam, "colmap_id", -1)),
        "image_name": str(getattr(cam, "image_name", "")),
        "image_width": int(getattr(cam, "image_width", 0)),
        "image_height": int(getattr(cam, "image_height", 0)),
    }


def select_fixed_views(scene, dataset, num_views: int, prefer_test: bool = True) -> List:
    """Deterministically pick fixed cameras for snapshot rendering."""
    views = []
    if prefer_test and getattr(dataset, "eval", False):
        try:
            test_views = scene.getTestCameras()
            if test_views and len(test_views) > 0:
                views = test_views
        except Exception:
            views = []

    if not views:
        views = scene.getTrainCameras()

    views = list(views)
    views.sort(key=lambda c: getattr(c, "image_name", ""))
    return views[: max(0, int(num_views))]


def write_fixed_views_json(out_dir: str, cams: List) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fixed_views.json")
    data = {
        "num_views": len(cams),
        "views": [_camera_to_meta(c) for c in cams],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
