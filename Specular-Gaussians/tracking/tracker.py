import os
import json
import time
from typing import Any, Dict, Optional, Callable

import torch

from .diff import to_plain, diff_dict
from .fixed_views import select_fixed_views, write_fixed_views_json
from .render_snapshots import save_fixed_view_renders


class _JSONLWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def write(self, obj: Dict[str, Any]):
        line = json.dumps(to_plain(obj), ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _maybe_num_points(gaussians) -> Optional[int]:
    try:
        if hasattr(gaussians, "get_xyz"):
            return int(gaussians.get_xyz.shape[0])
    except Exception:
        pass
    try:
        if hasattr(gaussians, "get_anchor"):
            return int(gaussians.get_anchor.shape[0])
    except Exception:
        pass
    return None


class TrainingTracker:
    """Append-only tracker for Specular-Gaussians training.

    Outputs under <model_path>\\tracking\\:
      - hparams.json (static)
      - runtime_changes.jsonl (per-iteration state diffs)
      - events.jsonl (structured events)
      - fixed_views.json + renders_index.json + renders/...
    """

    def __init__(
        self,
        *,
        args,
        dataset,
        opt,
        pipe,
        scene,
        gaussians,
        background,
        render_snapshot: Callable[[object, int], torch.Tensor],
    ):
        self.args = args
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.scene = scene
        self.gaussians = gaussians
        self.background = background
        self.render_snapshot = render_snapshot

        model_path = getattr(args, "model_path", None) or getattr(dataset, "model_path", None)
        if model_path is None:
            raise ValueError("Cannot determine model_path for tracking outputs")

        self.out_dir = os.path.join(model_path, "tracking")
        os.makedirs(self.out_dir, exist_ok=True)

        self.runtime_writer = _JSONLWriter(os.path.join(self.out_dir, "runtime_changes.jsonl"))
        self.events_writer = _JSONLWriter(os.path.join(self.out_dir, "events.jsonl"))

        self.render_manifest_path = os.path.join(self.out_dir, "renders_index.json")
        self.render_manifest: Dict[str, Dict[str, str]] = {}

        self.fixed_views = []

        self._last_state: Dict[str, Any] = {}
        self._last_flush_time = time.time()

    def write_hparams(self):
        data = {
            "cli_args": to_plain(vars(self.args)),
            "dataset": to_plain(vars(self.dataset)),
            "opt": to_plain(vars(self.opt)),
            "pipe": to_plain(vars(self.pipe)),
            "env": {
                "python_time": time.time(),
                "torch": getattr(torch, "__version__", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": str(torch.cuda.current_device()) if torch.cuda.is_available() else None,
            },
        }
        with open(os.path.join(self.out_dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def init_fixed_views(self):
        self.fixed_views = select_fixed_views(
            self.scene,
            self.dataset,
            num_views=getattr(self.args, "track_views", 10),
            prefer_test=not bool(getattr(self.args, "track_prefer_train", False)),
        )
        write_fixed_views_json(self.out_dir, self.fixed_views)

    def log_event(self, event_type: str, iteration: int, payload: Optional[Dict[str, Any]] = None):
        obj = {
            "type": str(event_type),
            "iteration": int(iteration),
            "ts": time.time(),
        }
        if payload:
            obj["payload"] = payload
        self.events_writer.write(obj)

    def _collect_state(
        self,
        iteration: int,
        *,
        extra_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "iteration": int(iteration),
            "pipe_debug": bool(getattr(self.pipe, "debug", False)),
            "active_sh_degree": int(getattr(self.gaussians, "active_sh_degree", -1))
            if hasattr(self.gaussians, "active_sh_degree")
            else None,
            "num_points": _maybe_num_points(self.gaussians),
        }

        with torch.no_grad():
            try:
                opacity = getattr(self.gaussians, "get_opacity", None)
                if opacity is not None and hasattr(opacity, "numel") and opacity.numel():
                    state.update(
                        {
                            "opacity_min": float(opacity.min().item()),
                            "opacity_mean": float(opacity.mean().item()),
                            "opacity_max": float(opacity.max().item()),
                        }
                    )
            except Exception:
                pass

            try:
                scaling = getattr(self.gaussians, "get_scaling", None)
                if scaling is not None and hasattr(scaling, "numel") and scaling.numel():
                    state["scaling_max"] = float(scaling.max().item())
            except Exception:
                pass

        if extra_state:
            for k, v in extra_state.items():
                state[str(k)] = v
        if metrics:
            for k, v in metrics.items():
                state[str(k)] = v
        return state

    def log_iteration(
        self,
        iteration: int,
        *,
        extra_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        curr = self._collect_state(iteration, extra_state=extra_state, metrics=metrics)
        changes = diff_dict(self._last_state, curr) if self._last_state else curr

        self.runtime_writer.write({
            "iteration": int(iteration),
            "ts": time.time(),
            "changes": changes,
        })
        self._last_state = curr

        now = time.time()
        if now - self._last_flush_time > 10:
            self.flush_manifests()
            self._last_flush_time = now

    def should_render(self, iteration: int) -> bool:
        if not getattr(self.args, "track_enable", False):
            return False

        start = int(getattr(self.args, "track_render_start", 1))
        end = int(getattr(self.args, "track_render_end", -1))
        if end < 0:
            end = int(getattr(self.opt, "iterations", iteration))

        if iteration < start or iteration > end:
            return False

        interval = int(getattr(self.args, "track_render_interval", 10))
        if interval <= 0:
            return False

        if iteration == start or iteration == end:
            return True

        if start == 1:
            return (iteration % interval) == 0

        return ((iteration - start) % interval) == 0

    def render_fixed_views(self, iteration: int):
        save_fixed_view_renders(
            out_dir=self.out_dir,
            iteration=iteration,
            cams=self.fixed_views,
            render_snapshot=self.render_snapshot,
            save_gt_once=bool(getattr(self.args, "track_save_gt_once", False)),
            manifest=self.render_manifest,
        )

    def flush_manifests(self):
        with open(self.render_manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.render_manifest, f, ensure_ascii=False, indent=2)
