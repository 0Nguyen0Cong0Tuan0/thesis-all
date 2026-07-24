#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fastgs
import torchvision
from utils.general_utils import safe_state
from utils.gaussian_heatmap import save_gaussian_view_heatmaps
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import json


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    residual_path = os.path.join(model_path, name, "ours_{}".format(iteration), "residual")

    total_time = 0.0

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(residual_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        rendering = render_fastgs(view, gaussians, pipeline, background, args.mult)["render"]
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        gt = view.original_image[0:3, :, :]

        # Tính sai số (residual)
        residual = torch.abs(gt - rendering)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(residual, os.path.join(residual_path, '{0:05d}'.format(idx) + ".png"))

    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")
    return fps


def save_fps_to_results(model_path, iteration, fps):
    results_path = os.path.join(model_path, "results.json")
    results = {}
    if os.path.isfile(results_path):
        try:
            with open(results_path, "r") as fp:
                results = json.load(fp)
        except (OSError, json.JSONDecodeError):
            results = {}

    method = f"ours_{iteration}"
    results.setdefault(method, {})["FPS"] = fps
    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=True)
    print(f"Saved test FPS to {results_path}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train and not args.only_heatmap:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if args.only_heatmap or not skip_test:
            if not args.only_heatmap:
                test_fps = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)
                save_fps_to_results(dataset.model_path, scene.loaded_iter, test_fps)
            heatmap_path = save_gaussian_view_heatmaps(
                scene.getTestCameras(),
                gaussians,
                scene.model_path,
                scene.loaded_iter,
                render_fastgs,
                (pipeline, background, args.mult),
            )
            print(f"Saved Gaussian distribution heatmaps to {heatmap_path}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--only_heatmap", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
