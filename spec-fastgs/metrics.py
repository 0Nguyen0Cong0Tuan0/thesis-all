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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        with Image.open(renders_dir / fname) as render:
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        with Image.open(gt_dir / fname) as gt:
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readSpecImages(spec_dir, image_name, image_index):
    stem = Path(image_name).stem
    index_stem = f"{image_index:05d}"
    spec_paths = {
        "only_asg": [spec_dir / f"{stem}_only_asg.png", spec_dir / f"{index_stem}_only_asg.png"],
        "only_sh": [spec_dir / f"{stem}_only_sh.png", spec_dir / f"{index_stem}_only_sh.png"],
        "residual_real": [spec_dir / f"{stem}_residual_real.png", spec_dir / f"{index_stem}_residual_real.png"],
    }
    spec_images = {}
    for key, candidates in spec_paths.items():
        for path in candidates:
            if path.exists():
                with Image.open(path) as image:
                    spec_images[key] = tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].cuda()
                break
    return spec_images

def safe_mean(values):
    return torch.tensor(values).mean().item() if len(values) > 0 else None

def masked_l1(pred, gt, mask):
    if mask.sum() == 0:
        return None
    return torch.abs(pred - gt)[mask.expand_as(pred)].mean()

def masked_psnr(pred, gt, mask):
    if mask.sum() == 0:
        return None
    mse = ((pred - gt)[mask.expand_as(pred)] ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-12))

def computeAuxMetrics(render, gt, spec_images):
    only_asg = spec_images.get("only_asg")
    residual_real = spec_images.get("residual_real")
    if only_asg is None or residual_real is None:
        return {}

    # Residual-derived mask approximates specular regions in rendered test views.
    residual_gray = residual_real.mean(dim=1, keepdim=True)
    asg_gray = only_asg.mean(dim=1, keepdim=True)
    spec_mask = residual_gray > 0.02
    non_spec_mask = ~spec_mask
    asg_mask = asg_gray > 0.02

    aux = {
        "Spec_L1": None,
        "Spec_PSNR": None,
        "NonSpec_L1": None,
        "NonSpec_PSNR": None,
        "ASG_Mean": only_asg.mean(),
        "ASG_Max": only_asg.max(),
        "Residual_Mean": residual_real.mean(),
        "ASG_Energy_In_Residual": None,
        "ASG_Residual_IoU": None,
    }

    aux["Spec_L1"] = masked_l1(render, gt, spec_mask)
    aux["Spec_PSNR"] = masked_psnr(render, gt, spec_mask)
    aux["NonSpec_L1"] = masked_l1(render, gt, non_spec_mask)
    aux["NonSpec_PSNR"] = masked_psnr(render, gt, non_spec_mask)

    asg_energy = only_asg.sum()
    if asg_energy > 0:
        aux["ASG_Energy_In_Residual"] = only_asg[spec_mask.expand_as(only_asg)].sum() / asg_energy

    union = torch.logical_or(asg_mask, spec_mask).sum()
    if union > 0:
        aux["ASG_Residual_IoU"] = torch.logical_and(asg_mask, spec_mask).sum().float() / union.float()

    return aux

def to_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().mean().item()
    return float(value)

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    grouped_dict = {}
    grouped_per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            grouped_dict[scene_dir] = {}
            grouped_per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in sorted(os.listdir(test_dir)):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                grouped_dict[scene_dir][method] = {"main_metrics": {}, "aux_metrics": {}}
                grouped_per_view_dict[scene_dir][method] = {"main_metrics": {}, "aux_metrics": {}}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                spec_dir = method_dir / "spec"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                aux_per_metric = {}

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    spec_images = readSpecImages(spec_dir, image_names[idx], idx)
                    aux_metrics = computeAuxMetrics(renders[idx], gts[idx], spec_images)
                    for key, value in aux_metrics.items():
                        aux_per_metric.setdefault(key, []).append(to_float(value))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                for key, values in aux_per_metric.items():
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        print("  {:<22s}: {:>12.7f}".format(key, torch.tensor(valid_values).mean()))
                print("")

                main_metrics = {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item()
                }
                main_per_view = {
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}
                }
                aux_metrics_mean = {}
                aux_per_view = {}
                for key, values in aux_per_metric.items():
                    valid_values = [v for v in values if v is not None]
                    aux_metrics_mean[key] = safe_mean(valid_values)
                    aux_per_view[key] = {name: value for name, value in zip(image_names, values)}

                full_dict[scene_dir][method].update(main_metrics)
                per_view_dict[scene_dir][method].update(main_per_view)
                grouped_dict[scene_dir][method]["main_metrics"].update(main_metrics)
                grouped_dict[scene_dir][method]["aux_metrics"].update(aux_metrics_mean)
                grouped_per_view_dict[scene_dir][method]["main_metrics"].update(main_per_view)
                grouped_per_view_dict[scene_dir][method]["aux_metrics"].update(aux_per_view)

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/results_grouped.json", 'w') as fp:
                json.dump(grouped_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view_grouped.json", 'w') as fp:
                json.dump(grouped_per_view_dict[scene_dir], fp, indent=True)
        except Exception as error:
            print("Unable to compute metrics for model", scene_dir)
            print("Reason:", error)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
