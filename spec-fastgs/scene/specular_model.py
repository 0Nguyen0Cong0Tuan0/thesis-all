# ============================================================
# Specular Model Wrapper (Spec-Gaussian style)
# ============================================================

import torch
import torch.nn as nn
import os
import json

from utils.spec_utils import SpecularNetwork, SpecularNetworkReal
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_linear_noise_func


class SpecularModel:
    """
    Wrapper for Specular Network (SG style)

    This class manages:
    - ASG-based SpecularNetwork
    - optimizer
    - learning rate schedule
    """

    def __init__(self, is_real=False, is_indoor=False, arch_cfg=None):
        """
        Args:
            is_real (bool): use real-scene variant
            is_indoor (bool): affects network config
            arch_cfg (dict|None): opt-in alternative architecture (see
                utils/spec_arch.SpecularNetworkV2). If None, falls back to the
                SPEC_ARCH env var (JSON). If still None, the ORIGINAL network is
                used — i.e. the default training path is unchanged. See the ablation
                in results/MLP_LATENT_ABLATION_2026-06-13.md for which configs help.
        """
        if arch_cfg is None:
            env = os.environ.get("SPEC_ARCH")
            if env:
                arch_cfg = json.loads(env)

        if arch_cfg:
            from utils.spec_arch import SpecularNetworkV2, count_params
            self.specular = SpecularNetworkV2(**arch_cfg).cuda()
            print(f"[Specular] V2 arch enabled: {arch_cfg} | "
                  f"shared params = {count_params(self.specular)}")
        elif is_real:
            self.specular = SpecularNetworkReal(is_indoor).cuda()
        else:
            self.specular = SpecularNetwork().cuda()

        self.optimizer = None
        self.spatial_lr_scale = 5

    # ------------------------------------------------------------
    # Forward step
    # ------------------------------------------------------------
    def step(self, asg_feature, viewdir, normal):
        """
        Args:
            asg_feature: [N, F] latent ASG feature (GaussianModel)
            viewdir: [N, 3]
            normal: [N, 3]

        Returns:
            specular_rgb: [N, 3]
        """
        return self.specular(asg_feature, viewdir, normal)

    # ------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------
    def train_setting(self, training_args):
        """
        Setup optimizer and scheduler
        """

        param_groups = [
            {
                'params': list(self.specular.parameters()),
                'lr': training_args.feature_lr / 10.0,
                "name": "specular"
            }
        ]

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        # scheduler (same style as SG)
        specular_start_iter = getattr(training_args, "specular_start_iter", 3000)
        max_steps = training_args.iterations - specular_start_iter
        if max_steps <= 0:
            max_steps = training_args.specular_lr_max_steps

        self.specular_scheduler_args = get_linear_noise_func(
            lr_init=training_args.feature_lr,
            lr_final=training_args.feature_lr / 20,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=max_steps
        )

    # ------------------------------------------------------------
    # Learning rate update
    # ------------------------------------------------------------
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "specular":
                lr = self.specular_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    # ------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(
            model_path,
            f"specular/iteration_{iteration}"
        )
        os.makedirs(out_weights_path, exist_ok=True)

        torch.save(
            self.specular.state_dict(),
            os.path.join(out_weights_path, 'specular.pth')
        )

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------
    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(
                os.path.join(model_path, "specular")
            )
        else:
            loaded_iter = iteration

        weights_path = os.path.join(
            model_path,
            f"specular/iteration_{loaded_iter}/specular.pth"
        )

        print(f"[Specular] Loading weights from: {weights_path}")
        self.specular.load_state_dict(torch.load(weights_path))

    # ------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------
    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)



