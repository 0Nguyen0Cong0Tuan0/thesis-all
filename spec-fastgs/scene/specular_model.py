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


def _infer_arch_cfg(state_dict):
    """Reconstruct a SpecularNetworkV2 config from checkpoint keys.

    Used as a fallback for V2 checkpoints saved before spec_arch.json existed
    (e.g. the v2.5 r2 run). Returns None for the ORIGINAL SpecularNetwork layout
    (fc1/fc2/fc3) so the default network is kept.
    """
    keys = set(state_dict.keys())
    # Original network has render_module.fc3; V2 has render_module.fc_out.
    if "render_module.fc_out.weight" not in keys:
        return None

    # latent mode: lowrank => gaussian_feature is a Sequential (.0/.1)
    if "gaussian_feature.0.weight" in keys:
        latent_mode = "lowrank"
        rank = state_dict["gaussian_feature.0.weight"].shape[0]
        asg_feature = state_dict["gaussian_feature.0.weight"].shape[1]
    else:
        latent_mode = "dense"
        rank = 8
        asg_feature = state_dict["gaussian_feature.weight"].shape[1]

    featureC = state_dict["render_module.fc_out.weight"].shape[1]

    # depth = number of distinct render_module.hidden.<i> blocks
    import re
    depth = len({int(m.group(1)) for k in keys
                 for m in [re.match(r"render_module\.hidden\.(\d+)\.", k)] if m})

    # activation: relu uses Sequential(Linear, ReLU) -> ".0.weight";
    # siren/wire wrap a .linear -> ".linear.weight" (indistinguishable from keys,
    # so we can only safely auto-detect relu; siren/wire must use spec_arch.json).
    activation = "relu" if "render_module.hidden.0.0.weight" in keys else "siren"

    # viewpe from first hidden in_features:
    #   in_mlpC = 6*viewpe + 3 + num_theta*num_phi*2 + 1, with num_theta*num_phi=32
    w0 = ("render_module.hidden.0.0.weight" if activation == "relu"
          else "render_module.hidden.0.linear.weight")
    in_mlpC = state_dict[w0].shape[1]
    viewpe = max(0, round((in_mlpC - (3 + 32 * 2 + 1)) / 6))

    return dict(asg_feature=asg_feature, featureC=featureC, activation=activation,
                viewpe=viewpe, depth=depth, latent_mode=latent_mode, rank=rank)


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

        self.arch_cfg = arch_cfg          # remembered so save_weights can persist it
        self.is_real = is_real
        self.is_indoor = is_indoor

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

        # Persist the architecture config so render/metrics can rebuild the EXACT
        # network (the V2 arch has a different state_dict layout than the original).
        # Without this, load_weights would build the default net and fail on a
        # key mismatch (the v2.5 r2 bug).
        with open(os.path.join(out_weights_path, 'spec_arch.json'), 'w') as f:
            json.dump(self.arch_cfg if self.arch_cfg else {}, f)

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

        weights_dir = os.path.join(model_path, f"specular/iteration_{loaded_iter}")
        weights_path = os.path.join(weights_dir, "specular.pth")

        print(f"[Specular] Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path)

        # Rebuild the matching architecture before loading. Priority:
        #   1) spec_arch.json saved next to the weights (v2.5+ checkpoints),
        #   2) the SPEC_ARCH env / arch_cfg this model was constructed with,
        #   3) inference from the state_dict keys (handles older V2 checkpoints,
        #      e.g. the r2 run trained before this fix).
        arch_cfg = None
        cfg_path = os.path.join(weights_dir, "spec_arch.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                arch_cfg = json.load(f) or None
        if arch_cfg is None:
            arch_cfg = self.arch_cfg
        if arch_cfg is None:
            arch_cfg = _infer_arch_cfg(state_dict)

        if arch_cfg:
            from utils.spec_arch import SpecularNetworkV2
            print(f"[Specular] Rebuilding V2 arch for load: {arch_cfg}")
            self.specular = SpecularNetworkV2(**arch_cfg).cuda()

        self.specular.load_state_dict(state_dict)

    # ------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------
    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)



