# ============================================================
# Specular Utilities (Spec-Gaussian compliant)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.quaternion_utils import init_predefined_omega


# ------------------------------------------------------------
# Positional Encoding
# ------------------------------------------------------------

def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


# ------------------------------------------------------------
# Rendering Equation Encoding (ASG core)
# ------------------------------------------------------------

class RenderingEquationEncoding(nn.Module):
    def __init__(self, num_theta, num_phi, device="cuda"):
        super().__init__()

        self.num_theta = num_theta
        self.num_phi = num_phi

        omega, omega_la, omega_mu = init_predefined_omega(num_theta, num_phi)
        self.omega = omega.view(1, num_theta, num_phi, 3).to(device)
        self.omega_la = omega_la.view(1, num_theta, num_phi, 3).to(device)
        self.omega_mu = omega_mu.view(1, num_theta, num_phi, 3).to(device)

    def forward(self, omega_o, a, la, mu):

        Smooth = F.relu(
            (omega_o[:, None, None] * self.omega).sum(dim=-1, keepdim=True)
        )

        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)

        exp_input = -la * (
            self.omega_la * omega_o[:, None, None]
        ).sum(dim=-1, keepdim=True).pow(2) - mu * (
            self.omega_mu * omega_o[:, None, None]
        ).sum(dim=-1, keepdim=True).pow(2)

        return a * Smooth * torch.exp(exp_input)


# ------------------------------------------------------------
# ASG Rendering Module (κ → RGB)
# ------------------------------------------------------------

class ASGRender(nn.Module):
    def __init__(self, viewpe=2, featureC=128, num_theta=4, num_phi=8,
                 normal_refine=False, refine_in_dim=24):
        super().__init__()

        self.num_theta = num_theta
        self.num_phi = num_phi
        self.viewpe = viewpe

        self.ree_function = RenderingEquationEncoding(num_theta, num_phi)

        self.in_mlpC = (
            2 * viewpe * 3 +
            3 +
            num_theta * num_phi * 2 +
            1   # normal dot viewdir
        )

        self.fc1 = nn.Linear(self.in_mlpC, featureC)
        self.fc2 = nn.Linear(featureC, featureC)
        self.fc3 = nn.Linear(featureC, 3)

        nn.init.constant_(self.fc3.bias, 0)

        # v2.9 (root cause B, disk-free): self-supervised learnable normal refinement.
        # The highlight PLACEMENT is governed entirely by `normal` (it sets reflect_dir
        # below). The geometric min-axis normal is noisy → low NCC. Instead of an
        # external monocular prior (disk-heavy, only weak-positive), predict a per-
        # Gaussian normal correction from the existing ASG latent and let the (now
        # working) multi-view specular loss identify the true normal — the Ref-NeRF /
        # 3DGS-DR principle. The correction is BOUNDED (0.3*tanh) and the output layer
        # is zero-initialised, so it starts as a no-op (== current behavior) and can
        # only gently refine — low-risk by construction.
        self.normal_refine = normal_refine
        if normal_refine:
            self.refine_mlp = nn.Sequential(
                nn.Linear(refine_in_dim, 32), nn.ReLU(), nn.Linear(32, 3)
            )
            nn.init.zeros_(self.refine_mlp[-1].weight)
            nn.init.zeros_(self.refine_mlp[-1].bias)

    def reflect(self, viewdir, normal):
        return 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir

    def safe_normalize(self, x):
        return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

    def forward(self, pts, viewdirs, features, normal):

        # --- ASG parameters ---
        asg_params = features.view(-1, self.num_theta, self.num_phi, 4)
        a, la, mu = torch.split(asg_params, [2, 1, 1], dim=-1)

        # --- optional learnable normal refinement (v2.9) ---
        if self.normal_refine:
            delta = 0.3 * torch.tanh(self.refine_mlp(pts))   # bounded correction
            normal = self.safe_normalize(normal + delta)

        # --- reflect direction ---
        reflect_dir = self.safe_normalize(
            self.reflect(-viewdirs, normal)
        )

        # --- ASG latent κ ---
        color_feature = self.ree_function(reflect_dir, a, la, mu)
        color_feature = color_feature.view(color_feature.size(0), -1)

        normal_dot_viewdir = ((-viewdirs) * normal).sum(dim=-1, keepdim=True)

        inputs = [color_feature, normal_dot_viewdir]

        if self.viewpe >= 0:
            inputs.append(viewdirs)

        if self.viewpe > 0:
            inputs.append(positional_encoding(viewdirs, self.viewpe))

        mlp_input = torch.cat(inputs, dim=-1)

        h1 = F.relu(self.fc1(mlp_input))
        h2 = F.relu(self.fc2(h1)) + h1
        rgb = self.fc3(h2)

        return rgb


# ------------------------------------------------------------
# Specular Network (Gaussian → ASG → κ → RGB)
# ------------------------------------------------------------

class SpecularNetwork(nn.Module):
    def __init__(self, normal_refine=False):
        super().__init__()

        # SG design
        self.asg_feature = 24
        self.num_theta = 4
        self.num_phi = 8

        self.asg_hidden = self.num_theta * self.num_phi * 4

        # feature → ASG params
        self.gaussian_feature = nn.Linear(self.asg_feature, self.asg_hidden)

        # ASG decoder
        self.render_module = ASGRender(
            viewpe=2,
            featureC=128,
            num_theta=self.num_theta,
            num_phi=self.num_phi,
            normal_refine=normal_refine,
            refine_in_dim=self.asg_feature,
        )

    def forward(self, x, view, normal):
        feature = self.gaussian_feature(x)
        spec = self.render_module(x, view, feature, normal)
        return spec


# ------------------------------------------------------------
# Real-scene variant (optional)
# ------------------------------------------------------------

class SpecularNetworkReal(nn.Module):
    def __init__(self, is_indoor=False, normal_refine=False):
        super().__init__()

        # Capacity bump (fix #1): the previous real variant used featureC=32 and
        # only 2x4=8 ASG lobes, which cannot represent sharp high-frequency
        # specular (metal/glass) — measured as dim, blurry, decorrelated highlights
        # (low energyRatio / low NCC vs GT). Match the full SpecularNetwork:
        # featureC=128 and 4x8=32 ASG lobes.
        self.asg_feature = 24
        self.num_theta = 4
        self.num_phi = 8

        self.asg_hidden = self.num_theta * self.num_phi * 4

        self.gaussian_feature = nn.Linear(self.asg_feature, self.asg_hidden)

        self.render_module = ASGRender(
            viewpe=2,
            featureC=128,
            num_theta=self.num_theta,
            num_phi=self.num_phi,
            normal_refine=normal_refine,
            refine_in_dim=self.asg_feature,
        )

    def forward(self, x, view, normal):
        feature = self.gaussian_feature(x)
        spec = self.render_module(x, view, feature, normal)
        return spec

