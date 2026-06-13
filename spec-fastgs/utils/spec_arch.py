# ============================================================
# Specular MLP / latent-space architecture variants (drop-in)
# ============================================================
# All variants preserve the SpecularNetwork interface:
#     forward(x, view, normal) -> [N, 3]
# and the ASGRender interface:
#     forward(pts, viewdirs, features, normal) -> [N, 3]
# so they are direct swaps for utils/spec_utils.py in train.py / render.py.
#
# Motivation (see results/SOLUTIONS_RESEARCH_2026-06-13.md): the specular branch
# under-produces energy (dim a*>1, blurry sigma~4.9). Two levers explored here:
#   (1) MLP structure / complexity  -> defeat ReLU spectral bias on high-freq
#       view-dependent signal: SIREN, WIRE (Gabor), wider Fourier view-encoding,
#       deeper+wider with residuals.
#   (2) Latent space                -> change how the per-Gaussian 24-d ASG latent
#       is parameterised: dense (baseline), low-rank (LoRA-style), FiLM modulation.
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.spec_utils import RenderingEquationEncoding, positional_encoding


# ------------------------------------------------------------
# Activation layers
# ------------------------------------------------------------

class SineLayer(nn.Module):
    """SIREN sine layer with the canonical Sitzmann et al. (2020) init."""
    def __init__(self, in_f, out_f, is_first=False, omega_0=30.0, bias=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_f, 1.0 / in_f)
            else:
                b = math.sqrt(6.0 / in_f) / omega_0
                self.linear.weight.uniform_(-b, b)
            if bias:
                self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class GaborLayer(nn.Module):
    """Real WIRE (Saragadam et al., 2023): cos(w0*Wx) * exp(-(s0*Wx)^2)."""
    def __init__(self, in_f, out_f, is_first=False, omega_0=10.0, sigma_0=10.0, bias=True):
        super().__init__()
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        with torch.no_grad():
            # modest init keeps the Gaussian envelope from saturating early
            nn.init.xavier_normal_(self.linear.weight, gain=0.5)
            if bias:
                self.linear.bias.zero_()

    def forward(self, x):
        z = self.linear(x)
        return torch.cos(self.omega_0 * z) * torch.exp(-(self.sigma_0 * z) ** 2)


def _relu_block(in_f, out_f):
    return nn.Sequential(nn.Linear(in_f, out_f), nn.ReLU(inplace=True))


# ------------------------------------------------------------
# Configurable ASG decoder (kappa -> RGB)
# ------------------------------------------------------------

class ASGRenderV2(nn.Module):
    """
    Drop-in replacement for utils/spec_utils.ASGRender with selectable
    activation / depth / width / view-encoding bandwidth.

    activation in {'relu','siren','wire'}; depth = #hidden layers (>=1);
    residual adds a skip every layer when shapes match (default True).
    """
    def __init__(self, viewpe=2, featureC=128, num_theta=4, num_phi=8,
                 activation='relu', depth=2, residual=True,
                 omega_0=30.0, sigma_0=10.0, device='cuda'):
        super().__init__()
        self.num_theta = num_theta
        self.num_phi = num_phi
        self.viewpe = viewpe
        self.activation = activation
        self.residual = residual
        self.ree_function = RenderingEquationEncoding(num_theta, num_phi, device=device)

        self.in_mlpC = 2 * viewpe * 3 + 3 + num_theta * num_phi * 2 + 1

        layers = []
        dims = [self.in_mlpC] + [featureC] * depth
        for i in range(depth):
            in_f, out_f = dims[i], dims[i + 1]
            if activation == 'siren':
                layers.append(SineLayer(in_f, out_f, is_first=(i == 0), omega_0=omega_0))
            elif activation == 'wire':
                layers.append(GaborLayer(in_f, out_f, is_first=(i == 0),
                                         omega_0=omega_0, sigma_0=sigma_0))
            elif activation == 'relu':
                layers.append(_relu_block(in_f, out_f))
            else:
                raise ValueError(f"unknown activation {activation}")
        self.hidden = nn.ModuleList(layers)
        self.fc_out = nn.Linear(featureC, 3)
        nn.init.constant_(self.fc_out.bias, 0)

    def reflect(self, viewdir, normal):
        return 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir

    def safe_normalize(self, x):
        return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

    def forward(self, pts, viewdirs, features, normal):
        asg_params = features.view(-1, self.num_theta, self.num_phi, 4)
        a, la, mu = torch.split(asg_params, [2, 1, 1], dim=-1)

        reflect_dir = self.safe_normalize(self.reflect(-viewdirs, normal))
        color_feature = self.ree_function(reflect_dir, a, la, mu)
        color_feature = color_feature.view(color_feature.size(0), -1)

        normal_dot_viewdir = ((-viewdirs) * normal).sum(dim=-1, keepdim=True)
        inputs = [color_feature, normal_dot_viewdir]
        if self.viewpe >= 0:
            inputs.append(viewdirs)
        if self.viewpe > 0:
            inputs.append(positional_encoding(viewdirs, self.viewpe))
        h = torch.cat(inputs, dim=-1)

        for layer in self.hidden:
            out = layer(h)
            if self.residual and out.shape == h.shape:
                h = out + h
            else:
                h = out
        return self.fc_out(h)


# ------------------------------------------------------------
# Latent-space adapters (per-Gaussian ASG latent parameterisation)
# ------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise linear modulation (Perez et al., 2018): gamma/beta from latent."""
    def __init__(self, latent_dim, feat_dim):
        super().__init__()
        self.to_gamma_beta = nn.Linear(latent_dim, 2 * feat_dim)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.feat_dim = feat_dim

    def forward(self, h, latent):
        gb = self.to_gamma_beta(latent)
        gamma, beta = gb[:, :self.feat_dim], gb[:, self.feat_dim:]
        return (1.0 + gamma) * h + beta


# ------------------------------------------------------------
# Configurable specular network
# ------------------------------------------------------------

class SpecularNetworkV2(nn.Module):
    """
    Configurable specular network. Defaults reproduce the current SpecularNetwork
    (dense 24-d latent -> Linear(24->128) -> ASGRender relu, viewpe=2, depth=2).

    latent_mode:
      'dense'  : per-Gaussian latent fed straight through gaussian_feature (baseline)
      'lowrank': gaussian_feature factored as Linear(F->r)->Linear(r->128) (LoRA-style,
                 shrinks shared params and regularises the latent->ASG map)
    film: if True, also modulate the first hidden ASG-MLP feature by the latent.
    """
    def __init__(self, asg_feature=24, num_theta=4, num_phi=8, featureC=128,
                 activation='relu', viewpe=2, depth=2, residual=True,
                 latent_mode='dense', rank=8, film=False,
                 omega_0=30.0, sigma_0=10.0, device='cuda'):
        super().__init__()
        self.asg_feature = asg_feature
        self.num_theta = num_theta
        self.num_phi = num_phi
        self.asg_hidden = num_theta * num_phi * 4
        self.latent_mode = latent_mode

        if latent_mode == 'dense':
            self.gaussian_feature = nn.Linear(asg_feature, self.asg_hidden)
        elif latent_mode == 'lowrank':
            self.gaussian_feature = nn.Sequential(
                nn.Linear(asg_feature, rank, bias=False),
                nn.Linear(rank, self.asg_hidden),
            )
        else:
            raise ValueError(f"unknown latent_mode {latent_mode}")

        self.render_module = ASGRenderV2(
            viewpe=viewpe, featureC=featureC, num_theta=num_theta, num_phi=num_phi,
            activation=activation, depth=depth, residual=residual,
            omega_0=omega_0, sigma_0=sigma_0, device=device,
        )
        self.film = FiLM(asg_feature, featureC) if film else None

    def forward(self, x, view, normal):
        feature = self.gaussian_feature(x)
        # (FiLM is applied inside a thin wrapper only if enabled; kept simple here
        #  because ASGRenderV2's first hidden layer already mixes the ASG feature.)
        return self.render_module(x if False else x, view, feature, normal) \
            if self.film is None else self._forward_film(x, view, feature, normal)

    def _forward_film(self, latent, view, feature, normal):
        # Replicates ASGRenderV2.forward but FiLM-modulates the first hidden output.
        rm = self.render_module
        asg_params = feature.view(-1, rm.num_theta, rm.num_phi, 4)
        a, la, mu = torch.split(asg_params, [2, 1, 1], dim=-1)
        reflect_dir = rm.safe_normalize(rm.reflect(-view, normal))
        cf = rm.ree_function(reflect_dir, a, la, mu).view(latent.size(0), -1)
        ndv = ((-view) * normal).sum(dim=-1, keepdim=True)
        inputs = [cf, ndv]
        if rm.viewpe >= 0:
            inputs.append(view)
        if rm.viewpe > 0:
            inputs.append(positional_encoding(view, rm.viewpe))
        h = torch.cat(inputs, dim=-1)
        for i, layer in enumerate(rm.hidden):
            out = layer(h)
            if i == 0:
                out = self.film(out, latent)
            h = out + h if (rm.residual and out.shape == h.shape) else out
        return rm.fc_out(h)


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
