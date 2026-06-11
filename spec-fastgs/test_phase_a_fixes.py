# CPU smoke test for the 2026-06-11 fixes (no CUDA / no rasterizer needed):
#   1. dedicated ASG optimizer: steps with MLP cadence, survives prune/cat
#   2. set_sh_lr_scale: scales f_rest LR only
#   3. get_normal_axis(indices=...): subset result == full result indexed
# Run from spec-fastgs/:  python test_phase_a_fixes.py

import sys, types, torch
from torch import nn

# stub CUDA-only extensions before importing the model
_sk = types.ModuleType('simple_knn'); _skc = types.ModuleType('simple_knn._C')
_skc.distCUDA2 = lambda x: x
sys.modules.setdefault('simple_knn', _sk); sys.modules.setdefault('simple_knn._C', _skc)
try:
    import plyfile  # noqa: F401
except ImportError:
    _ply = types.ModuleType('plyfile'); _ply.PlyData = object; _ply.PlyElement = object
    sys.modules['plyfile'] = _ply

# stub the CUDA rasterizer + fused_ssim so utils.fast_utils imports on CPU
_dgr = types.ModuleType('diff_gaussian_rasterization_fastgs')
_dgr.GaussianRasterizationSettings = object
_dgr.GaussianRasterizer = object
sys.modules.setdefault('diff_gaussian_rasterization_fastgs', _dgr)
_fs = types.ModuleType('fused_ssim')
_fs.fused_ssim = lambda *a, **k: torch.tensor(1.0)
sys.modules.setdefault('fused_ssim', _fs)

from scene.gaussian_model import GaussianModel
from utils.fast_utils import compute_metric_map
import utils.general_utils as gu


def _build_rotation_cpu(r):
    # identical math to gu.build_rotation, minus the hardcoded device='cuda'
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] + r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=q.device)
    rr, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z); R[:, 0, 1] = 2 * (x*y - rr*z); R[:, 0, 2] = 2 * (x*z + rr*y)
    R[:, 1, 0] = 2 * (x*y + rr*z); R[:, 1, 1] = 1 - 2 * (x*x + z*z); R[:, 1, 2] = 2 * (y*z - rr*x)
    R[:, 2, 0] = 2 * (x*z - rr*y); R[:, 2, 1] = 2 * (y*z + rr*x); R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


gu.build_rotation = _build_rotation_cpu

N = 12
gm = GaussianModel(3)
gm._xyz           = nn.Parameter(torch.randn(N, 3))
gm._features_dc   = nn.Parameter(torch.randn(N, 1, 3))
gm._features_rest = nn.Parameter(torch.randn(N, 15, 3))
gm._features_asg  = nn.Parameter(torch.randn(N, 24))
gm._opacity       = nn.Parameter(torch.randn(N, 1))
gm._scaling       = nn.Parameter(torch.randn(N, 3))
gm._rotation      = nn.Parameter(torch.randn(N, 4))

# mirror training_setup's optimizer split (CPU, no cuda buffers)
gm.optimizer = torch.optim.Adam([
    {'params': [gm._xyz],          'lr': 1e-3, 'name': 'xyz'},
    {'params': [gm._features_dc],  'lr': 1e-3, 'name': 'f_dc'},
    {'params': [gm._opacity],      'lr': 1e-3, 'name': 'opacity'},
    {'params': [gm._scaling],      'lr': 1e-3, 'name': 'scaling'},
    {'params': [gm._rotation],     'lr': 1e-3, 'name': 'rotation'},
], lr=0.0, eps=1e-15)
gm.shoptimizer  = torch.optim.Adam([{'params': [gm._features_rest], 'lr': 2.5e-4, 'name': 'f_rest'}], lr=0.0, eps=1e-15)
gm.asgoptimizer = torch.optim.Adam([{'params': [gm._features_asg],  'lr': 2.5e-3, 'name': 'f_asg'}],  lr=0.0, eps=1e-15)
gm.sh_base_lr = 2.5e-4

# --- 3. subset normal equals full normal (do this before shapes change) ---
viewdir = torch.nn.functional.normalize(torch.randn(N, 3), dim=-1)
idx = torch.tensor([0, 3, 5, 11])
full = gm.get_normal_axis(viewdir)
sub  = gm.get_normal_axis(viewdir[idx], indices=idx)
assert torch.allclose(full[idx], sub, atol=1e-6), "subset normal != full normal"
print("normal-subset OK")

# --- 2. SH LR scaling touches only f_rest ---
gm.set_sh_lr_scale(0.55)
assert abs(gm.shoptimizer.param_groups[0]['lr'] - 2.5e-4 * 0.55) < 1e-15
assert gm.optimizer.param_groups[0]['lr'] == 1e-3
print("sh-lr-scale OK")

# --- 1a. asg_optimizer_step updates features and clears grads ---
before = gm._features_asg.detach().clone()
gm._features_asg.sum().backward()
gm.asg_optimizer_step()
assert not torch.allclose(before, gm._features_asg.detach()), "asg step did not update"
assert gm._features_asg.grad is None, "asg grads not cleared"
print("asg-step OK")

# --- 1b. prune keeps asg optimizer + Adam state consistent ---
keep = torch.zeros(N, dtype=torch.bool); keep[:7] = True
gm.tmp_radii = torch.zeros(N)
gm.xyz_gradient_accum = torch.zeros(N, 1); gm.xyz_gradient_accum_abs = torch.zeros(N, 1)
gm.denom = torch.zeros(N, 1); gm.max_radii2D = torch.zeros(N)
gm.prune_points(~keep)
assert gm._features_asg.shape == (7, 24)
st = gm.asgoptimizer.state[gm.asgoptimizer.param_groups[0]['params'][0]]
assert st['exp_avg'].shape == (7, 24), "asg Adam state not pruned"
print("prune OK")

# --- 1c. densification cat extends asg optimizer too ---
d = {'xyz': torch.randn(2, 3), 'f_dc': torch.randn(2, 1, 3), 'f_rest': torch.randn(2, 15, 3),
     'f_asg': torch.randn(2, 24), 'opacity': torch.randn(2, 1), 'scaling': torch.randn(2, 3),
     'rotation': torch.randn(2, 4)}
out = gm.cat_tensors_to_optimizer(d)
assert set(out) == {'xyz', 'f_dc', 'f_rest', 'f_asg', 'opacity', 'scaling', 'rotation'}
assert out['f_asg'].shape == (9, 24)
st = gm.asgoptimizer.state[gm.asgoptimizer.param_groups[0]['params'][0]]
assert st['exp_avg'].shape == (9, 24), "asg Adam state not extended"
# a post-cat step must not crash (state/param shape agreement)
out['f_asg'].sum().backward()
gm.asg_optimizer_step()
print("cat OK")

# --- 4. highlight-robust metric map (Lambertian-violation fix) ---
H = W = 10
gt = torch.zeros(3, H, W); gt[:, 0:2, 0:2] = 1.0      # 4 bright (specular) pixels
render = gt.clone()
render[:, 0, 0] = 0.5                                  # error inside a highlight
render[:, 5, 5] = 0.5                                  # equal error on diffuse region

mm = compute_metric_map(render, gt, loss_thresh=0.1, highlight_quantile=1.0)
assert mm[0, 0] == 1 and mm[5, 5] == 1, "mask disabled: both errors should vote"
mm = compute_metric_map(render, gt, loss_thresh=0.1, highlight_quantile=0.95)
assert mm[0, 0] == 0, "highlight error should be excluded from the vote"
assert mm[5, 5] == 1, "diffuse error must still vote"
assert mm.dtype == torch.int32
print("highlight-mask OK")

print("ALL PHASE-A FIX TESTS PASSED")
