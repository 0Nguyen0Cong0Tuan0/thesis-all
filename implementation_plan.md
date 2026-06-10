# Fix Spec-FastGS: Proper Specular-Gaussians Integration

After deep-diving all three codebases (FastGS, Specular-Gaussians baseline, and your spec-fastgs), I've identified the root causes and designed the precise fixes. The core problem is that your specular model is a **simplified custom MLP** that lacks the ASG (Anisotropic Spherical Gaussian) rendering equation from the baseline — this is the single most important piece that makes Specular-Gaussians work.

## Summary of Root Causes

| # | Problem | Impact |
|---|---------|--------|
| 1 | **Wrong specular architecture**: Your `SpecularModel` is a plain 3-layer MLP (feat+viewdir+normal → RGB). The baseline uses ASG: Rendering Equation Encoding with predefined omega directions, reflection-based color features, positional-encoded viewdirs, and normal·viewdir dot product | Specular MLP has no physics-informed inductive bias — it cannot learn view-dependent specular highlights effectively |
| 2 | **Wrong ASG feature dimension**: Baseline uses `asg_degree=24` (synthetic) or `12` (real) per-Gaussian ASG features. Your `specular_feat_dim=8` is too small and doesn't match the `SpecularNetwork` input expectation | Dim mismatch would crash if connected; 8 is too low to encode ASG lobes |
| 3 | **Missing `quaternion_utils.py`**: `spec_utils.py` imports `from .quaternion_utils import init_predefined_omega` but the file doesn't exist in `specular/` | The proper ASG network (which you already copied as `spec_utils.py`) cannot even be imported |
| 4 | **Normal computation misses view-dependent flipping**: Your `compute_gaussian_normals()` uses `@torch.no_grad()` and picks the smallest-scale axis — correct concept but missing the baseline's `flip_align_view()` which flips normals to face the camera. Also blocks gradient flow entirely | Backlit Gaussians get wrong normal sign → specular highlights on wrong side |
| 5 | **No specular LR scheduler**: Baseline uses `get_linear_noise_func` to decay specular LR over training. You use a constant `Adam(lr=1e-3)` with no decay | Specular MLP overfit/unstable in late training |
| 6 | **render.py loads wrong SpecularModel**: `render.py` line 103 does `from specular import SpecularModel` — this imports your simplified MLP. But if training used the ASG network, the checkpoint won't load | State dict key mismatch at render time |

## Proposed Changes

### Component 1: Add Missing `quaternion_utils.py`

#### [NEW] [quaternion_utils.py](file:///d:/Thesis/All/spec-fastgs/specular/quaternion_utils.py)
Copy from `Specular-Gaussians/utils/quaternion_utils.py` into `spec-fastgs/specular/quaternion_utils.py`. This provides `init_predefined_omega()` which `spec_utils.py` already imports.

---

### Component 2: Replace Specular Model with Proper ASG Architecture

#### [MODIFY] [specular_model.py](file:///d:/Thesis/All/spec-fastgs/specular/specular_model.py)
Replace the simplified 3-layer MLP with a wrapper around the baseline's `SpecularNetwork` / `SpecularNetworkReal` from `spec_utils.py`. This is the single most critical change.

The new `SpecularModel` will:
- **Wrap** `SpecularNetwork` (synthetic) or `SpecularNetworkReal` (real scenes) from `spec_utils.py`
- Accept `is_real` and `is_indoor` flags (matching baseline)
- Provide `train_setting()`, `save_weights()`, `load_weights()`, `update_learning_rate()` methods (matching baseline `scene/specular_model.py`)
- Use `get_linear_noise_func` for LR scheduling (matching baseline exactly)

**Input/output interface stays the same** (feat, viewdir, normal → RGB), but the internal architecture becomes physics-based ASG.

#### [MODIFY] [__init__.py](file:///d:/Thesis/All/spec-fastgs/specular/__init__.py)
Update import to expose the new `SpecularModel`.

---

### Component 3: Fix ASG Feature Dimension in GaussianModel

#### [MODIFY] [gaussian_model.py](file:///d:/Thesis/All/spec-fastgs/scene/gaussian_model.py)
- Change `self.specular_feat_dim = 8` → dynamic based on `asg_degree` parameter
- Add `asg_degree` parameter to `__init__`
- This ensures per-Gaussian feature vectors are the right size for the ASG network (24 for synthetic, 12 for real)

---

### Component 4: Fix Normal Computation with View-Dependent Flipping

#### [MODIFY] [normal_utils.py](file:///d:/Thesis/All/spec-fastgs/specular/normal_utils.py)
- Remove `@torch.no_grad()` — normals need gradient flow so the specular MLP can backprop through them (baseline detaches normal inside `train.py`, not in the computation itself)
- Use baseline's `get_minimum_axis` approach (sorted eigenvalue-based) instead of the loop
- Add `flip_align_view()` call — flip normal direction to face camera, matching baseline behavior
- Accept `viewdir` as argument so we can do the flip

---

### Component 5: Update Renderer to Pass Viewdir for Normal Flip

#### [MODIFY] [__init__.py](file:///d:/Thesis/All/spec-fastgs/gaussian_renderer/__init__.py)
- Pass viewdir to `compute_gaussian_normals()` so it can flip normals to face camera
- Detach normal before passing to specular MLP (matching baseline: `normal.detach()`)

---

### Component 6: Update Training Loop for Proper Specular Model Integration

#### [MODIFY] [train.py](file:///d:/Thesis/All/spec-fastgs/train.py)
- Use `SpecularModel(is_real, is_indoor)` with `train_setting(opt)` matching baseline
- Use `specular_model.update_learning_rate(iteration)` for LR scheduling
- Use `specular_model.optimizer` instead of a separate `spec_optimizer` (the model manages its own optimizer, like baseline)
- Pass `asg_degree` to `GaussianModel`
- Call `specular_model.save_weights()` at save iterations

---

### Component 7: Fix render.py to Load Correct Model

#### [MODIFY] [render.py](file:///d:/Thesis/All/spec-fastgs/render.py)
- Load `SpecularModel(is_real, is_indoor)` with `.load_weights(model_path)` matching new architecture
- Remove hardcoded `SpecularModel()` with no params

---

### Component 8: Update run script to pass `--use_specular`

#### [MODIFY] [run_spec-fastgs.sh](file:///d:/Thesis/All/spec-fastgs/run_spec-fastgs.sh)
Already updated — the active `render.py` line includes `--use_specular`. Just need to verify `--is_real` / `--is_indoor` flags are set for appropriate scenes.

> [!IMPORTANT]
> **The mip-NeRF 360 scenes (bicycle, counter, kitchen, etc.) are real outdoor/indoor scenes**. You need to pass `--is_real` and `--is_indoor` (for room, counter, kitchen, bonsai) during training AND rendering.

---

## Open Questions

> [!IMPORTANT]
> **Q1: Which scenes are you benchmarking on?** The ASG architecture differs between synthetic (SpecularNetwork: asg_degree=24, 4θ×8φ, hidden=128) and real (SpecularNetworkReal: asg_degree=12, 2θ×4φ, hidden=32). Mip-NeRF 360 scenes are "real" scenes. If you're only benchmarking on mip-NeRF 360, we should default to `is_real=True`.

> [!IMPORTANT]
> **Q2: Do you want to keep `train_specular.py` as a separate two-stage pipeline, or remove it?** The recommended approach (matching baseline) is joint training in `train.py`. The `train_specular.py` script becomes unnecessary. I suggest keeping it but marking it as deprecated.

> [!IMPORTANT]
> **Q3: Should I also add `--is_real` and `--is_indoor` flags to the run script for each scene?** Indoor scenes in mip-NeRF 360: room, counter, kitchen, bonsai. Outdoor: bicycle, flowers, garden, stump, treehill.

## Verification Plan

### Automated Tests
1. **Import test**: `python -c "from specular.spec_utils import SpecularNetwork"` — verifies `quaternion_utils` is found
2. **Model instantiation**: `python -c "from specular.specular_model import SpecularModel; m = SpecularModel(is_real=True); print(m)"` — verifies new model builds
3. **Dimension check**: Verify `asg_degree` → `specular_feat_dim` mapping is consistent between GaussianModel and SpecularModel
4. **Training smoke test**: Run training for 100 iterations on a small scene to verify no crashes and specular loss decreases

### Manual Verification
- After training, verify `specular/iteration_30000/specular.pth` exists
- Run `render.py --use_specular` and verify it loads without errors
- Compare PSNR/SSIM between spec-fastgs and baseline on the same scene
