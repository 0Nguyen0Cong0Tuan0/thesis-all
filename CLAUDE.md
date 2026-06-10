# CLAUDE.md

## Project: Spec-FastGS (3D Object Reconstruction via 3DGS)

Thesis project that unifies **FastGS** (multi-view-consistency densification, fast
training) with **Spec-Gaussian** (replaces SH high-frequency handling with a 24-dim
per-Gaussian ASG latent + a `SpecularNetwork` MLP producing a specular residual added
to `sh_color`).

Key behaviors:
- Specular branch activates after iter 3000.
- SH gradients are shielded ×0.01 during iters 3000–5000.
- DashGaussian was deliberately dropped.
- ASG integration is already wired up and working (see repo `task.md`).

## Current phase: pipeline optimization

Targeting two bottlenecks:

1. **Speed** — training inflated from ~4 min to ~30 min, caused by:
   - 3 rasterization passes/iter in `train.py`.
   - Running `SpecularNetwork` on ALL N Gaussians (including invisible ones) every iter.
2. **Quality** — SSIM/LPIPS regression caused by:
   - Adaptive top-5% L2 mask with `spec_weight=2.0`.
   - Hard SH gradient freeze.

## Prioritized fixes

1. Eliminate the 2 redundant render passes.
2. Visibility-gated MLP (run only on `radii>0`).
3. Laplacian-pyramid L1 loss + drop `spec_weight` 2.0 → 0.5.
4. Soft cosine SH gradient decay (instead of hard freeze).
5. Stale MLP cache, K=4.
6. LoRA low-rank ASG factorization.
7. Hash-grid appearance field.

**Quick-win batch** = fixes 1+2+3+4 (~50 lines), targeting 30 min → ~8–12 min plus
recovered SSIM/LPIPS. Fixes **6 and 7** are flagged as the strongest novel thesis
contributions.

## Specular MLP architecture options

Options for changing the specular MLP architecture, all drop-in swaps sharing the same
`ASGRender` interface (`forward(pts, viewdirs, features, normal) → [N,3]`) in
`spec-fastgs/utils/spec_utils.py`.

### Fix first (bugs/mismatches vs Specular-Gaussians baseline)
- **Output activation**: spec-fastgs applies `sigmoid` (clamps to [0,1]); baseline uses
  **none** (unbounded). Sigmoid kills bright HDR specular peaks — likely a large chunk
  of the quality gap on specular regions.
- **`SpecularNetworkReal.asg_feature`**: spec-fastgs uses 24-dim; baseline uses **12-dim**
  (spec-fastgs doubles per-Gaussian ASG storage in real mode).

### Ranked options
1. **Option 1 — Remove sigmoid** *(1 line, try first)* — `spec_utils.py:124`,
   `rgb = torch.sigmoid(self.mlp(mlp_input))` → `rgb = self.mlp(mlp_input)`.
2. **Option 2 — Residual connection in `ASGRender`** *(paper 1710.04773)* — replace
   `nn.Sequential` with explicit forward and a skip: `h2 = F.relu(self.fc2(h1)) + h1`.
3. **Option 3 — SIREN activation** *(best for high-frequency specular)* — replace `ReLU`
   with `sin(ω₀·x)` (SineLayer + SIREN init); needs weight re-init / fresh training.
   Specular highlights are high-frequency view-dependent → SIREN is designed for this.
4. **Option 4 — Hyper-Connection MLP** *(paper 2409.19606, ICLR 2025)* — use n=2 (not 4;
   the MLP is only 3 layers). Static variant (SHC) is sufficient: learnable A_r/A_m/B
   matrices mixing n copies of the hidden state.
5. **Option 5 — Deeper + wider MLP** *(more capacity)* — `80→128→128→3` →
   `80→256→256→256→3` with residual connections.

### Recommended experiment order
1. Remove sigmoid alone — see if it closes the gap with no architecture change.
2. Sigmoid removed + residual skip.
3. Sigmoid removed + SIREN — highest expected gain on specular.
4. Sigmoid removed + Hyper-Connections (n=2) — strongest theoretical backing.

**Status (from prior session):** Option 1 done; moved on to Option 2.

Sources: Residual Connections (arXiv:1710.04773), Hyper-Connections (arXiv:2409.19606),
SpecNeRF (arXiv:2312.13102), 3DGS with Deferred Reflection (arXiv:2404.18454).

## Reference artifacts

Antigravity research:
- `C:\Users\nguye\.gemini\antigravity\brain\3c91828b-1b1a-4b1e-9717-55469dd3c185\artifacts\pipeline_optimization_research.md`
- `C:\Users\nguye\.gemini\antigravity\brain\3c91828b-1b1a-4b1e-9717-55469dd3c185\artifacts\thesis_and_codebase_analysis.md`
