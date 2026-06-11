# Spec-FastGS — Results Analysis & Ultra-Optimization Plan

Scene: `counter` (mip-NeRF 360). Run: `spec-fastgs_v2.2_nct`, 30k iters.
Final: **PSNR 30.72 / SSIM 0.925 / LPIPS 0.166**, **122m 29s**, 619,823 Gaussians, 9.7 GB.

---

## 1. Diagnostic analysis (adapted `analysis/analyze_results.py`, all 30 test views)

| Metric | Value | Reading |
|---|---|---|
| render/GT Laplacian-var ratio | **0.599** | render is ~40 % **blurrier** than GT |
| radial spectrum ratio, high band | **0.655** | loses 35 % of high-freq detail |
| RMSE dark / mid / bright | 4.8 / 8.5 / **17.6** | error concentrated in **bright/specular** |
| highlight RMSE (top-5 % residual) | **26.3** | worst region by far |
| highlight luminance bias | **−7.75** | highlights rendered **too dim** |
| specular energyRatio (render/GT) | **0.674** | **under-produces** specular by 33 % |
| specular error split | **98.5 % structural / 1.5 % magnitude** | NOT a brightness bug — it's blur/misplacement |
| specular best-match blur σ | **3.15 px** | render specular is over-softened |
| specular NCC | 0.657 | highlights moderately misplaced |
| worst views | 00003, 00029 | strongly specular; bias −22 / −11, NCC ≈ 0.49 |

**Conclusion:** quality is already good globally; the remaining gap is almost entirely
**high-frequency specular highlights that are blurred, spread (σ≈3 px), dim, and slightly
misplaced.** This is a *representation/structural* limit, not a tuning/brightness issue.

Visualizations written to `results/analysis_out/`:
`00001_errheat.png`, `00001_speclayer_GT_vs_render.png`.

---

## 2. Why training is 122 min (root causes in `spec-fastgs/train.py`)

1. **A second full render pass is STILL in the loop** (`train.py:214-248`, "DECOUPLED
   DENSIFICATION"). From iter 3000→`densify_until` it does an extra SH-only
   `render_fastgs` **plus a `torch.autograd.grad`** every iteration (~12k times).
   This roughly **doubles** per-iter cost in the densification phase. `task.md` claims
   redundant passes were removed — they were not in the run that produced these results.
2. **Visibility-cache thrashing** (`train.py:125-149`). The sparse-MLP mask is invalidated
   whenever the Gaussian count changes — which is **every densification (every 100 iters)**
   during 0→15k, plus first-sight per camera. So during the whole densification phase the
   MLP runs on **all ~600k** Gaussians, not the intended 10-30 % subset. The optimization
   silently doesn't fire when it matters most.
3. **Multi-view importance scoring** (`compute_gaussian_score_fastgs`, `train.py:312`) renders
   several cameras every `densification_interval` (100) iters.
4. **Over-densification: 619k Gaussians** (~3-4× a typical FastGS `counter`). Every cost above
   scales linearly with N. The spec branch + 2nd-pass geometry gradients drive the bloat.

---

## 3. Ultra-optimization plan (speed ↓ without losing score)

### Tier 1 — pure speed, ~zero quality risk (target 122 → ~45-55 min)
- **S1. Delete the 2nd render pass.** Reuse the single forward render's
  `viewspace_point_tensor`/`radii` for densification stats. If SH-only geometry gradients
  are wanted, take them from the main render graph (one backward), not a fresh render.
  *Biggest single win (~40 % of wall-clock in the densify phase).*
- **S2. Make the sparse MLP actually fire.** Either (a) gate the specular branch to run only
  **after** `densify_until_iter` (count is then stable, cache hits 90 %+), or (b) rebuild the
  visibility mask from *this* iteration's `radii` and reallocate on count change instead of
  falling back to the full set. Specular only activates at 3000 anyway; deferring its MLP to
  post-densification barely touches quality but removes it from the most expensive phase.
- **S3. Stale-MLP cache, K=4** (`task.md` Sol 3). Specular residual drifts slowly — refresh
  `mlp_color` every 4 iters, reuse otherwise. ~3-4× fewer MLP forward/backward calls.

### Tier 2 — speed + curbs over-densification (target → ~25-35 min)
- **S4. Coarse-to-fine / progressive resolution** — Spec-Gaussian's *own* strategy
  (and DashGaussian's scheduler). Train low-res early, raise to full res later. Cuts early
  rasterization cost **and**, per Spec-Gaussian, "prevents the need to increase the number of
  3D Gaussians" → directly attacks the 619k bloat.
- **S5. Tighten the Gaussian budget.** FastGS VCP should keep counts low; 619k says pruning is
  too soft here. Add a budget cap (Taming-3DGS) or stronger VCP pruning, target ~250k. Linear
  speedup on raster **and** MLP, and (per FastGS) comparable quality.

### Tier 3 — the thesis-grade move: kill the per-Gaussian MLP entirely
- **S6. Deferred (screen-space) specular shading** — *3DGS-DR* (SIGGRAPH'24) /
  *Ref-Gaussian* (ICLR'25). Rasterize base-color + normal + reflection-strength maps once,
  then shade **per pixel** in a single screen-space pass (env-map lookup or a tiny shared MLP).
  Cost becomes **independent of N** and the MLP runs ~1M pixels once instead of ~600k Gaussians
  every view. **This is the "ultra" solution: it fixes BOTH problems at once** —
  - speed: removes the per-Gaussian-per-view MLP that is the core inflation;
  - quality: per-pixel shading yields **sharp** highlights, directly fixing the blur
    (σ≈3.15) and under-production (energyRatio 0.674) the analysis found.
  Ref-Gaussian is MLP-free and reports faster train+render *and* higher metrics than ASG-MLP
  approaches. Strongest novel contribution; bigger lift than LoRA/hash-grid (CLAUDE.md 6/7).

### MLP-improvement options (if keeping the ASG-MLP path)
The analysis says the failure mode is *high-frequency* specular. ReLU MLPs are low-frequency
biased, so:
- **SIREN activation** (sin, paper 2006.09661) in `ASGRender` — built for high-freq view-
  dependent signals; directly targets the blur/under-production. (CLAUDE.md Option 3.)
- Sigmoid removal (done, `spec_utils.py:122`) and residual skip (done, `:121`) are already in;
  they help magnitude but the residual gap is *structural*, which SIREN/deferred-shading address.

---

## 4. Recommended order
1. **S1 + S2** (≈30 lines): ~122 → ~50 min, no quality change. Do first, re-measure.
2. **S4 (coarse-to-fine) + S5 (budget)**: → ~30 min and fewer Gaussians.
3. **S6 deferred shading** (or SIREN if staying with the MLP): recovers the specular sharpness
   the analysis flags, and for S6 also removes the MLP cost permanently — the thesis headline.

Sources: FastGS (arXiv:2511.04283), Spec-Gaussian (arXiv:2402.15870), Taming-3DGS
(arXiv:2406.15643), 3DGS Deferred Reflection (arXiv:2404.18454), Reflective Gaussian / Ref-Gaussian
(ICLR 2025), SIREN (arXiv:2006.09661).
