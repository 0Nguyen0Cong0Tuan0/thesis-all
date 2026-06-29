# Execution plan — "Ref-Gaussian quality at FastGS cost" (CVPR contribution)

**Date:** 2026-06-30
**Decision:** Option B — build ON Ref-Gaussian (open PBR baseline), add our efficiency angle.
Base repo: `github.com/fudan-zvg/ref-gaussian` (MIT, 118★, active).

## The thesis (one sentence)
Ref-Gaussian and the 2026 PBR line achieve SOTA reflective quality but are quality-first
(2DGS deferred + split-sum + ray-traced inter-reflection); **none target fast training / low
memory / few Gaussians.** We bring **FastGS multi-view-consistency densification + a compact
material/ASG latent + our specular diagnostic** into the deferred-PBR pipeline → **match (or
near) Ref-Gaussian quality at a fraction of the training time, memory, and #Gaussians.**

## Why this fixes what we couldn't before
- Placement (root cause B): deferred (pixel-level) shading removes the noisy shortest-axis-normal
  problem — RGS-DR (2026) confirms this is the right fix; our R4–R6 normal-supervision failures
  were fighting the wrong substrate.
- Magnitude (root cause A): PBR materials + split-sum env represent specular energy natively
  (no additive ASG residual hack).
- Our value-add is the axis the PBR papers ignore: **cost** (FastGS heritage).

## Environment reality (separate from our spec-fastgs stack)
Ref-Gaussian needs its OWN env: torch==2.0.0, python 3.8, submodules:
`diff-surfel-rasterization` (2DGS), `cubemapencoder`, `simple-knn`, `raytracing`.
- The Kaggle notebook we built (torch 1.13.1 + FastGS rasterizer) does NOT apply here — a new
  Kaggle setup is required.
- **`raytracing` (OptiX-style) is the build risk on Kaggle T4** and powers only inter-reflection
  (the most expensive, optional component). Plan: get the core deferred-PBR path running FIRST
  (no inter-reflection); treat inter-reflection as optional. Disabling it also *supports* our
  efficiency story.

## Datasets (Ref-Gaussian's, for direct comparability to its tables)
- **Shiny Blender** (synthetic + real) — Verbin et al., small/standard.
- **Glossy Synthetic** (NeRO) — needs `nero2blender.py` conversion.
- **NeRF Synthetic** — standard.
(Our previously-wired Anisotropic Synthetic / NSVF become secondary / extra evidence.)
→ Action: upload Shiny Blender (+ Glossy Synthetic) to Kaggle as a dataset.

## Phases
- **PBR-0 — reproduce baseline.** Stand up Ref-Gaussian on Kaggle; train ONE scene (e.g. Shiny
  Blender `toaster` or `helmet`), core path (no inter-reflection if raytracing won't build).
  Record PSNR/SSIM/LPIPS + **training time, peak memory, #Gaussians, FPS** = the numbers to beat.
- **PBR-1 — port FastGS densification.** Replace Ref-Gaussian's gradient-based 2DGS densification
  with FastGS multi-view-consistency densification (reimplement our vote for the 2DGS rasterizer).
  Target: fewer Gaussians + faster training at equal quality. (Our core contribution.)
- **PBR-2 — compact representation.** Low-rank/compact material+ASG latent for memory; optional
  view-dependent-opacity trick (AugGS) for parameter efficiency.
- **PBR-3 — diagnostic + benchmark + ablation.** Port the specular diagnostic; full table
  (ours vs Ref-Gaussian / 3DGS-DR / Spec-Gaussian) on Shiny Blender + Glossy Synthetic, reporting
  the quality AND cost columns; ablate each component.

## Headline table (target shape)
| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Train time↓ | Peak mem↓ | #GS↓ | FPS↑ |
Ours should win the right three (time/mem/#GS) while matching PSNR/SSIM/LPIPS within noise.

## Risks / honesty
- New env + 2DGS rasterizer + (maybe) raytracing build on Kaggle T4 — the setup itself is the
  first real risk; budget time for it.
- Competing on raw quality with NeurIPS/ICLR PBR is hard; our claim must be **efficiency at
  equal quality**, not new-best quality. Keep the framing disciplined.
- Much of the existing spec-fastgs code (forward ASG MLP, normal experiments) is NOT carried over;
  what transfers is the *idea* of FastGS multi-view densification + the diagnostic methodology.

## Immediate next step
Stand up Ref-Gaussian on Kaggle and reproduce one scene (PBR-0). Needs: (a) Shiny Blender on
Kaggle, (b) a Kaggle setup notebook for the torch-2.0 + 2DGS env. Draft both next.
