# PBR-for-Gaussian-Splatting research vs our ASG pipeline — strategic analysis

**Date:** 2026-06-30
**Trigger:** teapot R7 (spec_densify) result + 6 new PBR papers in `papers/new/`.

## 0. First: the R7 (spec_densify) verdict — it failed, and the idea is now scooped
Teapot A/B (both v3.0, only diff = `--spec_densify`; clean isolation):

| | base | r7 (spec_densify) | Δ |
|---|---|---|---|
| PSNR | **34.46** | 34.05 | −0.41 |
| SSIM | 0.9829 | 0.9821 | −0.0008 |
| LPIPS | 0.0311 | 0.0326 | +0.0015 |
| NCC | 0.935 | 0.925 | −0.010 |
| energyRatio | 0.849 | 0.840 | −0.009 |
| best σ | 1.10 | 1.32 | +0.22 (blurrier) |
| #Gaussians | 149,096 | 148,716 | ≈ equal (NO reduction) |

Negative on **every** axis, and it did not even reduce the Gaussian count (its core promise).
Root reason: on a clean synthetic specular scene the **baseline is already excellent**
(NCC 0.935 vs counter's ~0.52), so there is no allocation deficit for spec_densify to fix —
it only perturbs a good solution. **Plus it is scooped:** Aniso-GS (Computers & Graphics 2026)
already does "adaptive densification: suppress Gaussian growth in low-frequency regions,
promote refinement in high-frequency regions, *without increasing the total number of
Gaussians*", evaluated on the **same Anisotropic Synthetic dataset** (+5.86 dB over 3DGS).
→ **spec_densify is dead as the headline contribution.**

## 1. The 6 papers — the field has converged on PBR deferred reflection

| Paper (venue) | Core idea | Substrate | Relevance to us |
|---|---|---|---|
| **ARS-GS** (MDPI 2026) | **ASG** reflection + SH diffuse in a **PBR** pipeline, **skip connection** ASG↔Gaussian | 3DGS | Closest to us — keeps ASG, adds PBR. SOTA: PSNR 38.30 NeRF-syn, 46.31 Gloss Blender, 26.26 Ref-NeRF Sedan |
| **Ref-Gaussian** (ICLR'25-ish) | **PBR deferred** rendering (split-sum) + **Gaussian inter-reflection** + material-aware normal propagation | **2DGS** | "real-time + fast optimization + compute efficient", unified reflective/non-reflective |
| **MaterialRefGS** (NeurIPS'25) | **Multi-view-consistent material** inference + **ray-traced** env; tracks photometric variation across views to find reflective regions | **2DGS deferred** | The "track cross-view variation to locate reflective regions" = our diagnostic idea, productized |
| **RGS-DR** (2026) | **Deferred reflections + residual shading**, pixel-deferred surfel + specular residual | **2DGS** | **Explicitly says per-Gaussian shortest-axis-normal + normal-residual is NOISY → deferred is better.** This is exactly our v2.9-R6 failure, independently confirmed |
| **AugGS** (ICLR 2026) | View-dependent **opacity** kernel for specular + error-driven compensation; **plug-in** post-enhancement; sh=2 enough | 2DGS init | Parameter-efficient angle; pluggable |
| **Aniso-GS** (C&G 2026) | Adaptive **SH** appearance field + multi-view smoothing + **frequency-aware densification** | 3DGS | Scoops our densification idea, same dataset |

**The consensus recipe for SOTA specular now = 2DGS surfels + DEFERRED (pixel-level) shading
+ split-sum environment + per-Gaussian materials (albedo/roughness/metallic) + optional
ray-traced inter-reflection.** Forward per-Gaussian shading (what our pipeline does) is a
generation behind.

## 2. What this means for us (honest)
- Our two failed root-cause attacks now have an explanation: the *substrate* is wrong.
  - Root cause B (placement / noisy shortest-axis normals): RGS-DR shows this is intrinsic to
    per-Gaussian forward shading and is **fixed by deferred shading**, not by supervising the
    noisy normal (which we tried 3× and failed, R4–R6). Our negative result is now corroborated
    by a 2026 paper.
  - Root cause A (magnitude): PBR materials + split-sum env represent specular energy far better
    than an additive ASG residual.
- ARS-GS proves **ASG is still viable** — but only when embedded in a PBR pipeline. So our ASG
  investment isn't wasted; the missing piece is the PBR + deferred-shading scaffolding.

## 3. The opportunity (novel + matches the stated goal: metrics ↑, time/memory ↓)
Every PBR paper above optimizes **quality**; almost none target **fast training / low memory /
few Gaussians**. FastGS (our heritage) is exactly that and is **absent** from the PBR line of
work. That gap is the contribution:

> **"Fast, lightweight deferred-PBR specular"** — PBR-grade reflective quality at FastGS training
> speed, low memory, and far fewer Gaussians.

Mechanism sketch: deferred ASG/PBR specular (the proven quality lever; fixes placement the
deferred way) + FastGS multi-view densification (speed) + compact (low-rank) ASG/material latent
(memory) + our specular diagnostic (analysis). It directly repairs both failed root causes by
changing the substrate, and the efficiency axis is wide open.

## 4. Three strategic options (decision needed)
- **(A) Re-architect to deferred-PBR specular ourselves** (keep ASG, deferred shading, split-sum
  env, 2DGS) + FastGS speed + compactness. Highest ceiling, biggest effort/risk; competing with
  NeurIPS/ICLR PBR on quality is hard solo on Kaggle.
- **(B) Build ON an open-source PBR baseline** (ARS-GS or Ref-Gaussian both release code) and add
  our **FastGS densification + compact latent + diagnostic** → an **efficiency** contribution
  ("same PBR quality, much less compute/memory"). Lower risk, leverages their code, still novel
  on the axis the user cares about. **Recommended.**
- **(C) Reposition as an efficiency + diagnostic study** (no new-SOTA-quality claim): the
  diagnostic + the characterized negatives + a speed/memory Pareto. Most achievable; lower venue
  (3DV/BMVC/WACV) than CVPR.

## 5. Recommendation
**Option B.** Take the closest-to-us open PBR method (lead candidate **Ref-Gaussian**: deferred,
2DGS, fast, unified, code available; **ARS-GS** as the ASG-native alternative), reproduce its
baseline on the Spec-Gaussian datasets we already wired, then contribute the **FastGS-speed +
compact-memory** version + our diagnostic. Headline = "PBR specular quality at a fraction of the
training time / memory / Gaussian count." This is the only framing where a solo Kaggle-budget
project can land a big, defensible number against 2026 SOTA — and it is exactly metrics-up /
cost-down.

Next concrete step if B: pick the base method, clone + reproduce one scene, measure its
time/memory/#GS as the target to beat, then port FastGS densification into it.
