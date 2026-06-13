# Spec-FastGS Thesis — Reference Library

Curated paper collection for the thesis **"Spec-FastGS: 3D Object Reconstruction via 3D
Gaussian Splatting"**, which unifies **FastGS** (multi-view-consistency densification,
fast training) with **Spec-Gaussian** (ASG latent + `SpecularNetwork` MLP specular
residual replacing SH high-frequency handling).

PDFs are organized into category folders. For each paper: **Role** indicates how it is
used in the report — `[CORE]` direct baseline/component the thesis builds on,
`[REF]` cited as related/background, `[METHOD]` an architectural/optimization technique
referenced in the design, `[COMPARE]` a competing approach for the related-work /
evaluation discussion.

---

## 00_foundational — radiance fields & the 3DGS base

| Paper | arXiv | Venue | Role | Relevance to thesis |
|---|---|---|---|---|
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al.) | [2308.04079](https://arxiv.org/abs/2308.04079) | SIGGRAPH 2023 | **[CORE]** | The base representation. Defines SH view-dependent color, adaptive density control (clone/split), differentiable rasterizer that both FastGS and Spec-Gaussian extend. |
| NeRF: Representing Scenes as Neural Radiance Fields (Mildenhall et al.) | [2003.08934](https://arxiv.org/abs/2003.08934) | ECCV 2020 | [REF] | Origin of neural novel-view synthesis; motivates the move to explicit Gaussians. |
| Instant-NGP: Multiresolution Hash Encoding (Müller et al.) | [2201.05989](https://arxiv.org/abs/2201.05989) | SIGGRAPH 2022 | [METHOD] | Hash-grid appearance field (pending Sol-7 contribution) derives from this. |
| Mip-NeRF 360 (Barron et al.) | [2111.12077](https://arxiv.org/abs/2111.12077) | CVPR 2022 | [REF] | Source of the Mip-NeRF 360 benchmark used for evaluation. |
| Plenoxels: Radiance Fields without Neural Networks | [2112.05131](https://arxiv.org/abs/2112.05131) | CVPR 2022 | [REF] | Explicit-representation predecessor; context for fast optimization. |
| TensoRF: Tensorial Radiance Fields | [2203.09517](https://arxiv.org/abs/2203.09517) | ECCV 2022 | [REF] | Factorized representation; background for low-rank ASG (Sol-6 LoRA idea). |
| Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields | [2304.06706](https://arxiv.org/abs/2304.06706) | ICCV 2023 | [REF] | State-of-the-art NeRF quality baseline for comparison tables. |
| COLMAP — Structure-from-Motion Revisited (Schönberger & Frahm) | (in folder) | CVPR 2016 | [REF] | SfM initialization 3DGS depends on. |
| Structure-from-Motion survey | (in folder) | 2017 | [REF] | Background on the SfM pipeline. |

## 01_fast_densification — training acceleration & adaptive density control

| Paper | arXiv | Venue | Role | Relevance to thesis |
|---|---|---|---|---|
| **FastGS: Training 3D Gaussian Splatting in 100 Seconds** | [2511.04283](https://arxiv.org/abs/2511.04283) | CVPR 2026 | **[CORE]** | The speed half of Spec-FastGS. Multi-view-consistency densification/pruning, no budget mechanism. |
| Taming 3DGS: High-Quality Radiance Fields with Limited Resources | [2406.15643](https://arxiv.org/abs/2406.15643) | SIGGRAPH Asia 2024 | [COMPARE] | Score-based budgeted densification; contrast to FastGS's budget-free scheme. |
| Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels | [2412.00578](https://arxiv.org/abs/2412.00578) | CVPR 2025 | [COMPARE] | Rasterizer-level speedups; complements densification-level speedups. |
| DashGaussian: Optimizing 3DGS in 200 Seconds | [2503.18402](https://arxiv.org/abs/2503.18402) | CVPR 2025 | [COMPARE] | Resolution-scheduling speed baseline (FastGS compares against it). Deliberately dropped from our pipeline. |
| 3DStudent / efficient student densification | [2503.10148](https://arxiv.org/abs/2503.10148) | 2025 | [COMPARE] | Another efficiency competitor. |
| Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians | [2403.14166](https://arxiv.org/abs/2403.14166) | ECCV 2024 | [COMPARE] | Depth-reinit + densification control; relates to our visibility gating. |
| Mini-Splatting2: Building 360 Scenes within Minutes | [2411.12788](https://arxiv.org/abs/2411.12788) | 2024 | [COMPARE] | Aggressive densification for fast convergence. |
| AbsGS: Recovering Fine Details (homodirectional gradient) | [2404.10484](https://arxiv.org/abs/2404.10484) | ACM MM 2024 | [METHOD] | Gradient-based densification criterion; motivates multi-view-consistency scoring. |
| Pixel-GS: Density Control with Pixel-aware Gradient | [2403.15530](https://arxiv.org/abs/2403.15530) | ECCV 2024 | [METHOD] | Pixel-coverage-weighted gradients for densification. |
| Compact 3D Gaussian Representation | [2311.13681](https://arxiv.org/abs/2311.13681) | CVPR 2024 | [REF] | Compression/pruning; context for storage of per-Gaussian ASG latent. |
| EAGLES: Efficient Accelerated 3D Gaussians with Lightweight Encodings | [2312.04564](https://arxiv.org/abs/2312.04564) | ECCV 2024 | [REF] | Quantized attributes for speed/memory. |
| Revising Densification in Gaussian Splatting (Bulò et al.) | [2404.06109](https://arxiv.org/abs/2404.06109) | 2024 | [METHOD] | Opacity-correction + GOF densification metric; design reference for density control. |
| Efficient Density Control for 3D Gaussian Splatting | [2411.10133](https://arxiv.org/abs/2411.10133) | 2024 | [METHOD] | Long-axis split + recovery-aware pruning. |
| GaussianPro: 3DGS with Progressive Propagation | [2402.14650](https://arxiv.org/abs/2402.14650) | ICML 2024 | [METHOD] | Patch-match progressive densification; related multi-view-consistency idea. |
| 3D Gaussian Splatting as Markov Chain Monte Carlo | [2404.09591](https://arxiv.org/abs/2404.09591) | NeurIPS 2024 | [METHOD] | Reframes densification as MCMC sampling; theoretical context. |
| LightGaussian: Unbounded 3DGS Compression | [2311.17245](https://arxiv.org/abs/2311.17245) | NeurIPS 2024 | [REF] | Importance pruning + distillation for compact models. |
| CompGS: Smaller and Faster Gaussian Splatting with Vector Quantization | [2311.18159](https://arxiv.org/abs/2311.18159) | ECCV 2024 | [METHOD] | VQ codebook for per-Gaussian attributes — basis for the VQ-ASG latent contribution (solution research 2026-06-13). |

## 02_specular_reflection — view-dependent / glossy / reflective appearance

| Paper | arXiv | Venue | Role | Relevance to thesis |
|---|---|---|---|---|
| **Spec-Gaussian: Anisotropic View-Dependent Appearance for 3DGS** | [2402.15870](https://arxiv.org/abs/2402.15870) | NeurIPS 2024 | **[CORE]** | The specular half of Spec-FastGS. ASG appearance field + MLP replacing SH; coarse-to-fine training. Direct parent of `SpecularNetwork`. |
| GaussianShader: 3DGS with Shading Functions for Reflective Surfaces | [2311.17977](https://arxiv.org/abs/2311.17977) | CVPR 2024 | [COMPARE] | Shading-function reflective GS; alternative to ASG residual. |
| 3D Gaussian Splatting with Deferred Reflection | [2404.18454](https://arxiv.org/abs/2404.18454) | SIGGRAPH 2024 | [METHOD] | Per-pixel deferred specular shading + normal propagation; cited for MLP-output design. |
| GS-IR: 3D Gaussian Splatting for Inverse Rendering | [2311.16473](https://arxiv.org/abs/2311.16473) | CVPR 2024 | [REF] | BRDF/lighting decomposition with GS. |
| IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing | [2412.15867](https://arxiv.org/abs/2412.15867) | CVPR 2025 | [REF] | Full rendering equation via ray tracing; advanced reflection. |
| GS-ROR: Reflective Object Relighting via SDF Priors | [2406.18544](https://arxiv.org/abs/2406.18544) | 2024 | [REF] | Deferred GS + SDF for reflective relighting. |
| RTR-GS: Inverse Rendering with Radiance Transfer and Reflection | [2507.07733](https://arxiv.org/abs/2507.07733) | ACM MM 2025 | [REF] | Hybrid radiance/PBR for arbitrary reflectance. |
| VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting | [2501.17978](https://arxiv.org/abs/2501.17978) | 2025 | [METHOD] | View-dependent opacity for specular; closely related to our view-gated specular. |
| Reflective Gaussian Splatting (Ref-Gaussian) | [2412.19282](https://arxiv.org/abs/2412.19282) | ICLR 2025 | [COMPARE] | PBR deferred rendering + Gaussian inter-reflection; SOTA reflective GS. |
| Relightable 3D Gaussians (BRDF + ray tracing) | [2311.16043](https://arxiv.org/abs/2311.16043) | ECCV 2024 | [REF] | Per-Gaussian normal/BRDF/lighting; relightable extension. |
| Ref-NeRF: Structured View-Dependent Appearance | [2112.03907](https://arxiv.org/abs/2112.03907) | CVPR 2022 (Best Paper) | [METHOD] | Reflected-radiance reparameterization + IDE; intellectual ancestor of specular modeling. |
| NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections | [2405.14871](https://arxiv.org/abs/2405.14871) | SIGGRAPH Asia 2024 | [REF] | Reflection-ray casting for consistent specular. |
| SpecNeRF: Gaussian Directional Encoding for Specular Reflections | [2312.13102](https://arxiv.org/abs/2312.13102) | CVPR 2024 | [METHOD] | Learnable Gaussian directional encoding for near-field specular; cited for encoding design. |
| PhySG: Inverse Rendering with Spherical Gaussians | [2104.00674](https://arxiv.org/abs/2104.00674) | CVPR 2021 | [METHOD] | Spherical-Gaussian mixture BRDF + lighting — explicit high-specularity lobe basis (solution research 2026-06-13). |
| GlossGau: Inverse Rendering for Glossy Surface with Anisotropic Spherical Gaussian | [2502.14129](https://arxiv.org/abs/2502.14129) | 2025 | [REF] | Anisotropic SG for glossy surfaces; closest to our ASG lobe family. |
| NeRF in the Dark (RawNeRF): HDR View Synthesis from Noisy Raw Images | [2111.13679](https://arxiv.org/abs/2111.13679) | CVPR 2022 | [METHOD] | HDR/log tone-mapped loss — fix for the specular dimming (a*>1) deficit. |
| HDR-NeRF: High Dynamic Range Neural Radiance Fields | [2111.14451](https://arxiv.org/abs/2111.14451) | CVPR 2022 | [REF] | Differentiable tone mapper; context for highlight HDR handling. |
| GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction | [2506.13110](https://arxiv.org/abs/2506.13110) | 2025 | [REF] | Foundation-model normal supervision for reflective objects. |

## 03_quality_structure — anti-aliasing, surface, structured Gaussians

| Paper | arXiv | Venue | Role | Relevance to thesis |
|---|---|---|---|---|
| Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering | [2312.00109](https://arxiv.org/abs/2312.00109) | CVPR 2024 | [METHOD] | Anchor + MLP-decoded attributes; basis for the hash-grid appearance field (Sol-7). |
| Mip-Splatting: Alias-free 3D Gaussian Splatting | [2311.16493](https://arxiv.org/abs/2311.16493) | CVPR 2024 (Best Student Paper) | [METHOD] | 3D smoothing + 2D Mip filter; quality-regularization reference. |
| 2D Gaussian Splatting for Geometrically Accurate Radiance Fields | [2403.17888](https://arxiv.org/abs/2403.17888) | SIGGRAPH 2024 | [REF] | Surfel Gaussians; used by several reflective-GS methods above. |
| Gaussian Opacity Fields | [2404.10772](https://arxiv.org/abs/2404.10772) | SIGGRAPH Asia 2024 | [REF] | Surface extraction + improved densification metric. |
| SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing | [2403.19615](https://arxiv.org/abs/2403.19615) | 2024 | [REF] | Test-time anti-aliasing alternative to Mip-Splatting. |
| FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization | [2403.06908](https://arxiv.org/abs/2403.06908) | CVPR 2024 | [METHOD] | Fourier amplitude+phase loss, low→high annealing — de-blur fix for the specular σ deficit (solution research 2026-06-13). |
| DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing | [2403.17822](https://arxiv.org/abs/2403.17822) | WACV 2025 | [METHOD] | Monocular normal-prior supervision — fix for the noisy min-axis pseudo-normal (NCC deficit). |

## 04_method_blocks — architectural / optimization techniques used in the MLP & pipeline

| Paper | arXiv | Venue | Role | Relevance to thesis |
|---|---|---|---|---|
| SIREN: Implicit Neural Representations with Periodic Activation Functions | [2006.09661](https://arxiv.org/abs/2006.09661) | NeurIPS 2020 | [METHOD] | Sine activations for high-frequency signals — proposed specular-MLP swap (Option 3). |
| Hyper-Connections | [2409.19606](https://arxiv.org/abs/2409.19606) | ICLR 2025 | [METHOD] | Learnable depth/width connection mixing — specular-MLP swap (Option 4). |
| Residual Connections Encourage Iterative Inference | [1710.04773](https://arxiv.org/abs/1710.04773) | ICLR 2018 | [METHOD] | Theory behind the residual skip in `ASGRender` (Option 2). |
| Deep Residual Learning (ResNet) | [1512.03385](https://arxiv.org/abs/1512.03385) | CVPR 2016 | [METHOD] | Original residual connections; foundation for Option 2/5. |
| LoRA: Low-Rank Adaptation of Large Language Models | [2106.09685](https://arxiv.org/abs/2106.09685) | ICLR 2022 | [METHOD] | Low-rank factorization — basis for low-rank ASG (Sol-6 contribution). |
| Focal Frequency Loss for Image Reconstruction and Synthesis | [2012.12821](https://arxiv.org/abs/2012.12821) | ICCV 2021 | [METHOD] | Adaptive frequency-domain loss up-weighting hard frequencies — top de-blur loss candidate (solution research 2026-06-13). |
| Fourier Features Let Networks Learn High Frequency Functions | [2006.10739](https://arxiv.org/abs/2006.10739) | NeurIPS 2020 | [METHOD] | NTK-grounded positional encoding; defeats spectral bias on view dir. |
| WIRE: Wavelet Implicit Neural Representations | [2301.05187](https://arxiv.org/abs/2301.05187) | CVPR 2023 | [METHOD] | Gabor-wavelet activation; high-freq INR alternative to SIREN in ASGRender. |
| Deep Laplacian Pyramid Networks (LapSRN) | [1704.03915](https://arxiv.org/abs/1704.03915) | CVPR 2017 | [METHOD] | Multi-scale Laplacian detail loss — high-freq supervision. |
| Online Hard Example Mining (OHEM) | [1604.03540](https://arxiv.org/abs/1604.03540) | CVPR 2016 | [METHOD] | Hard-example mining — re-balance the 95/5 diffuse/highlight pixel imbalance. |
| Focal Loss for Dense Object Detection (RetinaNet) | [1708.02002](https://arxiv.org/abs/1708.02002) | ICCV 2017 | [METHOD] | (1-p)^γ down-weighting of easy samples — pixel-imbalance loss reweighting. |
| FiLM: Visual Reasoning with a General Conditioning Layer | [1709.07871](https://arxiv.org/abs/1709.07871) | AAAI 2018 | [METHOD] | Feature-wise modulation/hypernetwork — efficient per-Gaussian latent conditioning. |
| KAN: Kolmogorov–Arnold Networks | [2404.19756](https://arxiv.org/abs/2404.19756) | ICLR 2025 | [METHOD] | Learnable edge functions; novel high-freq architecture angle. |

## 05_surveys — background & positioning

| Paper | arXiv | Venue | Role | Relevance to thesis |
|---|---|---|---|---|
| A Survey on 3D Gaussian Splatting | [2401.03890](https://arxiv.org/abs/2401.03890) | TPAMI 2025 | [REF] | Taxonomy for the related-work chapter. |
| 3D Gaussian Splatting: Survey, Technologies, Challenges, Opportunities (Fei et al.) | [2403.11134](https://arxiv.org/abs/2403.11134) | 2024 | [REF] | Complementary survey for positioning the contribution. |

---

*Generated for the Spec-FastGS thesis reference collection. PDFs downloaded from arXiv.*

*Update 2026-06-13: added 16 papers (FFL, Fourier-features, WIRE, OHEM, Focal Loss, FiLM,
KAN, LapSRN, FreGS, DN-Splatter, CompGS, RawNeRF, HDR-NeRF, PhySG, GlossGau, GS-2DGS)
from the cross-field solution research for the specular-energy deficit. Full catalogue
with per-solution rationale and ranked recommendations: `results/SOLUTIONS_RESEARCH_2026-06-13.md`.*
