# Spec-FastGS — Literature Review

A detailed, paper-by-paper review of the 50 collected references, written to feed
directly into the thesis report. It is organized by the same six categories as the PDF
library. For **every** paper you get:

- **Problem** — what limitation it attacks.
- **Key idea** — the one-sentence contribution.
- **Method (how)** — the actual mechanism, with the equations/components that matter.
- **Results** — headline numbers.
- **Relevance to Spec-FastGS** — exactly how to cite/use it in your thesis.
- A **figure** extracted from the PDF (architecture/teaser) where it clarifies the method.

Extracted figures live in `images/` (filename = paper stem). `_fig.png` = a figure pulled
from inside the paper; `_p1.png` = the paper's first page rendered (used when the key
figure is vector art that cannot be lifted as a raster).

## Contents

| File | Category | Papers |
|---|---|---|
| [00_foundational.md](00_foundational.md) | Radiance fields & the 3DGS base | 9 |
| [01_fast_densification.md](01_fast_densification.md) | Fast training & adaptive density control | 16 |
| [02_specular_reflection.md](02_specular_reflection.md) | Specular / reflective / view-dependent appearance | 13 |
| [03_quality_structure.md](03_quality_structure.md) | Anti-aliasing, surfaces, structured Gaussians | 5 |
| [04_method_blocks.md](04_method_blocks.md) | MLP / optimization building blocks | 5 |
| [05_surveys.md](05_surveys.md) | Surveys | 2 |

See [`../BIBLIOGRAPHY.md`](../BIBLIOGRAPHY.md) for the one-line citation index with arXiv
links and the `[CORE]/[REF]/[METHOD]/[COMPARE]` role of each paper.

## How the two pillars sit in this literature

**Spec-FastGS = FastGS (speed pillar) ⊕ Spec-Gaussian (appearance pillar)**, both built on
vanilla **3DGS**. The fast-densification chapter (01) positions the *speed* contribution;
the specular chapter (02) positions the *appearance/quality* contribution; the method
blocks (04) justify the specular-MLP architecture choices (SIREN, residual/hyper
connections, LoRA) and the pending hash-grid appearance field.
