# Custom Rules & Practices for LaTeX Thesis Compilation

## LaTeX & XeLaTeX Compilation Rules

1. **Strict 3-Step Compilation Order for Biber/BibLaTeX**:
   - Never run `biber main` immediately after deleting `.aux`/`.bcf` files.
   - Always follow the strict sequence:
     1. `xelatex -interaction=nonstopmode main.tex` (generates fresh `.bcf` and `.aux`)
     2. `biber main` (compiles `.bib` into `.bbl`)
     3. `xelatex -interaction=nonstopmode main.tex` (embeds citations and updates references)
   - If cite keys render as raw bold strings (e.g. `3DGS kerbl2023gaussian`), it means `biber main` has not synced with `main.bcf`.

2. **Custom Color Scope in `xcolor`**:
   - Custom colors used in captions (e.g. `\definecolor{bestblue}{RGB}{0,70,180}`) MUST be declared in `main.tex` **preamble** before `\begin{document}`.
   - Declaring custom colors inside chapter files (e.g. `chapter4.tex`) will crash `xelatex` with `! Package xcolor Error: Undefined color` when building the Table of Contents (`main.toc`) or List of Tables (`main.lot`).

3. **Table Width & Overfull `\hbox` Prevention**:
   - To prevent table lề (margin) overflow (`Overfull \hbox`), always wrap wide tabular environments in `\resizebox{\textwidth}{!}{ ... }` (requires `\usepackage{graphicx}`).

4. **Python Script Output Encodings on Windows**:
   - When printing log output or file paths containing Vietnamese Unicode characters in Python scripts executed via shell, avoid printing non-ASCII filenames directly to stdout without `.encode("ascii", "replace").decode("ascii")` or explicit UTF-8 output handling to prevent `UnicodeEncodeError`.
