---
name: latex-thesis-compilation
description: Standard workflow and troubleshooting guidelines for compiling LaTeX/XeLaTeX thesis documents with BibLaTeX, Biber, xcolor, and multi-column tables.
---

# LaTeX Thesis Compilation Skill & Troubleshooting Guide

## Standard 3-Step Compilation Workflow

When compiling `main.tex` after editing `.tex` files or references:

1. **Step 1: Initial XeLaTeX Pass**
   Generate fresh `.aux`, `.toc`, `.lot`, `.bcf` files:
   ```bash
   xelatex -interaction=nonstopmode main.tex
   ```

2. **Step 2: Bibliography Compilation (Biber)**
   Compile BibTeX references from `main.bcf` into `main.bbl`:
   ```bash
   biber main
   ```

3. **Step 3: Final XeLaTeX Linking Passes**
   Link `.bbl` numbers and cross-references into `main.pdf`:
   ```bash
   xelatex -interaction=nonstopmode main.tex
   ```

---

## Troubleshooting Known Compilation Errors

### 1. Biber Cannot Find Control File `main.bcf`
- **Cause**: Running `biber main` after deleting `.aux`/`.bcf` files before running `xelatex`.
- **Solution**: Always run `xelatex -interaction=nonstopmode main.tex` first to create `main.bcf`.

### 2. Citations Render as Raw Bold Strings (e.g. `3DGS kerbl2023gaussian`)
- **Cause**: `.bbl` is missing or out of sync with `.bcf`.
- **Solution**: Execute Biber (`biber main`) followed by `xelatex main.tex`.

### 3. `! Package xcolor Error: Undefined color 'bestblue'`
- **Cause**: Color defined inside a chapter file (e.g., `Chapter4/chapter4.tex`), but referenced in table/figure captions read early by `main.lot` / `main.toc`.
- **Solution**: Define all `\definecolor{...}` in `main.tex` **preamble** before `\begin{document}`.

### 4. `Overfull \hbox` in Tables
- **Cause**: Table width exceeds `\textwidth`.
- **Solution**: Wrap the tabular block in `\resizebox{\textwidth}{!}{ \begin{tabular}... \end{tabular} }`.

### 5. `UnicodeEncodeError: 'charmap' codec can't encode...` in Python Helper Scripts
- **Cause**: Python script printing Vietnamese file paths to Windows console.
- **Solution**: Use `str.encode("ascii", "replace").decode("ascii")` or avoid printing non-ASCII paths to stdout.
