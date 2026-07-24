# Spec-FastGS: Real-time Specular 3D Gaussian Splatting

Spec-FastGS is an advanced hybrid neural rendering framework that achieves state-of-the-art photorealism for specular and highly reflective surfaces while maintaining real-time rendering speeds (>100 FPS).

By explicitly disentangling diffuse color (using Spherical Harmonics - SH) and specular reflections (using Anisotropic Spherical Gaussians - ASG) and combining them with the optimized FastGS CUDA rasterizer, Spec-FastGS solves the view-dependent aliasing and memory bloat issues of traditional 3DGS.

## 🛠️ Hardware Requirements

- **OS**: Linux (tested on Ubuntu 20.04/22.04)
- **GPU**: NVIDIA GPU with at least 12GB VRAM (24GB recommended for large scenes)
- **CUDA**: 11.6 or 11.8

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/spec-fastgs.git
cd spec-fastgs
```

### 2. Set up the Conda environment
```bash
conda create -n spec_fastgs python=3.9 -y
conda activate spec_fastgs
```

### 3. Install PyTorch & Dependencies
Please install the PyTorch version that matches your CUDA toolkit. For CUDA 11.8:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

*(Note: `requirements.txt` should include basic packages like `tqdm`, `plyfile`, `opencv-python`, `scipy`)*

### 4. Install CUDA Submodules
Spec-FastGS relies on highly optimized CUDA kernels for rasterization and KNN search.

```bash
# Install simple-knn
cd submodules/simple-knn
pip install .
cd ../..

# Install customized diff-gaussian-rasterization (FastGS radix-sort based)
cd submodules/diff-gaussian-rasterization
pip install .
cd ../..
```

## 📖 Usage Guide

Spec-FastGS introduces a specialized offline pre-processing pipeline to extract optical priors before training begins.

### 1. Data Preparation
Ensure your dataset is processed using COLMAP, following the standard 3DGS dataset structure:
```
<dataset_path>/
├── images/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
```

### 2. Pre-processing: Reflection Score Extraction
Before training, you must extract the Reflection Score priors to guide the model's initialization and density control.

Spec-FastGS provides a convenient shell script to handle this. You can edit the parameters inside `run_extract_reflection_prior.sh` (such as `DATA_ROOT`, `SCENE`, and `REF_PRIOR_METHOD`) and then run it:

```bash
bash run_extract_reflection_prior.sh
```
*Note: This script will extract the 2D probability maps and save them in the `reflection_prior/` folder of your dataset. Note that for full 3D prior point cloud initialization, you might also need to execute `generate_prior_pcd.py`.*

### 3. Training, Rendering, and Evaluation
Spec-FastGS includes several shell scripts to automate the entire pipeline (Training -> Rendering -> Evaluation) for different datasets.

You can modify and run the base script to execute the full pipeline:
```bash
bash run_spec-fastgs_base.sh
```

To run experiments on specific datasets, you can utilize the other provided scripts:
- `bash run_anisotropic-synthesis.sh`
- `bash run_mip360.sh`
- `bash run_ref_real.sh`
- `bash run_shiny.sh`

These scripts are pre-configured with the optimal hyperparameters for extracting the Reflection Score and performing Adaptive Density Control. Ensure you edit the `DATA_ROOT` and `SCENE` variables inside the scripts before executing.

## 📚 Core Contributions
- **Reflection Score Prior**: Mathematically extracts specular locations using optical heuristic models (e.g., Tan-Ikeuchi) to guide initialization.
- **Hybrid Appearance**: Combines Spherical Harmonics (diffuse) and Anisotropic Spherical Gaussians (specular) for accurate material modeling.
- **Multi-view Consistent Density Control**: Prevents floaters by evaluating photometric errors across multiple views before cloning/splitting Gaussians.
- **Hardware-Level Optimization**: Achieves >100 FPS by leveraging FastGS parallel radix sorting and prefix-sum algorithms.

## 📄 License
This project is built upon [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [FastGS](https://github.com/aipr-iitj/FastGS). Please refer to their respective licenses for commercial usage limitations.