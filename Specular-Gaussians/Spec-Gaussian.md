## Page 1

# Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian Splatting

Ziyi Yang¹ Xinyu Gao¹ Yang-Tian Sun² Yi-Hua Huang² Xiaoyang Lyu²
Wen Zhou³ Shaohui Jiao³ Xiaojuan Qi²† Xiaogang Jin¹†
¹Zhejiang University ²The University of Hong Kong ³ ByteDance Inc.

## Abstract

The recent advancements in 3D Gaussian splatting (3D-GS) have not only facilitated real-time rendering through modern GPU rasterization pipelines but have also attained state-of-the-art rendering quality. Nevertheless, despite its exceptional rendering quality and performance on standard datasets, 3D-GS frequently encounters difficulties in accurately modeling specular and anisotropic components. This issue stems from the limited ability of spherical harmonics (SH) to represent high-frequency information. To overcome this challenge, we introduce *Spec-Gaussian*, an approach that utilizes an anisotropic spherical Gaussian (ASG) appearance field instead of SH for modeling the view-dependent appearance of each 3D Gaussian. Additionally, we have developed a coarse-to-fine training strategy to improve learning efficiency and eliminate floaters caused by overfitting in real-world scenes. Our experimental results demonstrate that our method surpasses existing approaches in terms of rendering quality. Thanks to ASG, we have significantly improved the ability of 3D-GS to model scenes with specular and anisotropic components without increasing the number of 3D Gaussians. This improvement extends the applicability of 3D GS to handle intricate scenarios with specular and anisotropic surfaces. Our codes and datasets are available at https://ingra14m.github.io/Spec-Gaussian-website.

## 1 Introduction

High-quality reconstruction and photorealistic rendering from a collection of images are crucial for a variety of applications, such as augmented reality/virtual reality (AR/VR), 3D content production, and art creation. Classic methods employ primitive representations, like meshes [36] and points [4, 62], and take advantage of the rasterization pipeline optimized for contemporary GPUs to achieve real-time rendering. In contrast, neural radiance fields (NeRF) [34, 5, 35] utilize neural implicit representation to offer a continuous scene representation and employ volumetric rendering to produce rendering results. This approach allows for enhanced preservation of scene details and more effective reconstruction of scene geometries.

Recently, 3D Gaussian Splatting (3D-GS) [20] has emerged as a leading technique, delivering state-of-the-art quality and real-time speed. This method optimizes a set of 3D Gaussians that capture the appearance and geometry of a 3D scene simultaneously, offering a continuous representation that preserves details and produces high-quality results. Besides, the CUDA-customized differentiable rasterization pipeline for 3D Gaussians enables real-time rendering even at high resolution.

Despite its exceptional performance, 3D-GS struggles to model specular components within scenes (see Fig. 1). This issue primarily stems from the limited ability of low-order spherical harmonics (SH)

† Corresponding Authors.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

---


## Page 2

&lt;img&gt;
  &lt;img&gt;Figure 1: Our method not only achieves real-time rendering but also significantly enhances the capability of 3D-GS to model scenes with specular and anisotropic components. Key to this enhanced performance is our use of ASG appearance field to model the appearance of each 3D Gaussian, which results in substantial improvements in rendering quality for both complex and general scenes.&lt;/img&gt;
&lt;/img&gt;

to capture the high-frequency information in these scenarios. Consequently, this poses a challenge for 3D-GS to model scenes with reflections and specular components, as illustrated in Fig. 1.

To address the issue, we introduce a novel approach called *Spec-Gaussian*, which combines anisotropic spherical Gaussian (ASG) [55] for modeling anisotropic and specular components, an effective training mechanism to eliminate floaters and improve learning efficiencies, and anchor-based 3D Gaussians for acceleration and storage reduction. Specifically, the method incorporates two key designs: 1) A new 3D Gaussian representation that utilizes an ASG appearance field instead of SH to model the appearance of each 3D Gaussian. ASG with a few orders can effectively model high-frequency information that low-order SH cannot. This new design enables 3D-GS to more effectively model anisotropic and specular components in static scenes. 2) A coarse-to-fine training scheme specifically tailored for 3D-GS is designed to eliminate floaters and boost learning efficiency. This strategy effectively shortens learning time by optimizing low-resolution rendering in the initial stage, preventing the need to increase the number of 3D Gaussians and regularizing the learning process to avoid the generation of unnecessary geometric structures that lead to floaters.

By combining these advances, our approach can render high-quality results for specular highlights and anisotropy as shown in Fig. 4 while preserving the efficiency of Gaussians. Furthermore, comprehensive experiments reveal that our method not only endows 3D-GS with the ability to model specular highlights but also achieves state-of-the-art results in general benchmarks.

In summary, the major contributions of our work are as follows:
*   A novel ASG appearance field to model the view-dependent appearance of each 3D Gaussian, which enables 3D-GS to effectively represent scenes with specular and anisotropic components.
*   A coarse-to-fine training scheme that effectively regularizes training to eliminate floaters and improve the learning efficiency of 3D-GS in real-world scenes.
*   An anisotropic dataset has also been made to assess the capability of our model in representing anisotropy.

## 2 Related Work

### 2.1 Implicit Neural Radiance Fields

Neural rendering has attracted significant interest in the academic community for its unparalleled ability to generate photorealistic images. Methods like NeRF [34] utilize Multi-Layer Perceptrons (MLPs) to model the geometry and radiance fields of a scene. Leveraging the volumetric rendering equation and the inherent continuity and smoothness of MLPs, NeRF achieves high-quality scene reconstruction from a set of posed images, establishing itself as the state-of-the-art (SOTA) method for novel view synthesis. Subsequent research has extended the utility of NeRF to various applications, including mesh reconstruction [48, 24, 53, 31], inverse rendering [44, 66, 28, 57], optimization of camera parameters [26, 50, 49, 38], few-shot learning [11, 56, 52], and anti-aliasing [2, 1, 3].

However, this stream of methods relies on ray casting rather than rasterization to determine the color of each pixel. Consequently, every sampling point along the ray necessitates querying the MLPs, leading to significantly slow rendering speed and prolonged training convergence. This limitation substantially impedes their application in large-scene modeling and real-time rendering.

&lt;page_number&gt;2&lt;/page_number&gt;

---


## Page 3

&lt;img&gt;Figure 2: Pipeline of Spec-Gaussian. The optimization process begins with SfM points derived from COLMAP or generated randomly, serving as the initial state for the 3D Gaussians. To address the limitations of low-order SH and pure MLP in modeling high-frequency information, we additionally employ ASG in conjunction with a feature decoupling MLP to model the view-dependent appearance of each 3D Gaussian. Then, 3D Gaussians with opacity σ > 0 are rendered through a differentiable Gaussian rasterization pipeline, effectively capturing specular highlights and anisotropy in the scene.&lt;/img&gt;

To reduce the training time of MLP-based NeRF methods and improve rendering speed, subsequent work has enhanced NeRF’s efficiency in various ways. Structure-based techniques [63, 12, 40, 15, 7] have sought to improve inference or training efficiency by caching or distilling the implicit neural representation into more efficient data structures. Hybrid methods [27, 45] increase efficiency by incorporating explicit voxel-based data structures. Factorization methods [5, 16, 8, 14] apply a low-rank tensor assumption to decompose the scene into low-dimensional planes or vectors, achieving better geometric consistency. Compared to continuous implicit representations, the convergence of individual voxels in the grid is independent, significantly reducing training time. Additionally, Instant-NGP [35] utilizes a hash grid with a corresponding CUDA implementation for faster feature querying, enabling rapid training and interactive rendering of neural radiance fields. Spec-NeRF [32] achieves high-quality specular reflection modeling by introducing Gaussian directional encoding.

Despite achieving higher quality and faster rendering, these methods have not fundamentally overcome the substantial query overhead associated with ray casting. As a result, a notable gap remains before achieving real-time rendering. In this work, we build upon the recent 3D-GS [20], a point-based rendering method that leverages rasterization. Compared to ray casting-based methods, it significantly enhances both training and rendering speed.

### 2.2 Point-based Neural Radiance Fields

Point-based representations, similar to triangle mesh-based methods, can exploit the highly efficient rasterization pipeline of modern GPUs to achieve real-time rendering. Although these methods offer breakneck rendering speeds and are well-suited for editing tasks, they often suffer from holes and outliers, leading to artifacts in the rendered images. This issue arises from the discrete nature of point clouds, which can create gaps in the primitives and, consequently, in the rendered image.

To address these discontinuity issues, differentiable point-based rendering [62, 13, 21, 22] has been extensively explored for fitting complex geometric shapes. Notably, Zhang et al. [65] employ differentiable surface splatting and utilize a radial basis function (RBF) kernel to compute the contribution of each point to each pixel.

Recently, 3D-GS [20] has employed anisotropic 3D Gaussians, initialized from Structure from Motion (SfM), to represent 3D scenes. The innovative densification mechanism and CUDA-customized differentiable Gaussian rasterization pipeline of 3D-GS have not only achieved state-of-the-art (SOTA) rendering quality but also significantly surpassed the threshold of real-time rendering. Many concurrent works have rapidly extended 3D-GS to a variety of downstream applications, including dynamic scenes [30, 58, 59, 17, 23], text-to-3D generation [25, 46, 9, 61, 10], avatars [68, 67, 18, 41, 37], scene editing [54, 6], and quality enhancement [33].

Despite achieving SOTA results on commonly used benchmark datasets, 3D-GS still struggles to model scenes with specular and reflective components, which limits its practical application in real-time rendering at the photorealistic level. In this work, by replacing spherical harmonics (SH)

&lt;page_number&gt;3&lt;/page_number&gt;

---


## Page 4

with an anisotropic spherical Gaussian (ASG) appearance field, we have enabled 3D-GS to model complex specular scenes more effectively.

# 3 Method

The overview of our method is illustrated in Fig. 2. The input to our model is a set of posed images of a static scene, together with a sparse point cloud obtained from SfM [42]. The core of our method is to use the ASG appearance field to replace SH in modeling the appearance of 3D Gaussians (Sec. 3.2). Moreover, we introduce a simple yet effective coarse-to-fine training strategy to reduce floaters in real-world scenes (Sec. 3.3). To further reduce the storage overhead and rendering speed pressure introduced by ASG, we combine a hybrid Gaussian model that employs sparse anchor Gaussians to facilitate the generation of neural Gaussians (Sec. 3.4) to model the 3D scene.

## 3.1 Preliminaries

**3D Gaussian splatting.** 3D-GS [20] is a point-based method that employs anisotropic 3D Gaussians to represent scenes. Each 3D Gaussian is defined by a center position $x$, opacity $\sigma$, and a 3D covariance matrix $\Sigma$, which is decomposed into a quaternion $r$ and scaling $s$. The view-dependent appearance of each 3D Gaussian is represented using the first three orders of spherical harmonics (SH). This method not only retains the rendering details offered by volumetric rendering but also achieves real-time rendering through a CUDA-customized differentiable Gaussian rasterization process. Following [69], the 3D Gaussians can be projected to 2D using the 2D covariance matrix $\Sigma'$, defined as:

$$
\Sigma' = J V \Sigma V^T J^T,
\quad (1)
$$

where $J$ is the Jacobian of the affine approximation of the projective transformation, and $V$ represents the view matrix, transitioning from world to camera coordinates. To facilitate learning, the 3D covariance matrix $\Sigma$ is decomposed into two learnable components: the quaternion $r$, representing rotation, and the 3D-vector $s$, representing scaling. The resulting $\Sigma$ is thus represented as the combination of a rotation matrix $R$ and scaling matrix $S$ as:

$$
\Sigma = R S S^T R^T.
\quad (2)
$$

The color of each pixel on the image plane is then rendered through a point-based volumetric rendering (alpha blending) technique:

$$
C(p) = \sum_{i \in N} T_i \alpha_i c_i, \quad \alpha_i = \sigma_i e^{-\frac{1}{2}(p - \mu_i)^T \Sigma^{-1} (p - \mu_i)},
\quad (3)
$$

where $p$ denotes the pixel coordinate, $T_i$ is the transmittance defined by $\Pi_{j=1}^{i-1}(1 - \alpha_j)$, $c_i$ signifies the color of the sorted Gaussians associated with the queried pixel, and $\mu_i$ represents the coordinates of the 3D Gaussians when projected onto the 2D image plane.

**Anisotropic spherical Gaussian.** Anisotropic spherical Gaussian (ASG) [55] has been designed in the traditional rendering pipeline to efficiently approximate lighting and shading. Different from spherical Gaussian (SG), ASG has been demonstrated to effectively represent anisotropic scenes with a small number. In addition to retaining the fundamental properties of SG, ASG also exhibits rotational invariance and can represent full-frequency signals. The ASG function is defined as:

$$
ASG(\nu \mid [x, y, z], [\lambda, \mu], \xi) = \xi \cdot S(\nu; z) \cdot e^{-\lambda(\nu \cdot x)^2 - \mu(\nu \cdot y)^2},
\quad (4)
$$

where $\nu$ is the unit direction serving as the function input; $x$, $y$, and $z$ correspond to the tangent, bi-tangent, and lobe axis, respectively, and are mutually orthogonal; $\lambda \in \mathbb{R}^1$ and $\mu \in \mathbb{R}^1$ are the sharpness parameters for the x- and y-axis, satisfying $\lambda, \mu > 0$; $\xi \in \mathbb{R}^2$ is the lobe amplitude; $S$ is the smooth term defined as $S(\nu; z) = \max(\nu \cdot z, 0)$.

Inspired by the power of ASG in modeling scenes with complex anisotropy, we propose integrating ASG into Gaussian splatting to join the forces of classic models with new rendering pipelines for higher quality. For $N$ ASGs, we predefined orthonormal axes $x$, $y$, and $z$, initializing them to be uniformly distributed across a hemisphere. During training, we allow the remaining ASG parameters, $\lambda$, $\mu$, and $\xi$, to be learnable. We use the reflect direction $\omega_r$ as the input to query ASG for modeling the view-dependent specular information. Note that we use $N = 32$ ASGs for each 3D Gaussian.

&lt;page_number&gt;4&lt;/page_number&gt;

---


## Page 5

**Anchor-based Gaussian splatting.** Anchor-based Gaussian splatting was first proposed by Scaffold-GS [29]. Unlike the attributes carried by each entity in 3D-GS, each anchor Gaussian carries a position coordinate $\mathbf{P}_v \in \mathbb{R}^3$, a local feature $\mathbf{f}_v \in \mathbb{R}^{32}$, a displacement factor $\eta_v \in \mathbb{R}^3$, and $k$ learnable offsets $\mathbf{O}_v \in \mathbb{R}^{k \times 3}$. They use the COLMAP [42] point cloud to initialize each anchor 3D Gaussian, serving as the voxel centers to guide the generation of neural Gaussians. The position $\mathbf{P}_v$ of the anchor Gaussian is initialized as:

$$
\mathbf{P}_v = \left\{ \left\lfloor \frac{\mathbf{P}}{\epsilon} + 0.5 \right\rfloor \right\} \cdot \epsilon,
\quad (5)
$$

where $\mathbf{P}$ is the point cloud position, $\epsilon$ is the voxel size, and $\{\cdot\}$ denotes removing duplicated anchors.

Then anchor Gaussians can guide the generation of neural Gaussians, which have the same attributes as vanilla 3D-GS. For each visible anchor Gaussian within the viewing frustum, we spawn $k$ neural Gaussians and predict their attributes. The positions $\mathbf{x}$ of neural Gaussians are calculated as:

$$
\{\mathbf{x}_0, \dots, \mathbf{x}_{k-1}\} = \mathbf{P}_v + \{\mathbf{O}_0, \dots, \mathbf{O}_{k-1}\} \cdot \eta_v,
\quad (6)
$$

where $\mathbf{P}_v$ represents the position of the anchor Gaussian corresponding to $k$ neural Gaussians. The opacity $\sigma$ is calculated through a tiny MLP:

$$
\{\sigma_0, \dots, \sigma_{k-1}\} = \mathcal{F}_\sigma(\mathbf{f}_v, \delta_{cv}, \mathbf{d}_{cv}),
\quad (7)
$$

where $\delta_{cv}$ denotes the distance between the anchor Gaussian and the camera, and $\mathbf{d}_{cv}$ is the unit direction pointing from the camera to the anchor Gaussian. The rotation $r$ and scaling $s$ of each neural Gaussian are derived similarly using the corresponding tiny MLP $\mathcal{F}_r$ and $\mathcal{F}_s$.

### 3.2 Anisotropic View-Dependent Appearance

**ASG appearance field for 3D Gaussians.** Although SH has enabled view-dependent scene modeling, the low frequency of low-order SH makes it challenging to model scenes with complex optical phenomena such as specular highlights and anisotropic effects. Therefore, instead of using SH, we propose using an ASG appearance field based on Eq. (4) to model the appearance of each 3D Gaussian. However, the introduction of ASG increases the feature dimensions of each 3D Gaussian, raising the model's storage overhead. To address this, we employ a compact learnable MLP $\Theta$ to predict the parameters for $N$ ASGs, with each Gaussian carrying only additional local features $\mathbf{f} \in \mathbb{R}^{24}$ as the input to the MLP:

$$
\Theta(\mathbf{f}) \rightarrow \{\lambda, \mu, \xi\}_N.
\quad (8)
$$

To better differentiate between high and low-frequency information and further assist ASG in fitting high-frequency specular details, we decompose color $c$ into diffuse and specular components:

$$
c = c_d + c_s,
\quad (9)
$$

where $c_d$ represents the diffuse color, modeled using the first three orders of SH, and $c_s$ is the specular color calculated through ASG. We refer to this comprehensive approach to appearance modeling as the ASG appearance field.

Although ASG theoretically enhance the ability of SH to model anisotropy, directly using ASG to represent the specular color of each 3D Gaussian still falls short in accurately modeling anisotropic and specular components, as demonstrated in Fig. 6. Inspired by [14], we do not use ASG directly to represent color but instead employ ASG to model the latent feature of each 3D Gaussian. This latent feature, containing anisotropic information, is then fed into a tiny feature decoding MLP $\Psi$ to determine the final specular color:

$$
\Psi(\kappa, \gamma(\mathbf{d}), \langle n, -\mathbf{d} \rangle) \rightarrow c_s,
$$
$$
\kappa = \bigoplus_{i=1}^N ASG(\omega_r \mid [\mathbf{x}, \mathbf{y}, \mathbf{z}], [\lambda_i, \mu_i], \xi_i)
\quad (10)
$$

where $\kappa$ is the latent feature derived from ASG, $\bigoplus$ denotes the concatenation operation, $\gamma$ represents the positional encoding, $\mathbf{d}$ is the unit view direction pointing from the camera to each 3D Gaussian, $n$ is the normal of each 3D Gaussian, and $\omega_r$ is the unit reflect direction. This strategy significantly enhances the ability of 3D-GS to model scenes with complex optical phenomena, whereas neither pure ASG nor pure MLP can achieve anisotropic appearance modeling as effectively as our approach.

&lt;page_number&gt;5&lt;/page_number&gt;

---


## Page 6

Table 1: Quantitative Comparison on anisotropic synthetic dataset.

<table>
  <thead>
    <tr>
      <th>Dataset Method</th>
      <th>PSNR ↑</th>
      <th>SSIM ↑</th>
      <th>LPIPS ↓</th>
      <th>FPS</th>
      <th>Mem</th>
      <th>Num.(k)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3D-GS</td>
      <td>33.82</td>
      <td>0.966</td>
      <td>0.062</td>
      <td>325</td>
      <td>47MB</td>
      <td>201</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>35.34</td>
      <td>0.972</td>
      <td>0.052</td>
      <td>234</td>
      <td>27MB</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>36.76</td>
      <td>0.976</td>
      <td>0.046</td>
      <td>180</td>
      <td>25MB</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>37.42</td>
      <td>0.979</td>
      <td>0.044</td>
      <td>159</td>
      <td>45MB</td>
      <td>146</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>37.70</td>
      <td>0.980</td>
      <td>0.042</td>
      <td>145</td>
      <td>57MB</td>
      <td>183</td>
    </tr>
  </tbody>
</table>

&lt;img&gt;Figure 3: Using a coarse-to-fine strategy, our approach can eliminate the floaters without increasing the number of GS.&lt;/img&gt;

**Normal estimation.** Following [19, 43], we use the shortest axis of each Gaussian as its normal. This approach is based on the observation that 3D Gaussians tend to flatten gradually during the optimization process, allowing the shortest axis to serve as a reasonable approximation for the normal. The reflect direction $\omega_r$ can then be derived using the view direction and the local normal vector $n$ as:

$$
\omega_r = 2(\omega_o \cdot n) \cdot n - \omega_o,
$$
(11)

where $\omega_o = -d$ is a unit view direction pointing from each 3D Gaussian in world space to the camera. We use the reflect direction $\omega_r$ to query ASG, enabling better interpolation of latent features containing anisotropic information. Experimental results show that although this unsupervised normal estimation cannot generate physically accurate normals aligned with the real world, it is sufficient to produce relatively accurate reflect direction to assist ASG in fitting high-frequency information.

### 3.3 Coarse-to-fine Training

We observed that in many real-world scenarios, 3D-GS tends to overfit the training data, leading to the emergence of numerous floaters when rendering images from novel viewpoints. One important reason is that the COLMAP point cloud is too sparse. Poor initialization makes it difficult for 3D-GS to compensate for overly sparse areas through densification during optimization, leading to floaters in the rendering images. Moreover, 3D-GS accumulates gradients from each pixel to the GS: $\frac{dL}{dx} = \sum \frac{dL}{dp_i} \frac{dp_i}{dx}$, and the densification occurs when the accumulated amount exceeds a threshold $\tau_g = 0.0002$. However, having positive and negative gradients can cause GSs that should be densified to be ignored due to the large negative gradient.

Thus, to mitigate the occurrence of floaters in real-world scenes, we propose a coarse-to-fine training mechanism. We first impose an L1 constraint on the gradients from pixels to GS: $\frac{dL}{dx} = \sum ||\frac{dL}{dp_i} \frac{dp_i}{dx}||_1$, accumulating the numerical contribution from pixels to GS rather than gradients. This idea is similar to the concurrent works [60, 64]. Next, to avoid overfitting caused by excessive growth of 3D-GS during the early stages of optimization, we decide to train 3D-GS progressively from low to high resolution in real-world scenes:

$$
r(i) = \min([r_s + (r_e - r_s) \cdot i / \tau], r_e),
$$
(12)

where $r(i)$ is the image resolution at the $i$-th training iteration, $r_s$ is the starting image resolution, $r_e$ is the ending image resolution (the full resolution we aim to render), and $\tau$ is the threshold iteration, empirically set to 5k.

This training method allows 3D-GS to densify correctly and prevents excessive growth of 3D-GS in the early stages. Additionally, due to the lower resolution training in the initial phase, this mechanism reduces training time by approximately 10%. In our experiments, we offer a performance version with $\tau_g = 0.0005$ and light version with $\tau_g = 0.0006$.

### 3.4 Adaption for Anchor-Based Gaussian Splatting

While the ASG appearance field significantly improves the ability of 3D-GS to model specular and anisotropic features, it introduces additional computational overhead due to the additional local features $f$ associated with each Gaussian. Inspired by [29], we employ anchor-based Gaussian splatting to reduce storage overhead and accelerate the rendering.

Since the anisotropy modeled by ASG is continuous in space, it can be compressed into a lower-dimensional space. Thanks to the guidance of the anchor Gaussian, the anchor feature $f_v$ can be

&lt;page_number&gt;6&lt;/page_number&gt;

---


## Page 7

&lt;img&gt;
Figure 4: Visualization on NeRF dataset. Our method has achieved specular highlights modeling, which other 3D-GS-based methods fail to accomplish, while maintaining fast rendering speed.
&lt;/img&gt;

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset<br>Method | Metrics</th>
      <th colspan="4">Mip-NeRF 360</th>
      <th colspan="3">Mip-NeRF 360 Outdoor</th>
      <th colspan="3">Mip-NeRF 360 Indoor</th>
    </tr>
    <tr>
      <th>PSNR ↑</th>
      <th>SSIM ↑</th>
      <th>LPIPS ↓</th>
      <th>FPS</th>
      <th>Mem</th>
      <th>PSNR ↑</th>
      <th>SSIM ↑</th>
      <th>LPIPS ↓</th>
      <th>PSNR ↑</th>
      <th>SSIM ↑</th>
      <th>LPIPS ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Plenoxels</td>
      <td>23.08</td>
      <td>0.626</td>
      <td>0.463</td>
      <td>6.79</td>
      <td>2.1GB</td>
      <td>21.68</td>
      <td>0.513</td>
      <td>0.491</td>
      <td>24.83</td>
      <td>0.766</td>
      <td>0.426</td>
    </tr>
    <tr>
      <td>iNGP</td>
      <td>25.59</td>
      <td>0.699</td>
      <td>0.331</td>
      <td>9.43</td>
      <td>48MB</td>
      <td>22.75</td>
      <td>0.567</td>
      <td>0.403</td>
      <td>29.14</td>
      <td>0.863</td>
      <td>0.242</td>
    </tr>
    <tr>
      <td>Mip-NeRF360</td>
      <td>27.69</td>
      <td>0.792</td>
      <td>0.237</td>
      <td>0.06</td>
      <td>8.6MB</td>
      <td>24.47</td>
      <td>0.691</td>
      <td>0.283</td>
      <td>31.72</td>
      <td>0.917</td>
      <td>0.180</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>27.79</td>
      <td>0.826</td>
      <td>0.202</td>
      <td>115</td>
      <td>748MB</td>
      <td>25.02</td>
      <td>0.742</td>
      <td>0.232</td>
      <td>31.25</td>
      <td>0.931</td>
      <td>0.164</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>27.98</td>
      <td>0.824</td>
      <td>0.207</td>
      <td>96</td>
      <td>203MB</td>
      <td>25.07</td>
      <td>0.736</td>
      <td>0.243</td>
      <td>31.61</td>
      <td>0.933</td>
      <td>0.162</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>28.14</td>
      <td>0.824</td>
      <td>0.196</td>
      <td>70</td>
      <td>260MB</td>
      <td>24.98</td>
      <td>0.735</td>
      <td>0.223</td>
      <td>32.09</td>
      <td>0.935</td>
      <td>0.161</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>28.07</td>
      <td>0.834</td>
      <td>0.183</td>
      <td>44</td>
      <td>684MB</td>
      <td>25.09</td>
      <td>0.752</td>
      <td>0.203</td>
      <td>31.80</td>
      <td>0.936</td>
      <td>0.158</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>28.18</td>
      <td>0.835</td>
      <td>0.176</td>
      <td>33</td>
      <td>847MB</td>
      <td>25.11</td>
      <td>0.754</td>
      <td>0.195</td>
      <td>32.01</td>
      <td>0.937</td>
      <td>0.153</td>
    </tr>
  </tbody>
</table>

Table 2: **Quantitative comparison of on real-world datasets.** We report PSNR, SSIM, LPIPS (VGG) and color each cell as **best**, **second best** and **third best**. Our method has achieved the best rendering quality, while striking a good balance between FPS and the storage memory.

used directly to compress $N$ ASGs, further reducing storage pressure. To make the ASG of neural Gaussians position-aware, we introduce the unit view direction to decompress ASG parameters. Consequently, the ASG parameters prediction in Eq. (8) is revised as follows:

$$
\Theta(\mathbf{f}_v, \mathbf{d}_{cn}) \rightarrow \{\lambda, \mu, \xi\}_N,
\quad (13)
$$

where $\mathbf{d}_{cn}$ denotes the unit view direction from the camera to each neural Gaussian. Additionally, we set the diffuse part of the neural Gaussian $c_d = \phi(\mathbf{f}_v)$, directly predicted through an MLP $\phi$, to ensure the smoothness of the diffuse component and reduce the difficulty of convergence.

### 3.5 Losses

We optimize the learnable parameters and MLPs using the same loss function as 3D-GS [20]. The total supervision is given by:

$$
\mathcal{L} = (1 - \lambda_{D-SSIM})\mathcal{L}_1 + \lambda_{D-SSIM}\mathcal{L}_{D-SSIM},
\quad (14)
$$

where the $\lambda_{D-SSIM} = 0.2$ is consistently used in our experiments.

### 4 Experiments

In this section, we present both quantitative and qualitative results of our method. To evaluate its effectiveness, we compared it to several state-of-the-art methods across various datasets. We color each cell as **best**, **second best** and **third best**. Our method includes three versions, each based on different foundational methods with distinct hyperparameter settings. The performance version (Ours) is based on 3D-GS [20] with $\tau_g = 0.0005$; the light version (Ours-light), also based on 3D-GS, has $\tau_g = 0.0006$; and the mini version (Ours-w/ anchor) is based on Scaffold-GS [29], with $\tau_g = 0.0006$. Our method demonstrates superior performance in modeling complex specular and anisotropic features, as evidenced by comparisons on the NeRF, NSVF, and our "Anisotropic Synthetic" datasets. Additionally, we showcase its versatility by comparing its performance in diffuse scenarios, further proving the robustness of our approach.

&lt;page_number&gt;7&lt;/page_number&gt;

---


## Page 8

Table 3: Results on NeRF synthetic dataset.
<table>
  <thead>
    <tr>
      <th>Dataset Method | Metrics</th>
      <th>NeRF Synthetic PSNR ↑</th>
      <th>SSIM ↑</th>
      <th>LPIPS ↓</th>
      <th>FPS</th>
      <th>Mem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>iNGP-Base</td>
      <td>33.18</td>
      <td>0.963</td>
      <td>0.045</td>
      <td>~10</td>
      <td>13MB</td>
    </tr>
    <tr>
      <td>Mip-NeRF</td>
      <td>33.09</td>
      <td>0.961</td>
      <td>0.043</td>
      <td>&lt;1</td>
      <td>10MB</td>
    </tr>
    <tr>
      <td>Tri-MipRF</td>
      <td>33.65</td>
      <td>0.963</td>
      <td>0.042</td>
      <td>~5</td>
      <td>60MB</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>33.32</td>
      <td>0.969</td>
      <td>0.031</td>
      <td>315</td>
      <td>69MB</td>
    </tr>
    <tr>
      <td>GS-Shader</td>
      <td>33.38</td>
      <td>0.968</td>
      <td>0.030</td>
      <td>97</td>
      <td>29MB</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>33.68</td>
      <td>0.967</td>
      <td>0.034</td>
      <td>240</td>
      <td>19MB</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>33.96</td>
      <td>0.969</td>
      <td>0.032</td>
      <td>172</td>
      <td>19MB</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>34.08</td>
      <td>0.970</td>
      <td>0.029</td>
      <td>148</td>
      <td>58MB</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>34.19</td>
      <td>0.971</td>
      <td>0.028</td>
      <td>121</td>
      <td>72MB</td>
    </tr>
  </tbody>
</table>

Table 4: Results on NSVF synthetic dataset.
<table>
  <thead>
    <tr>
      <th>Dataset Method | Metrics</th>
      <th>NSVF Synthetic PSNR ↑</th>
      <th>SSIM ↑</th>
      <th>LPIPS ↓</th>
      <th>FPS</th>
      <th>Mem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TensoRF</td>
      <td>36.52</td>
      <td>0.982</td>
      <td>0.026</td>
      <td>1.5</td>
      <td>65MB</td>
    </tr>
    <tr>
      <td>Tri-MipRF</td>
      <td>34.58</td>
      <td>0.973</td>
      <td>0.030</td>
      <td>~5</td>
      <td>60MB</td>
    </tr>
    <tr>
      <td>NeuRBF</td>
      <td>37.80</td>
      <td>0.986</td>
      <td>0.019</td>
      <td>~1</td>
      <td>580MB</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>37.07</td>
      <td>0.987</td>
      <td>0.015</td>
      <td>306</td>
      <td>71MB</td>
    </tr>
    <tr>
      <td>GS-Shader</td>
      <td>33.85</td>
      <td>0.981</td>
      <td>0.020</td>
      <td>68</td>
      <td>33MB</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>36.43</td>
      <td>0.984</td>
      <td>0.017</td>
      <td>218</td>
      <td>17MB</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>37.71</td>
      <td>0.987</td>
      <td>0.015</td>
      <td>152</td>
      <td>16MB</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>38.28</td>
      <td>0.988</td>
      <td>0.013</td>
      <td>124</td>
      <td>70MB</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>38.40</td>
      <td>0.988</td>
      <td>0.012</td>
      <td>108</td>
      <td>89MB</td>
    </tr>
  </tbody>
</table>

&lt;img&gt;Figure 5: Visualization on Mip-NeRF 360 indoor scenes. Our method achieves superior recovery of specular effects compared to SOTA methods.&lt;/img&gt;

4.1 Implementation Details
We implemented our framework using PyTorch [39] and modified the differentiable Gaussian rasterization to include depth visualization. For the ASG appearance field, the decoupling MLP Ψ consists of 3 layers, each with 64 hidden units, and the positional encoding for the view direction is of order 2. Regarding coarse-to-fine training, which is applied only to real-world scenes to remove floaters, we start with a resolution $r_s$ that is 4x downsampled. To further accelerate rendering, we prefilter and allow only those Gaussians with opacity $\sigma_n > 0$ to pass through the ASG appearance field and Gaussian rasterization pipelines. All experiments were conducted on an NVIDIA RTX 3090.

4.2 Results and Comparisons
Synthetic bounded scenes. We used the NeRF, NSVF, and our "Anisotropic Synthetic" datasets as the experimental datasets for synthetic scenes. Our comparisons were made with the most relevant state-of-the-art methods, including 3D-GS [20], Scaffold-GS [29], GaussianShader [19], and several NeRF-based methods such as NSVF [27], TensoRF [5], NeuRBF [8], and Tri-MipRF [16].

As shown in Fig. 4 (with PSNR and LPIPS), and Tabs. 3-4, our method achieved the highest performance with fewer Gaussians compared to vanilla 3D-GS. It also improved upon the issues that 3D-GS faced in modeling high-frequency specular highlights and complex anisotropy as shown in Tab. 1 with fewer Gaussians and better metrics. See more in the supplementary materials.

Real-world unbounded scenes. To verify the versatility of our method in real-world scenarios, we used the Mip360 [2] dataset, which contains indoor scenes with specular highlights. As shown in Tab. 2, our method surpasses state-of-the-art methods on Mip-NeRF 360. Furthermore, our method effectively balances FPS, storage, and rendering quality. It enhances rendering quality without increasing storage or significantly reducing FPS. As illustrated in Fig. 5 and Fig. 7, our method has also significantly improved the visual effect. It removes a large number of floaters in outdoor scenes and successfully models the high-frequency specular highlights in indoor scenes. This demonstrates that our approach is not only adept at modeling complex specular scenes but also effectively improves rendering quality in general scenarios.

4.3 Ablation Study
ASG feature decoupling MLP. We conducted an ablation study to evaluate the key components of the ASG appearance field, which include ASG features, decoupling MLP, and the separation of

&lt;page_number&gt;8&lt;/page_number&gt;

---


## Page 9

&lt;img&gt;
    &lt;img&gt;Figure 6: Ablation on ASG appearance field. We show that directly using ASG to model color leads to the failure in modeling anisotropy and specular highlights. By decoupling the ASG features through MLP, we can realistically model complex optical phenomena.&lt;/img&gt;
    &lt;img&gt;Figure 7: Ablation on coarse-to-fine training. Experimental results demonstrate that our simple yet effective training mechanism can effectively remove floaters without increasing the number of 3D Gaussians, thereby alleviating the overfitting problem prevalent in 3D-GS-based methods.&lt;/img&gt;
&lt;/img&gt;

diffuse and specular colors. As demonstrated in Fig. 6 (with PSNR and LPIPS), directly using ASG to output color results in the inability to model specular and anisotropic components. In contrast to directly using an MLP for color modeling, as in Scaffold-GS [29], separately modeling diffuse and specular color can enhance the fitting ability for high-frequency information. ASG can encode higher-frequency anisotropic features. With the help of ASG’s ability to encode high-frequency anisotropic features, the decoupling MLP can fit complex optical phenomena, leading to more accurate rendering results. We also demonstrated that higher-order SH (6-order) and more MLP layers (4-layers) do not help 3D-GS and Scaffold-GS achieve satisfactory results, highlighting the importance of ASG.

**Coarse-to-fine training.** We conducted an ablation study to assess the impact of coarse-to-fine (c2f) training. As illustrated in Fig. 7 (with LPIPS and number of Gaussian), both 3D-GS and Scaffold-GS exhibit a large number of floaters in the novel view synthesis. Coarse-to-fine training effectively reduces the number of floaters, alleviating the overfitting issue commonly encountered by 3D-GS in real-world scenarios. Applying an L1 constraint to the gradients used for 3D-GS densification further reduced the number of floaters and Gaussians. See more in the supplementary materials.

## 5 Conclusion

In this work, we introduce *Spec-Gaussian*, a novel approach to 3D Gaussian splitting that features an anisotropic view-dependent appearance. Leveraging the powerful capabilities of ASG, our method effectively overcomes the challenges encountered by vanilla 3D-GS in rendering scenes with specular highlights and anisotropy. Additionally, we innovatively implement a coarse-to-fine training mechanism to eliminate floaters in real-world scenes. Both quantitative and qualitative experiments demonstrate that our method not only equips 3D-GS with the ability to model specular highlights and anisotropy but also enhances the overall rendering quality of 3D-GS in general scenes.

**Limitations.** Although our method enables 3D-GS to model complex specular and anisotropic features, it still faces challenges in handling reflections. Specular and anisotropic effects are primarily influenced by material properties, whereas reflections are closely related to the environment and geometry. Due to the lack of explicit geometry in 3D-GS, we cannot differentiate between reflections and materials using constraints like normals, as employed in Ref-NeRF [47] and NeRO [28]. We plan to explore solutions for modeling reflections with 3D-GS in future work.

&lt;page_number&gt;9&lt;/page_number&gt;

---


## Page 10

# 6 Acknowlegements

We thank Chao Wan from Cornell University for the help during rebuttal period. This work was supported by the National Natural Science Foundation of China (Grant No. 62036010). Ziyi Yang was also supported by ByteDance MMLab.

# References

[1] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. *ICCV*, 2021.
[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5470–5479, 2022.
[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. *ICCV*, 2023.
[4] Mario Botsch, Alexander Hornung, Matthias Zwicker, and Leif Kobbelt. High-quality surface splatting on today's gpus. In *Proceedings Eurographics/IEEE VGTC Symposium Point-Based Graphics, 2005.*, pages 17–141. IEEE, 2005.
[5] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In *European Conference on Computer Vision*, pages 333–350. Springer, 2022.
[6] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting, 2023.
[7] Zhiqin Chen, Thomas Funkhouser, Peter Hedman, and Andrea Tagliasacchi. Mobilenerf: Exploiting the polygon rasterization pipeline for efficient neural field rendering on mobile architectures. *arXiv preprint arXiv:2208.00277*, 2022.
[8] Zhang Chen, Zhong Li, Liangchen Song, Lele Chen, Jingyi Yu, Junsong Yuan, and Yi Xu. Neurbf: A neural fields representation with adaptive radial basis functions. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4182–4194, 2023.
[9] Zilong Chen, Feng Wang, and Huaping Liu. Text-to-3d using gaussian splatting. *arXiv preprint arXiv:2309.16585*, 2023.
[10] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. *arXiv preprint arXiv:2311.13384*, 2023.
[11] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised NeRF: Fewer views and faster training for free. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2022.
[12] Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 14346–14355, 2021.
[13] Markus Gross and Hanspeter Pfister. *Point-based graphics*. Elsevier, 2011.
[14] Kang Han and Wei Xiang. Multiscale tensor decomposition and rendering equation encoding for view synthesis. In *The IEEE/CVF Computer Vision and Pattern Recognition Conference*, pages 4232–4241, 2023.
[15] Peter Hedman, Pratul P Srinivasan, Ben Mildenhall, Jonathan T Barron, and Paul Debevec. Baking neural radiance fields for real-time view synthesis. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5875–5884, 2021.
[16] Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, and Yuewen Ma. Tri-miprf: Tri-mip representation for efficient anti-aliasing neural radiance fields. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 19774–19783, 2023.
[17] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. pages 1–11, 2023.
[18] Yuheng Jiang, Zhehao Shen, Penghao Wang, Zhuo Su, Yu Hong, Yingliang Zhang, Jingyi Yu, and Lan Xu. Hifi4g: High-fidelity human performance rendering via compact gaussian splatting. *arXiv preprint arXiv:2312.03461*, 2023.
[19] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. *arXiv preprint arXiv:2311.17977*, 2023.
[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42(4), July 2023.
[21] Leonid Keselman and Martial Hebert. Approximate differentiable rendering with algebraic surfaces. In *European Conference on Computer Vision*, pages 596–614. Springer, 2022.
[22] Leonid Keselman and Martial Hebert. Flexible techniques for differentiable rendering with 3d gaussians. *arXiv preprint arXiv:2308.14737*, 2023.
[23] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. *arXiv preprint arXiv:2312.16812*, 2023.
[24] Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. Neuralangelo: High-fidelity neural surface reconstruction. In *IEEE Conference on Computer*

&lt;page_number&gt;10&lt;/page_number&gt;

---


## Page 11

Vision and Pattern Recognition (CVPR), 2023.
[25] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching, 2023.
[26] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In IEEE International Conference on Computer Vision (ICCV), 2021.
[27] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. Advances in Neural Information Processing Systems, 33:15651–15663, 2020.
[28] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng Wang, Lingjie Liu, Taku Komura, and Wenping Wang. Nero: Neural geometry and brdf reconstruction of reflective objects from multiview images. In SIGGRAPH, 2023.
[29] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. arXiv preprint arXiv:2312.00109, 2023.
[30] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 3DV, 2024.
[31] Xiaoyang Lyu, Yang-Tian Sun, Yi-Hua Huang, Xiuzhe Wu, Ziyi Yang, Yilun Chen, Jiangmiao Pang, and Xiaojuan Qi. 3dgsr: Implicit surface reconstruction with 3d gaussian splatting. arXiv preprint arXiv:2404.00409, 2024.
[32] Li Ma, Vasu Agrawal, Haithem Turki, Changil Kim, Chen Gao, Pedro Sander, Michael Zollhöfer, and Christian Richardt. Specnerf: Gaussian directional encoding for specular reflections. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21188–21198, 2024.
[33] Dawid Malarz, Weronika Smolak, Jacek Tabor, Sławomir Tadeja, and Przemysław Spurek. Gaussian splatting with nerf-based color and opacity.
[34] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.
[35] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1–102:15, July 2022.
[36] Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao, Wenzheng Chen, Alex Evans, Thomas Müller, and Sanja Fidler. Extracting triangular 3d models, materials, and lighting from images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8280–8290, 2022.
[37] Haokai Pang, Heming Zhu, Adam Kortylewski, Christian Theobalt, and Marc Habermann. Ash: Animatable gaussian splats for efficient and photoreal human rendering. 2023.
[38] Keunhong Park, Philipp Henzler, Ben Mildenhall, Jonathan T Barron, and Ricardo Martin-Brualla. Camp: Camera preconditioning for neural radiance fields. ACM Transactions on Graphics (TOG), 42(6):1–11, 2023.
[39] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32, 2019.
[40] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14335–14345, 2021.
[41] Shunsuke Saito, Gabriel Schwartz, Tomas Simon, Junxuan Li, and Giljoo Nam. Relightable gaussian codec avatars. 2023.
[42] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern recognition, pages 4104–4113, 2016.
[43] Yahao Shi, Yanmin Wu, Chenming Wu, Xing Liu, Chen Zhao, Haocheng Feng, Jingtuo Liu, Liangjun Zhang, Jian Zhang, Bin Zhou, et al. Gir: 3d gaussian inverse rendering for relightable scene factorization. arXiv preprint arXiv:2312.05133, 2023.
[44] Pratul P. Srinivasan, Boyang Deng, Xiuming Zhang, Matthew Tancik, Ben Mildenhall, and Jonathan T. Barron. Nerv: Neural reflectance and visibility fields for relighting and view synthesis. In CVPR, 2021.
[45] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5459–5469, 2022.
[46] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023.
[47] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T. Barron, and Pratul P. Srinivasan. Ref-NeRF: Structured view-dependent appearance for neural radiance fields. CVPR, 2022.
[48] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. NeurIPS, 2021.
[49] Peng Wang, Lingzhe Zhao, Ruijie Ma, and Peidong Liu. BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4170–4179, June 2023.
[50] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. NeRF—: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064, 2021.
[51] Suttisak Wizadwongsa, Pakkapon Phongthawee, Jiraphon Yenphraphai, and Supasorn Suwajanakorn. Nex: Real-time view synthesis with neural basis expansion. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

&lt;page_number&gt;11&lt;/page_number&gt;

---


## Page 12

[52] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P. Srinivasan, Dor Verbin, Jonathan T. Barron, Ben Poole, and Aleksander Holynski. Reconfusion: 3d reconstruction with diffusion priors. *arXiv*, 2023.
[53] Tong Wu, Jiaqi Wang, Xingang Pan, Xudong Xu, Christian Theobalt, Ziwei Liu, and Dahua Lin. Voxurf: Voxel-based efficient and accurate neural surface reconstruction. In *International Conference on Learning Representations (ICLR)*, 2023.
[54] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physics-integrated 3d gaussians for generative dynamics. *arXiv preprint arXiv:2311.12198*, 2023.
[55] Kun Xu, Wei-Lun Sun, Zhao Dong, Dan-Yong Zhao, Run-Dong Wu, and Shi-Min Hu. Anisotropic spherical gaussians. *ACM Transactions on Graphics*, 32(6):209:1–209:11, 2013.
[56] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.
[57] Ziyi Yang, Yanzhen Chen, Xinyu Gao, Yazhen Yuan, Yu Wu, Xiaowei Zhou, and Xiaogang Jin. Sire-ir: Inverse rendering for brdf reconstruction with shadow and illumination removal in high-illuminance scenes. *arXiv preprint arXiv:2310.13030*, 2023.
[58] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 20331–20341, 2024.
[59] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. *arXiv preprint arXiv 2310.10642*, 2023.
[60] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong Dou. Absgs: Recovering fine details in 3d gaussian splatting. In *ACM Multimedia 2024*, 2024.
[61] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. *arXiv preprint arXiv:2310.08529*, 2023.
[62] Wang Yifan, Felice Serena, Shihao Wu, Cengiz Öztireli, and Olga Sorkine-Hornung. Differentiable surface splatting for point-based geometry processing. *ACM Transactions on Graphics (TOG)*, 38(6):1–14, 2019.
[63] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5752–5761, 2021.
[64] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient and compact surface reconstruction in unbounded scenes. *arXiv preprint arXiv:2404.10772*, 2024.
[65] Qiang Zhang, Seung-Hwan Baek, Szymon Rusinkiewicz, and Felix Heide. Differentiable point-based radiance fields for efficient view synthesis. *arXiv preprint arXiv:2205.14330*, 2022.
[66] Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul Debevec, William T Freeman, and Jonathan T Barron. Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. *ACM Transactions on Graphics (ToG)*, 40(6):1–18, 2021.
[67] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu. Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. *arXiv*, 2023.
[68] Wojciech Zielonka, Timur Bagautdinov, Shunsuke Saito, Michael Zollhöfer, Justus Thies, and Javier Romero. Drivable 3d gaussian avatars. 2023.
[69] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Ewa volume splatting. In *Proceedings Visualization, 2001. VIS'01.*, pages 29–538. IEEE, 2001.

&lt;page_number&gt;12&lt;/page_number&gt;

---


## Page 13

A Appendix / supplemental material

This supplementary material provides more results that accompany the paper.
* Section B provides more ablations.
* Section C provides additional results, including more visualizations and quantitative results on complete datasets.

B More Ablations

In this section, we present the complete quantitative ablations on the key components of our method. We first evaluate the role of each component of the ASG appearance field in NeRF synthetic scenes as shown in Tab. 5. The introduction of ASG improves the ability to model specular highlights and reduces the number of 3D Gaussians. The inclusion of normals did not significantly increase computational overhead, but it did enhance rendering metrics and visual quality. More importantly, we achieve better rendering quality with fewer Gaussians than vanilla 3D-GS, a characteristic that can be further explored in the future.

Next, we evaluated our full method on the Mip360 dataset in Tab. 6. It is important to note that the Mip360 dataset is divided into indoor and outdoor scenes. Indoor scenes have more specular highlights, while outdoor scenes contain a large number of floaters. The coarse-to-fine approach itself improves the quality of 3D-GS in real-world scenes, mainly by eliminating a significant amount of floaters in outdoor settings. Although the introduction of the ASG appearance field significantly increases rendering overhead, it did greatly enhance the modeling of specular highlights in indoor scenes. Under the constraints of the coarse-to-fine mechanism, our complete method combines the advantages of both, achieving the best rendering quality. To further improve rendering speed, we also implement a light version and a mini version based on Scaffold-GS. These versions offer a trade-off between rendering quality and speed and can be used as needed. The quality of the Mip360 scenes demonstrates that our method is not only capable of handling scenes with specular highlights but is also robust in real-world diffuse scenarios.

C More Comparisons

In this section, we present the complete quantitative results of our experiments. We report PSNR, SSIM, LPIPS (VGG), and color each cell as best, second best and third best.

C.1 NeRF Synthetic Scenes

As shown in Tabs. 7-9, our method demonstrates the best rendering quality metrics in almost every scene. It's important to note that the experimental setup for Tri-MipRF [16] differs from other methods. It uses both the training and validation sets as training data, expanding the scale of the model's data. When its training data is limited to the training set, its metrics suffer a noticeable drop. Nevertheless, to ensure that the experimental results fully reflect the highest performance of each method, and to prevent significant drops in metrics due to differences in experimental environments, we still present the metrics from the Tri-MipRF official paper. Our method achieved more prominent metrics in scenes with notable specular reflection and anisotropy, such as Drums, Lego, and Ship. This demonstrates that our method not only improves the overall rendering quality but also has a more significant advantage in complex specular scenarios.

C.2 NSVF Synthetic Scenes

The NSVF [27] dataset, in comparison to NeRF, features more noticeable metallic specular reflection, as presented in the Wineholder, Steamtrain, and Spaceship scenes. It is important to note that Tri-MipRF fails to converge in the Steam scene with the official code, so we did not report metrics for that scenario. As shown in Tabs. 10-12, we present the per-scene experimental results of PSNR, SSIM, and LPIPS in the supplementary material. The experimental results indicate that compared to other methods based on 3D-GS [20], our method has significant advantages in metallic highlights

&lt;page_number&gt;13&lt;/page_number&gt;

---


## Page 14

Table 5: Ablation on ASG appearance field.
<table>
  <thead>
    <tr>
      <th>Dataset Method</th>
      <th>PSNR ↑</th>
      <th>SSIM ↑</th>
      <th>LPIPS ↓</th>
      <th>FPS ↑</th>
      <th>Num.(k) ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3D-GS</td>
      <td>33.32</td>
      <td>0.969</td>
      <td>0.031</td>
      <td>315</td>
      <td>295</td>
    </tr>
    <tr>
      <td>w/o ASG</td>
      <td>34.03</td>
      <td>0.969</td>
      <td>0.030</td>
      <td>175</td>
      <td>271</td>
    </tr>
    <tr>
      <td>w/o decoup-MLP</td>
      <td>33.95</td>
      <td>0.970</td>
      <td>0.030</td>
      <td>217</td>
      <td>244</td>
    </tr>
    <tr>
      <td>w/o normal</td>
      <td>34.10</td>
      <td>0.971</td>
      <td>0.029</td>
      <td>139</td>
      <td>238</td>
    </tr>
    <tr>
      <td>Full (Light)</td>
      <td>34.08</td>
      <td>0.970</td>
      <td>0.029</td>
      <td>148</td>
      <td>186</td>
    </tr>
    <tr>
      <td>Full</td>
      <td>34.19</td>
      <td>0.971</td>
      <td>0.028</td>
      <td>121</td>
      <td>237</td>
    </tr>
  </tbody>
</table>

Table 6: Ablation on full method.
<table>
  <thead>
    <tr>
      <th>Dataset Method</th>
      <th>PSNR ↑</th>
      <th>SSIM ↑</th>
      <th>LPIPS ↓</th>
      <th>FPS ↑</th>
      <th>Num.(M) ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3D-GS</td>
      <td>27.47</td>
      <td>0.812</td>
      <td>0.222</td>
      <td>115</td>
      <td>3.23</td>
    </tr>
    <tr>
      <td>w/o ASG field</td>
      <td>27.61</td>
      <td>0.830</td>
      <td>0.184</td>
      <td>113</td>
      <td>3.21</td>
    </tr>
    <tr>
      <td>w/o c2f</td>
      <td>28.01</td>
      <td>0.823</td>
      <td>0.203</td>
      <td>28</td>
      <td>3.56</td>
    </tr>
    <tr>
      <td>w/ anchor</td>
      <td>28.14</td>
      <td>0.824</td>
      <td>0.196</td>
      <td>70</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Full (Light)</td>
      <td>28.07</td>
      <td>0.834</td>
      <td>0.183</td>
      <td>44</td>
      <td>2.52</td>
    </tr>
    <tr>
      <td>Full</td>
      <td>28.18</td>
      <td>0.835</td>
      <td>0.176</td>
      <td>33</td>
      <td>3.10</td>
    </tr>
  </tbody>
</table>

&lt;img&gt;Figure 8: Visualization on Mip-NeRF 360 outdoor scenes. Our method achieves robust floater removal by coarse to fine training.&lt;/img&gt;

and complex transmission scenarios. Additionally, we compared it with the SOTA NeRF-based methods based on NeRF. Our approach enables 3D-GS to surpass the latest SOTA of NeRF, achieving high-frequency highlight modeling that 3D-GS couldn’t realize but NeRF could, thereby achieving truly high-quality rendering as shown in Fig. 14.

&lt;img&gt;Figure 9: More comparisons with baselines. Our method achieves robust floater removal by coarse-to-fine training.&lt;/img&gt;

&lt;page_number&gt;14&lt;/page_number&gt;

---


## Page 15

&lt;img&gt;Specular highlights and Reflection illustration&lt;/img&gt;
Figure 10: Illustration of specular highlights and reflections.

&lt;img&gt;GS-Shader, Ours, GT comparison on Ref-NeRF dataset&lt;/img&gt;
Figure 11: Comparison on Ref-NeRF dataset.

&lt;img&gt;3D-GS, Scaffold-GS, Ours, GT visualization on Nex [51] dataset&lt;/img&gt;
Figure 12: Visualization on Nex [51] dataset.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Chair</th>
      <th>Drums</th>
      <th>Ficus</th>
      <th>Hotdog</th>
      <th>Lego</th>
      <th>Materials</th>
      <th>Mic</th>
      <th>Ship</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>iNGP-Base</td>
      <td>35.00</td>
      <td>26.02</td>
      <td>33.51</td>
      <td>37.40</td>
      <td>36.39</td>
      <td>29.78</td>
      <td>36.22</td>
      <td>31.10</td>
      <td>33.18</td>
    </tr>
    <tr>
      <td>Mip-NeRF</td>
      <td>35.14</td>
      <td>25.48</td>
      <td>33.29</td>
      <td>37.48</td>
      <td>35.70</td>
      <td>30.71</td>
      <td>36.51</td>
      <td>30.41</td>
      <td>33.09</td>
    </tr>
    <tr>
      <td>Tri-MipRF</td>
      <td>36.10</td>
      <td>26.59</td>
      <td>34.51</td>
      <td>38.54</td>
      <td>36.15</td>
      <td>30.73</td>
      <td>37.75</td>
      <td>28.78</td>
      <td>33.65</td>
    </tr>
    <tr>
      <td>GS-Shader</td>
      <td>35.83</td>
      <td>26.36</td>
      <td>34.97</td>
      <td>37.85</td>
      <td>35.87</td>
      <td>30.07</td>
      <td>35.23</td>
      <td>30.82</td>
      <td>33.38</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>35.36</td>
      <td>26.15</td>
      <td>34.87</td>
      <td>37.72</td>
      <td>35.78</td>
      <td>30.00</td>
      <td>35.36</td>
      <td>30.80</td>
      <td>33.32</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>35.28</td>
      <td>26.44</td>
      <td>35.21</td>
      <td>37.73</td>
      <td>35.69</td>
      <td>30.65</td>
      <td>37.25</td>
      <td>31.17</td>
      <td>33.68</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>35.57</td>
      <td>26.58</td>
      <td>35.71</td>
      <td>38.12</td>
      <td>36.62</td>
      <td>30.66</td>
      <td>36.81</td>
      <td>31.63</td>
      <td>33.96</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>35.69</td>
      <td>26.77</td>
      <td>36.03</td>
      <td>38.25</td>
      <td>36.11</td>
      <td>30.84</td>
      <td>36.95</td>
      <td>31.97</td>
      <td>34.08</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>35.72</td>
      <td>26.92</td>
      <td>36.10</td>
      <td>38.25</td>
      <td>36.46</td>
      <td>30.98</td>
      <td>37.09</td>
      <td>31.97</td>
      <td>34.19</td>
    </tr>
  </tbody>
</table>
Table 7: Per-scene PSNR comparison on the NeRF dataset.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Chair</th>
      <th>Drums</th>
      <th>Ficus</th>
      <th>Hotdog</th>
      <th>Lego</th>
      <th>Materials</th>
      <th>Mic</th>
      <th>Ship</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>iNGP-Base</td>
      <td>0.979</td>
      <td>0.937</td>
      <td>0.981</td>
      <td>0.982</td>
      <td>0.982</td>
      <td>0.951</td>
      <td>0.990</td>
      <td>0.896</td>
      <td>0.963</td>
    </tr>
    <tr>
      <td>Mip-NeRF</td>
      <td>0.981</td>
      <td>0.932</td>
      <td>0.980</td>
      <td>0.982</td>
      <td>0.978</td>
      <td>0.959</td>
      <td>0.991</td>
      <td>0.882</td>
      <td>0.961</td>
    </tr>
    <tr>
      <td>Tri-MipRF</td>
      <td>0.985</td>
      <td>0.939</td>
      <td>0.983</td>
      <td>0.984</td>
      <td>0.982</td>
      <td>0.953</td>
      <td>0.992</td>
      <td>0.879</td>
      <td>0.963</td>
    </tr>
    <tr>
      <td>GS-Shader</td>
      <td>0.987</td>
      <td>0.949</td>
      <td>0.985</td>
      <td>0.985</td>
      <td>0.983</td>
      <td>0.960</td>
      <td>0.991</td>
      <td>0.905</td>
      <td>0.968</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>0.987</td>
      <td>0.955</td>
      <td>0.987</td>
      <td>0.985</td>
      <td>0.983</td>
      <td>0.960</td>
      <td>0.992</td>
      <td>0.907</td>
      <td>0.969</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>0.985</td>
      <td>0.950</td>
      <td>0.985</td>
      <td>0.983</td>
      <td>0.980</td>
      <td>0.960</td>
      <td>0.992</td>
      <td>0.898</td>
      <td>0.967</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>0.986</td>
      <td>0.953</td>
      <td>0.987</td>
      <td>0.985</td>
      <td>0.982</td>
      <td>0.962</td>
      <td>0.992</td>
      <td>0.904</td>
      <td>0.969</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>0.987</td>
      <td>0.955</td>
      <td>0.988</td>
      <td>0.985</td>
      <td>0.981</td>
      <td>0.963</td>
      <td>0.993</td>
      <td>0.905</td>
      <td>0.970</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>0.987</td>
      <td>0.958</td>
      <td>0.988</td>
      <td>0.985</td>
      <td>0.982</td>
      <td>0.963</td>
      <td>0.993</td>
      <td>0.909</td>
      <td>0.971</td>
    </tr>
  </tbody>
</table>
Table 8: Per-scene SSIM comparison on the NeRF dataset.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Chair</th>
      <th>Drums</th>
      <th>Ficus</th>
      <th>Hotdog</th>
      <th>Lego</th>
      <th>Materials</th>
      <th>Mic</th>
      <th>Ship</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>iNGP-Base</td>
      <td>0.022</td>
      <td>0.071</td>
      <td>0.023</td>
      <td>0.027</td>
      <td>0.017</td>
      <td>0.060</td>
      <td>0.010</td>
      <td>0.132</td>
      <td>0.045</td>
    </tr>
    <tr>
      <td>Mip-NeRF</td>
      <td>0.021</td>
      <td>0.065</td>
      <td>0.020</td>
      <td>0.027</td>
      <td>0.021</td>
      <td>0.040</td>
      <td>0.009</td>
      <td>0.138</td>
      <td>0.043</td>
    </tr>
    <tr>
      <td>Tri-MipRF</td>
      <td>0.016</td>
      <td>0.066</td>
      <td>0.020</td>
      <td>0.021</td>
      <td>0.016</td>
      <td>0.052</td>
      <td>0.008</td>
      <td>0.136</td>
      <td>0.042</td>
    </tr>
    <tr>
      <td>GS-Shader</td>
      <td>0.012</td>
      <td>0.040</td>
      <td>0.013</td>
      <td>0.019</td>
      <td>0.014</td>
      <td>0.033</td>
      <td>0.006</td>
      <td>0.103</td>
      <td>0.030</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>0.011</td>
      <td>0.037</td>
      <td>0.012</td>
      <td>0.020</td>
      <td>0.016</td>
      <td>0.037</td>
      <td>0.006</td>
      <td>0.106</td>
      <td>0.031</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>0.013</td>
      <td>0.042</td>
      <td>0.013</td>
      <td>0.023</td>
      <td>0.019</td>
      <td>0.040</td>
      <td>0.008</td>
      <td>0.114</td>
      <td>0.034</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>0.013</td>
      <td>0.038</td>
      <td>0.012</td>
      <td>0.022</td>
      <td>0.016</td>
      <td>0.037</td>
      <td>0.007</td>
      <td>0.112</td>
      <td>0.032</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>0.012</td>
      <td>0.035</td>
      <td>0.011</td>
      <td>0.019</td>
      <td>0.017</td>
      <td>0.034</td>
      <td>0.006</td>
      <td>0.101</td>
      <td>0.029</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>0.011</td>
      <td>0.033</td>
      <td>0.011</td>
      <td>0.018</td>
      <td>0.014</td>
      <td>0.031</td>
      <td>0.006</td>
      <td>0.099</td>
      <td>0.028</td>
    </tr>
  </tbody>
</table>
Table 9: Per-scene LPIPS (VGG) comparison on the NeRF dataset.

&lt;page_number&gt;15&lt;/page_number&gt;

---


## Page 16

&lt;img&gt;
    &lt;img&gt;Figure 13: Visualization on our "Anisotropic Synthetic" dataset. We show the comparison between our method and 3D-GS across all eight scenes. Qualitative experimental results demonstrate the significant advantage of our method in modeling anisotropic scenes, thereby enhancing the rendering quality of 3D-GS.&lt;/img&gt;
&lt;/img&gt;

&lt;img&gt;
    &lt;img&gt;Figure 14: Visualization on NSVF dataset. Our method significantly improves the ability to model metallic materials compared to other GS-based methods. At the same time, our method also demonstrates the capability to model refractive parts, reflecting the powerful fitting ability of our method.&lt;/img&gt;
&lt;/img&gt;

<table>
  <thead>
    <tr>
      <th></th>
      <th>Bike</th>
      <th>Life</th>
      <th>Palace</th>
      <th>Robot</th>
      <th>Space</th>
      <th>Steam</th>
      <th>Toad</th>
      <th>Wine</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NeRF</td>
      <td>31.77</td>
      <td>31.08</td>
      <td>31.76</td>
      <td>28.69</td>
      <td>34.66</td>
      <td>30.84</td>
      <td>29.42</td>
      <td>28.23</td>
      <td>30.81</td>
    </tr>
    <tr>
      <td>NSVF</td>
      <td>37.75</td>
      <td>34.60</td>
      <td>34.05</td>
      <td>35.24</td>
      <td>39.00</td>
      <td>35.13</td>
      <td>33.25</td>
      <td>32.04</td>
      <td>35.13</td>
    </tr>
    <tr>
      <td>TensoRF</td>
      <td>39.23</td>
      <td>34.51</td>
      <td>37.56</td>
      <td>38.26</td>
      <td>38.60</td>
      <td>37.87</td>
      <td>34.85</td>
      <td>31.32</td>
      <td>36.52</td>
    </tr>
    <tr>
      <td>Tri-MipRF</td>
      <td>36.98</td>
      <td>33.98</td>
      <td>36.55</td>
      <td>33.49</td>
      <td>37.60</td>
      <td>-</td>
      <td>33.48</td>
      <td>29.97</td>
      <td>34.58</td>
    </tr>
    <tr>
      <td>NeuRBF</td>
      <td>40.71</td>
      <td>36.08</td>
      <td>38.93</td>
      <td>39.13</td>
      <td>40.44</td>
      <td>38.35</td>
      <td>35.73</td>
      <td>32.99</td>
      <td>37.80</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>40.76</td>
      <td>33.19</td>
      <td>38.89</td>
      <td>39.16</td>
      <td>36.80</td>
      <td>37.67</td>
      <td>37.33</td>
      <td>32.76</td>
      <td>37.07</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>39.87</td>
      <td>35.00</td>
      <td>38.53</td>
      <td>37.92</td>
      <td>34.36</td>
      <td>37.12</td>
      <td>36.29</td>
      <td>32.32</td>
      <td>36.43</td>
    </tr>
    <tr>
      <td>GS-Shader</td>
      <td>37.38</td>
      <td>27.36</td>
      <td>36.55</td>
      <td>37.00</td>
      <td>32.61</td>
      <td>35.27</td>
      <td>34.50</td>
      <td>30.16</td>
      <td>33.85</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>40.63</td>
      <td>35.56</td>
      <td>38.95</td>
      <td>38.52</td>
      <td>39.47</td>
      <td>37.98</td>
      <td>36.55</td>
      <td>34.04</td>
      <td>37.71</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>41.48</td>
      <td>36.11</td>
      <td>39.23</td>
      <td>39.54</td>
      <td>39.89</td>
      <td>38.19</td>
      <td>37.22</td>
      <td>34.59</td>
      <td>38.28</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>41.67</td>
      <td>36.15</td>
      <td>39.33</td>
      <td>39.65</td>
      <td>40.03</td>
      <td>38.26</td>
      <td>37.43</td>
      <td>34.69</td>
      <td>38.40</td>
    </tr>
  </tbody>
</table>

Table 10: Per-scene PSNR comparison on the NSVF dataset.

&lt;page_number&gt;16&lt;/page_number&gt;

---


## Page 17

<table>
  <thead>
    <tr>
      <th></th>
      <th>Bike</th>
      <th>Life</th>
      <th>Palace</th>
      <th>Robot</th>
      <th>Space</th>
      <th>Steam</th>
      <th>Toad</th>
      <th>Wine</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NeRF</td>
      <td>0.970</td>
      <td>0.946</td>
      <td>0.950</td>
      <td>0.960</td>
      <td>0.980</td>
      <td>0.966</td>
      <td>0.920</td>
      <td>0.920</td>
      <td>0.952</td>
    </tr>
    <tr>
      <td>NSVF</td>
      <td>0.991</td>
      <td>0.971</td>
      <td>0.969</td>
      <td>0.988</td>
      <td>0.991</td>
      <td>0.986</td>
      <td>0.968</td>
      <td>0.965</td>
      <td>0.979</td>
    </tr>
    <tr>
      <td>TensoRF</td>
      <td>0.993</td>
      <td>0.968</td>
      <td>0.979</td>
      <td>0.994</td>
      <td>0.989</td>
      <td>0.991</td>
      <td>0.978</td>
      <td>0.961</td>
      <td>0.982</td>
    </tr>
    <tr>
      <td>Tri-MipRF</td>
      <td>0.990</td>
      <td>0.962</td>
      <td>0.973</td>
      <td>0.985</td>
      <td>0.986</td>
      <td>-</td>
      <td>0.968</td>
      <td>0.945</td>
      <td>0.973</td>
    </tr>
    <tr>
      <td>NeuRBF</td>
      <td>0.995</td>
      <td>0.977</td>
      <td>0.985</td>
      <td>0.995</td>
      <td>0.993</td>
      <td>0.993</td>
      <td>0.983</td>
      <td>0.972</td>
      <td>0.986</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>0.994</td>
      <td>0.979</td>
      <td>0.983</td>
      <td>0.994</td>
      <td>0.991</td>
      <td>0.993</td>
      <td>0.985</td>
      <td>0.975</td>
      <td>0.987</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>0.993</td>
      <td>0.979</td>
      <td>0.981</td>
      <td>0.995</td>
      <td>0.985</td>
      <td>0.992</td>
      <td>0.982</td>
      <td>0.971</td>
      <td>0.984</td>
    </tr>
    <tr>
      <td>GS-Shader</td>
      <td>0.992</td>
      <td>0.964</td>
      <td>0.979</td>
      <td>0.994</td>
      <td>0.985</td>
      <td>0.990</td>
      <td>0.980</td>
      <td>0.966</td>
      <td>0.981</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>0.994</td>
      <td>0.979</td>
      <td>0.982</td>
      <td>0.994</td>
      <td>0.993</td>
      <td>0.992</td>
      <td>0.984</td>
      <td>0.975</td>
      <td>0.987</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>0.995</td>
      <td>0.982</td>
      <td>0.984</td>
      <td>0.993</td>
      <td>0.994</td>
      <td>0.994</td>
      <td>0.984</td>
      <td>0.977</td>
      <td>0.988</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>0.995</td>
      <td>0.982</td>
      <td>0.984</td>
      <td>0.995</td>
      <td>0.994</td>
      <td>0.994</td>
      <td>0.985</td>
      <td>0.978</td>
      <td>0.988</td>
    </tr>
  </tbody>
</table>

Table 11: Per-scene SSIM comparison on the NSVF dataset.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Bike</th>
      <th>Life</th>
      <th>Palace</th>
      <th>Robot</th>
      <th>Space</th>
      <th>Steam</th>
      <th>Toad</th>
      <th>Wine</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TensoRF</td>
      <td>0.010</td>
      <td>0.048</td>
      <td>0.022</td>
      <td>0.010</td>
      <td>0.020</td>
      <td>0.017</td>
      <td>0.031</td>
      <td>0.051</td>
      <td>0.026</td>
    </tr>
    <tr>
      <td>Tri-MipRF</td>
      <td>0.012</td>
      <td>0.048</td>
      <td>0.023</td>
      <td>0.019</td>
      <td>0.019</td>
      <td>-</td>
      <td>0.036</td>
      <td>0.055</td>
      <td>0.030</td>
    </tr>
    <tr>
      <td>NeuRBF</td>
      <td>0.006</td>
      <td>0.036</td>
      <td>0.016</td>
      <td>0.009</td>
      <td>0.011</td>
      <td>0.011</td>
      <td>0.025</td>
      <td>0.036</td>
      <td>0.019</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>0.005</td>
      <td>0.028</td>
      <td>0.017</td>
      <td>0.006</td>
      <td>0.009</td>
      <td>0.007</td>
      <td>0.018</td>
      <td>0.025</td>
      <td>0.015</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>0.007</td>
      <td>0.030</td>
      <td>0.019</td>
      <td>0.008</td>
      <td>0.019</td>
      <td>0.010</td>
      <td>0.022</td>
      <td>0.021</td>
      <td>0.017</td>
    </tr>
    <tr>
      <td>GS-Shader</td>
      <td>0.007</td>
      <td>0.051</td>
      <td>0.020</td>
      <td>0.008</td>
      <td>0.016</td>
      <td>0.010</td>
      <td>0.023</td>
      <td>0.029</td>
      <td>0.020</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>0.005</td>
      <td>0.027</td>
      <td>0.018</td>
      <td>0.007</td>
      <td>0.007</td>
      <td>0.008</td>
      <td>0.021</td>
      <td>0.025</td>
      <td>0.015</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>0.005</td>
      <td>0.024</td>
      <td>0.015</td>
      <td>0.006</td>
      <td>0.007</td>
      <td>0.007</td>
      <td>0.018</td>
      <td>0.022</td>
      <td>0.013</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>0.004</td>
      <td>0.022</td>
      <td>0.014</td>
      <td>0.005</td>
      <td>0.007</td>
      <td>0.007</td>
      <td>0.017</td>
      <td>0.021</td>
      <td>0.012</td>
    </tr>
  </tbody>
</table>

Table 12: Per-scene LPIPS (VGG) comparison on the NSVF dataset.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Teapot</th>
      <th>Plane</th>
      <th>Record</th>
      <th>Ashtray</th>
      <th>Dishes</th>
      <th>Headphone</th>
      <th>Jupyter</th>
      <th>Lock</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3D-GS</td>
      <td>27.24</td>
      <td>26.80</td>
      <td>43.81</td>
      <td>34.43</td>
      <td>29.62</td>
      <td>38.72</td>
      <td>40.52</td>
      <td>29.36</td>
      <td>33.81</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>30.64</td>
      <td>29.14</td>
      <td>47.79</td>
      <td>35.66</td>
      <td>32.12</td>
      <td>37.19</td>
      <td>40.04</td>
      <td>30.13</td>
      <td>35.34</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>33.53</td>
      <td>31.56</td>
      <td>50.35</td>
      <td>36.14</td>
      <td>32.95</td>
      <td>38.48</td>
      <td>40.10</td>
      <td>30.96</td>
      <td>36.76</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>34.75</td>
      <td>31.01</td>
      <td>50.30</td>
      <td>37.76</td>
      <td>33.03</td>
      <td>39.39</td>
      <td>41.42</td>
      <td>31.68</td>
      <td>37.42</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>35.24</td>
      <td>30.95</td>
      <td>50.90</td>
      <td>38.03</td>
      <td>33.04</td>
      <td>40.12</td>
      <td>41.47</td>
      <td>31.86</td>
      <td>37.70</td>
    </tr>
  </tbody>
</table>

Table 13: Per-scene PSNR comparison on our "Anisotropic Synthetic" dataset.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Teapot</th>
      <th>Plane</th>
      <th>Record</th>
      <th>Ashtray</th>
      <th>Dishes</th>
      <th>Headphone</th>
      <th>Jupyter</th>
      <th>Lock</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3D-GS</td>
      <td>0.968</td>
      <td>0.946</td>
      <td>0.994</td>
      <td>0.969</td>
      <td>0.947</td>
      <td>0.989</td>
      <td>0.985</td>
      <td>0.932</td>
      <td>0.966</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>0.979</td>
      <td>0.965</td>
      <td>0.998</td>
      <td>0.973</td>
      <td>0.967</td>
      <td>0.986</td>
      <td>0.983</td>
      <td>0.924</td>
      <td>0.972</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>0.985</td>
      <td>0.973</td>
      <td>0.999</td>
      <td>0.974</td>
      <td>0.973</td>
      <td>0.988</td>
      <td>0.984</td>
      <td>0.930</td>
      <td>0.976</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>0.987</td>
      <td>0.967</td>
      <td>0.998</td>
      <td>0.984</td>
      <td>0.970</td>
      <td>0.990</td>
      <td>0.987</td>
      <td>0.948</td>
      <td>0.979</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>0.988</td>
      <td>0.967</td>
      <td>0.999</td>
      <td>0.985</td>
      <td>0.970</td>
      <td>0.990</td>
      <td>0.987</td>
      <td>0.951</td>
      <td>0.980</td>
    </tr>
  </tbody>
</table>

Table 14: Per-scene SSIM comparison on our "Anisotropic Synthetic" dataset.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Teapot</th>
      <th>Plane</th>
      <th>Record</th>
      <th>Ashtray</th>
      <th>Dishes</th>
      <th>Headphone</th>
      <th>Jupyter</th>
      <th>Lock</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3D-GS</td>
      <td>0.043</td>
      <td>0.085</td>
      <td>0.019</td>
      <td>0.044</td>
      <td>0.120</td>
      <td>0.015</td>
      <td>0.075</td>
      <td>0.098</td>
      <td>0.062</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>0.029</td>
      <td>0.057</td>
      <td>0.006</td>
      <td>0.038</td>
      <td>0.082</td>
      <td>0.021</td>
      <td>0.086</td>
      <td>0.099</td>
      <td>0.052</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>0.022</td>
      <td>0.042</td>
      <td>0.004</td>
      <td>0.039</td>
      <td>0.067</td>
      <td>0.017</td>
      <td>0.084</td>
      <td>0.093</td>
      <td>0.046</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>0.021</td>
      <td>0.052</td>
      <td>0.007</td>
      <td>0.022</td>
      <td>0.079</td>
      <td>0.014</td>
      <td>0.076</td>
      <td>0.080</td>
      <td>0.044</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>0.021</td>
      <td>0.051</td>
      <td>0.005</td>
      <td>0.020</td>
      <td>0.077</td>
      <td>0.013</td>
      <td>0.071</td>
      <td>0.075</td>
      <td>0.042</td>
    </tr>
  </tbody>
</table>

Table 15: Per-scene LPIPS (VGG) comparison on our "Anisotropic Synthetic" dataset.

<table>
  <thead>
    <tr>
      <th></th>
      <th>bicycle</th>
      <th>flowers</th>
      <th>garden</th>
      <th>stump</th>
      <th>treehill</th>
      <th>room</th>
      <th>counter</th>
      <th>kitchen</th>
      <th>bonsai</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Plenoxels</td>
      <td>21.91</td>
      <td>20.10</td>
      <td>23.49</td>
      <td>20.66</td>
      <td>22.25</td>
      <td>27.59</td>
      <td>23.62</td>
      <td>23.42</td>
      <td>24.67</td>
    </tr>
    <tr>
      <td>iNGP</td>
      <td>22.17</td>
      <td>20.65</td>
      <td>25.07</td>
      <td>23.47</td>
      <td>22.37</td>
      <td>29.69</td>
      <td>26.69</td>
      <td>29.48</td>
      <td>30.69</td>
    </tr>
    <tr>
      <td>Mip-NeRF360</td>
      <td>24.37</td>
      <td>21.73</td>
      <td>26.98</td>
      <td>26.40</td>
      <td>22.87</td>
      <td>31.63</td>
      <td>29.55</td>
      <td>32.23</td>
      <td>33.46</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>25.63</td>
      <td>21.94</td>
      <td>27.73</td>
      <td>27.02</td>
      <td>22.79</td>
      <td>31.80</td>
      <td>29.12</td>
      <td>31.61</td>
      <td>32.48</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>25.61</td>
      <td>21.74</td>
      <td>27.82</td>
      <td>26.79</td>
      <td>23.38</td>
      <td>32.14</td>
      <td>29.62</td>
      <td>31.81</td>
      <td>32.87</td>
    </tr>
    <tr>
      <td>Ours-w/ anchor</td>
      <td>25.44</td>
      <td>21.36</td>
      <td>27.97</td>
      <td>26.91</td>
      <td>23.23</td>
      <td>32.27</td>
      <td>30.12</td>
      <td>32.04</td>
      <td>33.91</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>25.87</td>
      <td>21.81</td>
      <td>28.06</td>
      <td>27.23</td>
      <td>22.48</td>
      <td>32.11</td>
      <td>29.74</td>
      <td>32.09</td>
      <td>33.26</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>25.90</td>
      <td>21.86</td>
      <td>28.07</td>
      <td>27.25</td>
      <td>22.48</td>
      <td>32.11</td>
      <td>30.12</td>
      <td>32.25</td>
      <td>33.54</td>
    </tr>
  </tbody>
</table>

Table 16: Per-scene PSNR comparison on the Mip-NeRF 360 dataset.

&lt;page_number&gt;17&lt;/page_number&gt;

---


## Page 18

<table>
  <thead>
    <tr>
      <th></th>
      <th>bicycle</th>
      <th>flowers</th>
      <th>garden</th>
      <th>stump</th>
      <th>treehill</th>
      <th>room</th>
      <th>counter</th>
      <th>kitchen</th>
      <th>bonsai</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Plenoxels</td>
      <td>0.496</td>
      <td>0.431</td>
      <td>0.606</td>
      <td>0.523</td>
      <td>0.509</td>
      <td>0.842</td>
      <td>0.759</td>
      <td>0.648</td>
      <td>0.814</td>
    </tr>
    <tr>
      <td>iNGP</td>
      <td>0.512</td>
      <td>0.486</td>
      <td>0.701</td>
      <td>0.594</td>
      <td>0.542</td>
      <td>0.871</td>
      <td>0.817</td>
      <td>0.858</td>
      <td>0.906</td>
    </tr>
    <tr>
      <td>Mip-NeRF360</td>
      <td>0.685</td>
      <td>0.583</td>
      <td>0.813</td>
      <td>0.744</td>
      <td>0.632</td>
      <td>0.913</td>
      <td>0.894</td>
      <td>0.920</td>
      <td>0.941</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>0.778</td>
      <td>0.623</td>
      <td>0.874</td>
      <td>0.784</td>
      <td>0.651</td>
      <td>0.928</td>
      <td>0.916</td>
      <td>0.933</td>
      <td>0.948</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>0.773</td>
      <td>0.609</td>
      <td>0.867</td>
      <td>0.774</td>
      <td>0.657</td>
      <td>0.931</td>
      <td>0.919</td>
      <td>0.931</td>
      <td>0.950</td>
    </tr>
    <tr>
      <td>Ours-w anchor</td>
      <td>0.775</td>
      <td>0.611</td>
      <td>0.869</td>
      <td>0.775</td>
      <td>0.645</td>
      <td>0.932</td>
      <td>0.920</td>
      <td>0.934</td>
      <td>0.953</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>0.795</td>
      <td>0.645</td>
      <td>0.879</td>
      <td>0.795</td>
      <td>0.647</td>
      <td>0.934</td>
      <td>0.920</td>
      <td>0.936</td>
      <td>0.952</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>0.797</td>
      <td>0.648</td>
      <td>0.881</td>
      <td>0.797</td>
      <td>0.647</td>
      <td>0.935</td>
      <td>0.923</td>
      <td>0.937</td>
      <td>0.953</td>
    </tr>
  </tbody>
</table>

Table 17: SSIM Comparison on the Mip-NeRF 360 dataset.

<table>
  <thead>
    <tr>
      <th></th>
      <th>bicycle</th>
      <th>flowers</th>
      <th>garden</th>
      <th>stump</th>
      <th>treehill</th>
      <th>room</th>
      <th>counter</th>
      <th>kitchen</th>
      <th>bonsai</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Plenoxels</td>
      <td>0.506</td>
      <td>0.521</td>
      <td>0.386</td>
      <td>0.503</td>
      <td>0.540</td>
      <td>0.419</td>
      <td>0.441</td>
      <td>0.447</td>
      <td>0.398</td>
    </tr>
    <tr>
      <td>iNGP</td>
      <td>0.446</td>
      <td>0.441</td>
      <td>0.257</td>
      <td>0.421</td>
      <td>0.450</td>
      <td>0.261</td>
      <td>0.306</td>
      <td>0.195</td>
      <td>0.205</td>
    </tr>
    <tr>
      <td>Mip-NeRF360</td>
      <td>0.301</td>
      <td>0.344</td>
      <td>0.170</td>
      <td>0.261</td>
      <td>0.339</td>
      <td>0.211</td>
      <td>0.204</td>
      <td>0.127</td>
      <td>0.176</td>
    </tr>
    <tr>
      <td>3D-GS</td>
      <td>0.204</td>
      <td>0.328</td>
      <td>0.103</td>
      <td>0.207</td>
      <td>0.318</td>
      <td>0.191</td>
      <td>0.178</td>
      <td>0.113</td>
      <td>0.173</td>
    </tr>
    <tr>
      <td>Scaffold-GS</td>
      <td>0.224</td>
      <td>0.339</td>
      <td>0.112</td>
      <td>0.228</td>
      <td>0.315</td>
      <td>0.182</td>
      <td>0.177</td>
      <td>0.114</td>
      <td>0.174</td>
    </tr>
    <tr>
      <td>Ours-w anchor</td>
      <td>0.205</td>
      <td>0.292</td>
      <td>0.110</td>
      <td>0.215</td>
      <td>0.293</td>
      <td>0.185</td>
      <td>0.179</td>
      <td>0.115</td>
      <td>0.166</td>
    </tr>
    <tr>
      <td>Ours-light</td>
      <td>0.173</td>
      <td>0.279</td>
      <td>0.097</td>
      <td>0.190</td>
      <td>0.275</td>
      <td>0.182</td>
      <td>0.173</td>
      <td>0.111</td>
      <td>0.168</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>0.166</td>
      <td>0.263</td>
      <td>0.092</td>
      <td>0.184</td>
      <td>0.269</td>
      <td>0.177</td>
      <td>0.166</td>
      <td>0.108</td>
      <td>0.162</td>
    </tr>
  </tbody>
</table>

Table 18: LPIPS Comparison on the Mip-NeRF 360 dataset.

C.3 Anisotropic Synthetic Scenes

"Anisotropic Synthetic" is a synthetic dataset we rendered ourselves, which includes 8 scenes with significant anisotropy. We tested some existing 3D-GS-based methods on "Anisotropic Synthetic." As shown in Tabs. 13-15, our method achieved a very significant improvement in rendering metrics. Fig. 13 shows the comparison between our method and 3D-GS across all eight scenes. Qualitative experiments also demonstrate the significant visual advantages of our method, highlighting the substantial improvement our method brings to anisotropic parts, thereby enhancing the overall rendering quality.

C.4 Mip-360 Scenes

The MipNeRF-360 scenes include five outdoor and four indoor scenarios. There are several scenes rich in specular reflections, such as bonsai, room, and kitchen. As shown in Tabs. 16-18, our method achieved significant advantages in the four indoor scenes. This reflects our method's strengths in modeling specular reflections and anisotropy. In outdoor scenes, our method also achieved rendering metrics comparable to the SOTA methods. Furthermore, with the help of the coarse-to-fine training mechanism, our method significantly reduced the number of floaters as shown in Fig. 11, resulting in a substantial improvement in visual effects.

&lt;page_number&gt;18&lt;/page_number&gt;

---


## Page 19

# NeurIPS Paper Checklist

## 1. Claims

**Question:** Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

**Answer:** [Yes]

**Justification:** I am sure that the abstract and introduction accurately reflect the paper's contributions and scope.

**Guidelines:**
* The answer NA means that the abstract and introduction do not include the claims made in the paper.
* The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
* The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
* It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

**Question:** Does the paper discuss the limitations of the work performed by the authors?

**Answer:** [Yes]

**Justification:** I am sure that the paper discuss the limitations of the work.

**Guidelines:**
* The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
* The authors are encouraged to create a separate "Limitations" section in their paper.
* The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
* The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
* The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
* The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
* If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
* While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

**Question:** For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

**Answer:** [Yes]

&lt;page_number&gt;19&lt;/page_number&gt;

---


## Page 20

Justification: Yes, the paper does.
Guidelines:
* The answer NA means that the paper does not include theoretical results.
* All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
* All assumptions should be clearly stated or referenced in the statement of any theorems.
* The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
* Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
* Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: Yes, the paper does.
Guidelines:
* The answer NA means that the paper does not include experiments.
* If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
* If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
* Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general, releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
* While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
    (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
    (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
    (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
    (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

&lt;page_number&gt;20&lt;/page_number&gt;

---


## Page 21

Answer: [No]
Justification: The code can be released upon acceptance, but now it’s not a clean version.
Guidelines:
* The answer NA means that paper does not include experiments requiring code.
* Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.
* While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
* The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.
* The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
* The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
* At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
* Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details
Question: Does the paper specify all the training and test details (e.g., data splits, hyper-parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes]
Justification: The paper contains details about the training model.
Guidelines:
* The answer NA means that the paper does not include experiments.
* The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
* The full details can be provided either with the code, in appendix, or as supplemental material.

7. Experiment Statistical Significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes]
Justification: We evaluate the results through PSNR, SSIM and LPIPS.
Guidelines:
* The answer NA means that the paper does not include experiments.
* The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
* The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
* The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
* The assumptions made should be given (e.g., Normally distributed errors).
* It should be clear whether the error bar is the standard deviation or the standard error of the mean.

&lt;page_number&gt;21&lt;/page_number&gt;

---


## Page 22

* It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
* For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
* If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

8. Experiments Compute Resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes]
Justification: They are presented in the paper.
Guidelines:
* The answer NA means that the paper does not include experiments.
* The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
* The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
* The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

9. Code Of Ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: Yes, we do.
Guidelines:
* The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
* If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
* The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

10. Broader Impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA]
Justification: There is no societal impact of the work performed.
Guidelines:
* The answer NA means that there is no societal impact of the work performed.
* If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
* Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
* The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to

&lt;page_number&gt;22&lt;/page_number&gt;

---


## Page 23

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
* The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
* If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

11. **Safeguards**
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper poses no such risks.
Guidelines:
* The answer NA means that the paper poses no such risks.
* Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
* Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
* We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

12. **Licenses for existing assets**
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes]
Justification: Yes, they do.
Guidelines:
* The answer NA means that the paper does not use existing assets.
* The authors should cite the original paper that produced the code package or dataset.
* The authors should state which version of the asset is used and, if possible, include a URL.
* The name of the license (e.g., CC-BY 4.0) should be included for each asset.
* For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
* If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
* For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
* If this information is not available online, the authors are encouraged to reach out to the asset's creators.

13. **New Assets**
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

&lt;page_number&gt;23&lt;/page_number&gt;

---


## Page 24

Answer: [NA]
Justification: The paper does not release new assets.
Guidelines:
* The answer NA means that the paper does not release new assets.
* Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
* The paper should discuss whether and how consent was obtained from people whose asset is used.
* At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

14. **Crowdsourcing and Research with Human Subjects**
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
* The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
* Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
* According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

15. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects**
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
* The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
* Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
* We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
* For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

&lt;page_number&gt;24&lt;/page_number&gt;