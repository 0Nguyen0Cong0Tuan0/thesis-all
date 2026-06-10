## Page 1

FastGS: Training 3D Gaussian Splatting in 100 Seconds

Shiwei Ren* Tianci Wen* Yongchun Fang† Biao Lu
Nankai University
renshiwei, wentc, lubiao@mail.nankai.edu.cn, fangyc@nankai.edu.cn
https://fastgs.github.io

&lt;img&gt;Figure 1. We propose FastGS, a general acceleration framework for 3D Gaussian Splatting (3DGS) that significantly reduces training time without sacrificing rendering quality. In static scenes, our method completes training on the Tanks & Temples train scene within 100 seconds. Furthermore, our method achieves 2.82× and 2.24× faster training for dynamic and surface reconstruction, respectively.&lt;/img&gt;

| Method | Time/min | PSNR/dB | N_GS/M |
| :--- | :--- | :--- | :--- |
| Speedy-Splat | 5.42 | 21.59 | 0.11 |
| FastGS(ours) | 1.33 | 22.57 | 0.23 |
| Taming-3DGS | 3.03 | 22.50 | 0.37 |
| DashGaussian | 4.32 | 22.19 | 1.00 |
| FastGS-Big(ours) | 1.93 | 22.68 | 0.46 |
| Deformable-3DGS | 21.55 | 26.86 | 0.15 |
| Deformable-3DGS+ours | 7.63 | 32.20 | 0.07 |
| PGSR | 34.17 | 24.92 | 1.71 |
| PGSR+ours | 15.27 | 25.07 | 0.33 |

**Abstract**
The dominant 3D Gaussian splatting (3DGS) acceleration methods fail to properly regulate the number of Gaussians during training, causing redundant computational time overhead. In this paper, we propose FastGS, a novel, simple, and general acceleration framework that fully considers the importance of each Gaussian based on multi-view consistency, efficiently solving the trade-off between training time and rendering quality. We innovatively design a densification and pruning strategy based on multi-view consistency, dispensing with the budgeting mechanism. Extensive experiments on Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets demonstrate that our method significantly outperforms the state-of-the-art methods in training speed, achieving a 3.29× training acceleration and comparable rendering quality compared with DashGaussian on the Mip-NeRF 360 dataset and a 15.45× acceleration compared with vanilla 3DGS on the Deep Blending dataset. We demonstrate that FastGS exhibits strong generality, delivering 2-6× training acceleration across various tasks, including dynamic scene reconstruction, surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and simultaneous localization and mapping.

**1. Introduction**
Novel view synthesis (NVS) is a fundamental problem in computer vision and graphics, with broad applications in augmented reality [46], virtual reality [40], and autonomous driving [38]. Neural Radiance Field (NeRF) [26] methods model scenes as continuous volumetric functions and render photorealistic views, but require hours of training per scene. Recently, 3D Gaussian Splatting (3DGS) [17] has achieved rendering quality comparable to NeRF while offering significantly faster training and rendering speed. It models 3D scenes via explicit Gaussian primitives and employs a tile-based rasterizer. Benefiting from its efficiency, 3DGS has been successfully applied to a wide range of tasks, including dynamic scene reconstruction, surface reconstruction, and simultaneous localization and mapping (SLAM). However, a major bottleneck in its current practical application is the extended training time, which often requires tens of minutes per scene, hindering user-friendly deployment.

*Equal contribution.
†Corresponding author.

&lt;page_number&gt;1&lt;/page_number&gt;
&lt;watermark&gt;arXiv:2511.04283v3 [cs.CV] 6 Dec 2025&lt;/watermark&gt;

---


## Page 2

A detailed analysis of the vanilla 3DGS [17] training pipeline reveals two primary limitations: (1) its adaptive density control (ADC) of Gaussians often introduces numerous redundant Gaussians and (2) inefficiencies in the rendering pipeline. While recent works [9, 11, 12, 24] have significantly optimized the rendering pipeline, ADC remains a major area for improvement.

ADC in vanilla 3DGS [17] comprises two main components. The first is Gaussian densification, which clones or splits a Gaussian based on its positional gradient. The second is Gaussian pruning, which removes Gaussians with low opacity or oversized scales. Existing 3DGS acceleration methods [4, 6, 8, 10, 12, 13, 18, 24, 27, 35] have introduced improvements to ADC. One direction of improvement focuses on designing mechanisms to constrain Gaussian densification, aiming to minimize the growth of redundant Gaussians. For example, Taming-3DGS [24] employs a budget-constrained optimization to control Gaussian growth. Similarly, DashGaussian [4] leverages an adaptive Gaussian primitive budgeting method to maintain continuous densification throughout training. The other direction involves refining the pruning strategy to accelerate training by deleting a greater number of Gaussians. For example, Speedy-Splat [12] applies a soft pruning strategy during densification and a hard pruning strategy afterward.

One major drawback of these methods is the limited effectiveness of their densification and pruning strategies, which fail to maintain rendering quality while avoiding excessive Gaussian redundancy, resulting in an inefficient representation, as illustrated in Fig. 2. This indicates that their Gaussian control strategies are suboptimal. Based on our observations, some densification and pruning methods [4, 10, 18, 33] do not leverage multi-view consistency, while others [12, 13, 24] exploit it suboptimally. Specifically, they enforce multi-view consistency merely through Gaussian-associated scores, which we argue is insufficient. It may lead to the excessive growth of redundant Gaussians that provide only marginal improvements to the rendering quality from a few viewpoints while contributing little to others. On one hand, for certain densification methods such as Taming-3DGS [24], Gaussian importance is considered across views. However, it fully relies on Gaussian-associated scores rather than their actual contribution to rendering quality, resulting in weak multi-view constraints and leading to redundancy. Moreover, it lacks a dedicated redesign of the pruning strategy. On the other hand, for some pruning methods, such as Speedy-Splat [12], multi-view information is also considered, but it uses the gradients of Gaussian rather than evaluating each Gaussian's contribution to multi-view rendering quality. This indirect enforcement of multi-view consistency leads to significant degradation in rendering quality.

To address the above issues, we propose FastGS, a new, simple, and general 3DGS acceleration framework, capable of training a scene in around 100 seconds while maintaining comparable rendering quality, as shown in Fig. 1. In fact, nearly every Gaussian primitive participates in rendering the same region across multiple viewpoints. Our insight is similar to the concept behind bundle adjustment in traditional 3D reconstruction, where each 3D Gaussian should maintain multi-view consistency. This implies that the 3D Gaussian should enhance rendering quality across multiple views of the same region. Therefore, we introduce a multi-view consistent densification (VCD) strategy, which uses a multi-view reconstruction quality importance score to evaluate whether a Gaussian contributes beneficially to the improvement of multi-view rendering quality. Based on the same idea, we propose a multi-view consistent pruning (VCP) strategy, which removes redundant Gaussians that are useless to multi-view rendering quality. Notably, because VCD and VCP accurately identify which Gaussians need to be densified or pruned, our method does not require a budget mechanism, making it easily applicable to other tasks. To summarize our contributions,

1. New, simple, and general framework for 3DGS acceleration that can train a scene in around 100 seconds while achieving comparable rendering quality.
2. Efficient densification and pruning strategy strictly controlling the addition and removal of each Gaussian based on its contribution to multi-view reconstruction quality, greatly accelerating the training process.
3. General and state-of-the-art performance across various tasks. Our method outperforms state-of-the-art (SOTA) methods in training speed while maintaining comparable rendering quality on static scenes. It generalizes well to dynamic scene reconstruction, surface reconstruction, sparse-view reconstruction, large-scale recon-

&lt;img&gt;
Figure 2. Gaussian count over training iterations. Benefiting from the efficient VCD and VCP strategies, FastGS keeps the number of Gaussians consistently low throughout the entire training process on the treehill scene of Mip-NeRF 360 [2].
&lt;/img&gt;

| Iteration | Taming-3DGS (5.37min, 22.92dB) | Mini-Splatting (16.85min, 22.62dB) | DashGaussian (7.93min, 22.94dB) | Speedy-Splat (13.05min, 22.50dB) | 3DGS (20.30min, 22.58dB) | FastGS (1.72min, 22.94dB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3000 | 1.2 | 0.8 | 1.3 | 1.1 | 1.2 | 0.5 |
| 6000 | 1.8 | 1.2 | 2.1 | 1.7 | 1.8 | 0.6 |
| 9000 | 2.3 | 1.5 | 2.6 | 2.1 | 2.3 | 0.7 |
| 12000 | 2.7 | 1.7 | 3.0 | 2.4 | 2.7 | 0.8 |
| 15000 | 3.0 | 1.9 | 3.3 | 2.6 | 3.0 | 0.9 |
| 18000 | 3.2 | 2.0 | 3.4 | 2.8 | 3.2 | 0.9 |
| 21000 | 3.3 | 2.1 | 3.4 | 2.9 | 3.3 | 0.8 |
| 24000 | 3.4 | 2.1 | 3.4 | 3.0 | 3.4 | 0.7 |
| 27000 | 3.4 | 2.1 | 3.4 | 3.0 | 3.4 | 0.6 |
| 30000 | 3.4 | 2.1 | 3.4 | 3.0 | 3.4 | 0.5 |

&lt;page_number&gt;2&lt;/page_number&gt;

---


## Page 3

struction, and SLAM.

## 2. Related Work

In this section, we review related works that focus on accelerating both the training and inference of 3DGS [17].

**Gaussian Densification.** A key bottleneck of 3DGS is the excessive Gaussians that significantly slow down the training process. Recent works [4, 6, 18, 24, 32] attempt to address this issue by refining the densification strategy to control the primitive count. Specifically, Taming-3DGS [24] employs a budgeting mechanism to control Gaussian growth, primarily computing importance scores based on Gaussian-associated properties. DashGaussian [4] introduces a resolution-guided primitive scheduler that progressively reconstructs the scene throughout the entire training process. Nevertheless, they still require millions of Gaussians to maintain rendering quality, resulting in heavy optimization overhead.

**Gaussian Pruning.** Besides modifying densification, some methods [1, 8, 10, 12, 13, 27, 33] achieve acceleration by designing pruning strategies to remove a large number of Gaussians. Some of them design importance scores according to Gaussian-associated properties to guide pruning, discarding less critical Gaussians [1, 10, 33]. Mini-Splatting [8] removes a large number of Gaussians through a simplification strategy based on intersection preserving and sampling. PUP 3DGS [13] and Speedy-Splat [12] remove Gaussians by computing a Gaussian-associated Hessian approximation across all training views. Nonetheless, some of them fail to fully remove redundant Gaussians, while others remove many Gaussians at the cost of significantly degraded rendering quality.

**Other Methods.** Some works focus on optimizing 3DGS rasterization or optimization strategies. Taming-3DGS [24] replaces per-pixel with per-splat parallel backpropagation, which significantly speeds up the optimization process and serves as a strong baseline for subsequent research. StopThePop [30], FlashGS [9], and Speedy-Splat [12] use precise tile intersection to reduce Gaussian-tile pairs and accelerate rasterization. 3DGS-LM [15] replaces Adam with Levenberg-Marquardt for faster convergence, and 3DGS² [21] achieves near second-order convergence via prioritized per-kernel updates.

## 3. Background: 3D Gaussian Splatting

3DGS models a scene as an explicit point-based representation composed of a set of anisotropic 3D Gaussians:
$$
\{G_i(\mathbf{x}) = \exp(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_{i3D}^{-1}(\mathbf{x} - \boldsymbol{\mu}_i))\}_{i=1}^N. \quad (1)
$$
To construct this representation, a sparse point cloud obtained from SfM is used to initialize the positions of Gaussian primitives. Each primitive $G_i$ is parameterized by mean $\boldsymbol{\mu}_i \in \mathbb{R}^3$, rotation $r_i \in \mathbb{R}^4$, scale $s_i \in \mathbb{R}^3$, opacity $\sigma_i \in \mathbb{R}$, and color coefficients $\mathbf{c}_i \in \mathbb{R}^{16 \times 3}$ represented in view-dependent spherical harmonics (SH). The rotation and scale together define the covariance matrix as
$$
\boldsymbol{\Sigma}_{i3D} = R_i S_i S_i^\top R_i^\top. \quad (2)
$$
To render from a given camera viewpoint, all 3D Gaussians need to be projected into the 2D image plane. Given a viewing transformation $W$, the covariance matrix in camera coordinates is computed as
$$
\boldsymbol{\Sigma}'_{i3D} = J W \boldsymbol{\Sigma}_{i3D} W^\top J^\top, \quad (3)
$$
where $J$ denotes the Jacobian of the affine approximation of the projective transformation. The projected 3D Gaussian is then approximated as a 2D elliptical Gaussian on the image plane, with covariance $\boldsymbol{\Sigma}'_{i2D}$ obtained by marginalizing $\boldsymbol{\Sigma}'_{i3D}$ along the viewing direction. Each 2D Gaussian contributes to pixels within its footprint using $\alpha$-blending: given Gaussians $\{G_i\}_{i=1}^N$ sorted by depth, the accumulated color of pixel $p$ is computed as
$$
C(p) = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j), \quad \alpha_i = \sigma_i G'_i(p), \quad (4)
$$
where
$$
G'_i(p) = \exp\left(-\frac{1}{2}(p - \boldsymbol{\mu}_{i2D})^\top \boldsymbol{\Sigma}_{i2D}^{-1}(p - \boldsymbol{\mu}_{i2D})\right), \quad (5)
$$
with $\boldsymbol{\mu}_{i2D}$ and $\boldsymbol{\Sigma}_{i2D}$ denoting the mean and covariance of the projected 2D Gaussian, respectively.

## 4. FastGS

### 4.1. Overview

The framework of our method is shown in Fig. 3. We initialize 3D Gaussians using SfM point clouds and train the 3DGS model on multi-view images. The addition and removal of 3D Gaussians are controlled via the proposed multi-view consistent densification and pruning strategies. As illustrated in Fig. 3b and Fig. 3c, Taming-3DGS [24] and Speedy-Splat [12] also estimate importance scores from multiple views, for densification and pruning, respectively. However, both rely on Gaussian-associated scores to control the number of Gaussians. This suboptimal use of multi-view information results in redundancy for Taming-3DGS [24] and degraded rendering quality for Speedy-Splat [12]. In contrast, as illustrated in Fig. 3a, our method evaluates the importance of each Gaussian based on multi-view reconstruction quality, rather than Gaussian-associated properties. Furthermore, our method leverages multi-view consistency constraints to effectively guide both densification and pruning, which will be detailed in Sections Sec. 4.2

&lt;page_number&gt;3&lt;/page_number&gt;

---


## Page 4

&lt;img&gt;Figure 3. The pipeline of FastGS. (a) We redesign the ADC of the vanilla 3DGS [17] based on multi-view consistency. To accurately assess the importance of each Gaussian, we sample training views and generate the corresponding per-pixel L1 loss maps. For each sampled view, a multi-view score is computed for each Gaussian by counting the number of high-error pixels within its 2D footprint, which is subsequently used to guide Gaussian densification and pruning. (b) Taming-3DGS [24] primarily computes the importance score based on Gaussian-associated properties across sampled views. (c) Speedy-Splat [12] computes the Gaussian score by accumulating Gaussian-associated Hessian approximations across all training views. We visualize the densification results from 0.5K to 15K iterations without pruning on the far left, and pruning results on the far right using Speedy-Splat [12]'s pruning strategy and VCP on vanilla 3DGS [17].&lt;/img&gt;

| Component | Input/Process | Output/Score |
| :--- | :--- | :--- |
| (a) FastGS | Sampled View, Loss Map, Image Space, Gaussian | Multi-view Score ($S_d$, $S_p$) |
| (b) Taming-3DGS | Sampled Views $V_K$, $L_1 + E_{edge}$ | $S_G = \sum F(o, s, d, g, ...)$ |
| (c) Speedy-Splat | Training Views $V_1...V_N$, $\nabla_G$ | $S_G = \sum (\nabla_G)^2$ |

and Sec. 4.3. To further improve rasterization efficiency, Sec. 4.4 introduces the compact box (CB) we use, which is adapted from the precise tile-intersection strategy proposed in Speedy-Splat [12].

### 4.2. Multi-view Consistent Densification

The vanilla 3DGS [17] densifies Gaussians solely based on the gradient magnitude in the image space, which leads to a large number of redundant Gaussians. Other densification methods [4, 6, 43] also generate millions of Gaussians, leading to inefficiency. We argue that the redundancy arises because these methods fail to rigorously determine from multiple views whether a Gaussian needs densification. As illustrated in Fig. 3b, Taming-3DGS [24] considers multi-view consistency during densification. However, it primarily computes the score based on Gaussian-associated properties (e.g., opacity, scale, depth, and gradient), making it difficult to enforce strict multi-view consistency for a Gaussian. This also leads to redundancy, as visualized on the left of Fig. 3b. Moreover, the computation of its score is complex and relatively inefficient. To address these issues, we propose a new, simple densification strategy VCD based on multi-view consistency. As illustrated in Fig. 3a, it computes the average number of high-error pixels in each Gaussian's 2D footprint across sampled views, where high-error pixels are identified solely from the per-pixel L1 loss between the ground truth and the rendering. As shown on the left of Fig. 3a, VCD achieves comparable rendering quality with fewer Gaussians, thereby greatly avoiding redundancy. We then detail how VCD is implemented.

Given K camera views $V = \{v^j\}_{j=1}^K$, randomly sampled from the training views, together with their corresponding ground-truth images $G = \{g^j\}_{j=1}^K$ and rendered images $R = \{r^j\}_{j=1}^K$. For each view $v^j$, we compute the error between the rendered color $r_{u,v}^j$ and the ground-truth

&lt;page_number&gt;4&lt;/page_number&gt;

---


## Page 5

color $g_{u,v}^j$ at pixel $(u, v)$:
$$
e_{u,v}^j = \frac{1}{C'} \sum_{c'=1}^{C'} |r_{u,v}^{j,c'} - g_{u,v}^{j,c'}|, \quad (6)
$$
where $c' \in \{1, 2, \dots, C'\}$ denotes the color channel. We then construct the loss map $\mathcal{M}^j \in \mathbb{R}^{W \times H}$ from the per-pixel errors:
$$
\mathcal{M}^j = \mathcal{N}(\{e_{u,v}^j\}_{u=0,v=0}^{W-1,H-1}), \quad (7)
$$
where $\mathcal{N}(\cdot)$ denotes a min-max normalization function. A threshold $\tau$ is then applied to $\mathcal{M}^j$ to identify pixel $p_h$ with high reconstruction error, forming a mask:
$$
\mathcal{M}_{\text{mask}}^j = \mathbb{I}(\mathcal{M}^j > \tau), \quad (8)
$$
where pixels $P$ with $\mathcal{M}_{\text{mask}}^j(u, v) = 1$ indicate regions of poor reconstruction quality.

Next, we need to find the Gaussian primitives associated with these high-error pixels. For each 3D Gaussian primitive $\mathcal{G}_i$, we project it onto the 2D image space to obtain its 2D footprint $\Omega_i$. We then use an indicator function $\mathbb{I}(\mathcal{M}_{\text{mask}}^j(p) = 1)$ to determine whether a pixel has high error. We compute an importance score $s_d^i$ for each Gaussian primitive, which accumulates the number of high-error pixels contained in the 2D footprint across all sampled views and then averages the accumulated value:
$$
s_d^i = \frac{1}{K} \sum_{j=1}^K \sum_p^{\Omega_i} \mathbb{I}(\mathcal{M}_{\text{mask}}^j(p) = 1), \quad (9)
$$
where a higher $s_d^i$ indicates that the Gaussian consistently lies in high-error regions across multiple views, thus suggesting it as a candidate for densification. A Gaussian primitive $\mathcal{G}_i$ is selected for densification only when its importance score $s_d^i$ exceeds a threshold $\tau_d$, ensuring that new Gaussians focus on under-reconstructed regions across views. Notably, we can efficiently determine the number of high-error pixels within the 2D footprint directly from the forward pass of the render.

### 4.3. Multi-view Consistent Pruning

The vanilla 3DGS [17] removes Gaussians with low opacity or overly large scale, but cannot effectively address redundancy. Recent pruning strategies [1, 7, 8, 10, 33] similarly fail to eliminate redundancy and can even significantly degrade rendering quality. In all cases, they do not determine Gaussian redundancy based on multi-view consistency. As illustrated in Fig. 3c, Speedy-Splat [12] computes the pruning score by accumulating Gaussian-associated Hessian approximations across all training views. Hence, it leads to degraded rendering quality due to its indirect use of multi-view consistency, as visualized on the right of Fig. 3c. To remove truly redundant Gaussians, we propose a new, simple pruning strategy VCP based on multi-view consistency, as illustrated in Fig. 3a. Similar to VCD, it evaluates the score according to each Gaussian's impact on multi-view reconstruction quality. As shown on the right of Fig. 3a, VCP removes a significant number of redundant Gaussians while preserving rendering quality. We then detail how VCP is implemented.

Specifically, for each view $v^j \in V$, we compute the photometric loss between the rendered image $r^j$ and the corresponding ground-truth image $g^j$:
$$
E_{\text{photo}}^j = (1 - \lambda)L_1^j + \lambda(1 - L_{\text{SSIM}}^j), \quad (10)
$$
where $L_1^j$ and $L_{\text{SSIM}}^j$ denote the mean absolute error and the structural similarity loss over the entire image, respectively. Since the photometric loss provides a reliable indicator of reconstruction fidelity, we incorporate it with Eq. (9) to derive the pruning score for each Gaussian primitive $\mathcal{G}_i$:
$$
s_p^i = \mathcal{N}\left(\sum_{j=1}^K \left(\sum_p^{\Omega_i} \mathbb{I}(\mathcal{M}_{\text{mask}}^j(p) = 1)\right) \cdot E_{\text{photo}}^j\right). \quad (11)
$$
Here, $s_p^i$ can be interpreted as a quantitative measure of the contribution of the Gaussian primitive $\mathcal{G}_i$ to the degradation of the overall rendering quality. A Gaussian primitive $\mathcal{G}_i$ is selected for pruning if its pruning score $s_p^i$ exceeds a predefined threshold $\tau_p$, indicating that it has relatively low contribution to rendering quality across multiple views.

### 4.4. Compact Box

During the preprocessing stage of rasterization, the vanilla 3DGS [17] uses the 3-sigma rule to obtain 2D ellipses, generating many Gaussian-tile pairs that introduce computational redundancy and reduce rendering efficiency. Speedy-Splat [12] partially addresses this with precise tile intersection, yet we observe that some 2D Gaussians still have a negligible impact on pixels in certain tiles. As illustrated in Fig. 4, to further reduce unnecessary pairs, we introduce a compact box (CB), which builds upon and extends Speedy-Splat [12]'s precise tile-intersection strategy.

&lt;img&gt;
(a) 3DGS
(b) Speedy-Splat
(c) Compact Box (ours)
&lt;/img&gt;

| Method | Tile Intersection Strategy | Number of Gaussian-tile Pairs |
| :--- | :--- | :--- |
| (a) 3DGS | 3-sigma rule (2D ellipse) | High (includes many redundant pairs) |
| (b) Speedy-Splat | Precise tile intersection | Reduced (numbered tiles 1-6) |
| (c) Compact Box (ours) | Compact box (CB) | Minimum (only essential tiles) |
Figure 4. **Compact box.** Compared with vanilla 3DGS [17] and Speedy-Splat [12], incorporating CB leads to a reduced number of Gaussian-tile pairs.

&lt;page_number&gt;5&lt;/page_number&gt;

---


## Page 6

&lt;img&gt;
    &lt;img&gt;Figure 5. Qualitative results of Tab. 1. We present qualitative results on the kitchen and stump scenes from Mip-NeRF 360 [2], the playroom scene from Deep Blending [5], and the truck scene from Tanks & Temples [20]. Notably, the rendered results of the kitchen scene under multiple viewpoints demonstrate that our method achieves more consistent details across views.&lt;/img&gt;
&lt;/img&gt;

by pruning Gaussian-tile pairs with minimal contribution based on the Mahalanobis distance from the Gaussian center. This further accelerates rendering while maintaining quality. Details are provided in Sec. 8 of the supplementary material.

### 4.5. Optimization

Same as the vanilla 3DGS [17], we optimize the learnable parameters with respect to the L1 loss over rendered pixel colors, combined with the SSIM term [36] $\mathcal{L}_{\text{SSIM}}$. The total supervision is defined as:

$$
\mathcal{L} = (1 - \lambda)\mathcal{L}_1 + \lambda(1 - \mathcal{L}_{\text{SSIM}}).
\quad (12)
$$

### 5. Experiments

#### 5.1. Experimental Setup

**Datasets.** Same as vanilla 3DGS [17], we conduct experiments on three real-world datasets: Mip-NeRF 360 [2], Deep-Blending [14], and Tanks & Temples [20]. Moreover, we evaluate dynamic scene reconstruction on D-NeRF [29], NeRF-DS [41], and Neu3D [22] datasets. Tanks & Temples [20], LLFF [25], BungeeNeRF [39], and Replica [34] are respectively used for surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and SLAM.

**Metrics.** To evaluate the performance, we report commonly used metrics for novel view rendering quality, including PSNR, SSIM [37], and LPIPS [45]. In addition, training efficiency and model compactness are assessed by reporting the total training time (in minutes), the final number of Gaussians, and the rendering speed (FPS).

**Implementation Details.** All methods, including ours and the other compared approaches, are trained for 30K iterations using the Adam [19] optimizer. For our approach, we set $K = 10$ and $\lambda = 0.2$ in all experiments. In the base setting, densification is performed every 500 iterations until the 15,000th iteration, and pruning is executed every 500 iterations before 15K and every 3,000 iterations afterwards. To ensure fairness, all experiments are conducted with an NVIDIA RTX 4090 GPU, and all comparison methods are implemented using their official code. The default configu-

&lt;page_number&gt;6&lt;/page_number&gt;

---


## Page 7

Table 1. **Quantitative comparisons with existing 3DGS fast optimization methods.** With FastGS, the training of 3DGS can be completed in around **100 seconds**, while achieving comparable rendering quality to the other methods. Best results are marked as **best score**, second best score, and third best score. Time is reported in minutes.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="5">Mip-NeRF 360 [2]</th>
      <th colspan="5">Deep Blending [14]</th>
      <th colspan="5">Tanks & Temples [20]</th>
    </tr>
    <tr>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓ FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS [17]</td>
      <td>20.93</td>
      <td>27.53</td>
      <td>0.812</td>
      <td>0.221</td>
      <td>2.63M</td>
      <td>146</td>
      <td>19.77</td>
      <td>29.71</td>
      <td>0.903</td>
      <td>0.241</td>
      <td>2.46M</td>
      <td>158</td>
      <td>11.34</td>
      <td>23.71</td>
      <td>0.850</td>
      <td>0.170</td>
      <td>1.57M</td>
      <td>195</td>
    </tr>
    <tr>
      <td>Mini-Splatting [8]</td>
      <td>17.69</td>
      <td>27.32</td>
      <td>0.821</td>
      <td>0.217</td>
      <td>0.53M</td>
      <td>567</td>
      <td>13.35</td>
      <td>29.99</td>
      <td>0.907</td>
      <td>0.244</td>
      <td>0.56M</td>
      <td>624</td>
      <td>9.06</td>
      <td>23.46</td>
      <td>0.844</td>
      <td>0.181</td>
      <td>0.30M</td>
      <td>756</td>
    </tr>
    <tr>
      <td>Speedy-splat [12]</td>
      <td>13.38</td>
      <td>26.91</td>
      <td>0.781</td>
      <td>0.295</td>
      <td>0.30M</td>
      <td>552</td>
      <td>10.75</td>
      <td>29.42</td>
      <td>0.898</td>
      <td>0.272</td>
      <td>0.25M</td>
      <td>664</td>
      <td>6.32</td>
      <td>23.38</td>
      <td>0.816</td>
      <td>0.242</td>
      <td>0.18M</td>
      <td>691</td>
    </tr>
    <tr>
      <td>Taming-3DGS [24]</td>
      <td>5.36</td>
      <td>27.48</td>
      <td>0.794</td>
      <td>0.261</td>
      <td>0.68M</td>
      <td>221</td>
      <td>3.06</td>
      <td>29.50</td>
      <td>0.894</td>
      <td>0.278</td>
      <td>0.29M</td>
      <td>352</td>
      <td>2.71</td>
      <td>23.89</td>
      <td>0.833</td>
      <td>0.214</td>
      <td>0.32M</td>
      <td>379</td>
    </tr>
    <tr>
      <td>DashGaussian [4]</td>
      <td>6.35</td>
      <td>27.73</td>
      <td>0.817</td>
      <td>0.218</td>
      <td>2.40M</td>
      <td>155</td>
      <td>4.16</td>
      <td>29.65</td>
      <td>0.906</td>
      <td>0.246</td>
      <td>1.94M</td>
      <td>208</td>
      <td>4.28</td>
      <td>24.00</td>
      <td>0.853</td>
      <td>0.178</td>
      <td>1.21M</td>
      <td>240</td>
    </tr>
    <tr>
      <td>FastGS (Ours)</td>
      <td>1.93</td>
      <td>27.56</td>
      <td>0.797</td>
      <td>0.261</td>
      <td>0.40M</td>
      <td>579</td>
      <td>1.28</td>
      <td>30.03</td>
      <td>0.901</td>
      <td>0.270</td>
      <td>0.22M</td>
      <td>714</td>
      <td>1.32</td>
      <td>24.15</td>
      <td>0.839</td>
      <td>0.210</td>
      <td>0.24M</td>
      <td>655</td>
    </tr>
    <tr>
      <td>FastGS-Big (Ours)</td>
      <td>3.58</td>
      <td>27.93</td>
      <td>0.820</td>
      <td>0.216</td>
      <td>1.15M</td>
      <td>469</td>
      <td>2.00</td>
      <td>30.12</td>
      <td>0.907</td>
      <td>0.243</td>
      <td>0.65M</td>
      <td>607</td>
      <td>2.03</td>
      <td>24.39</td>
      <td>0.855</td>
      <td>0.175</td>
      <td>0.54M</td>
      <td>569</td>
    </tr>
  </tbody>
</table>

&lt;img&gt;Visualization results on four representative tasks. We present the rendered results of the coffee-martini, fern, Ignatius, and bilbao scenes from the Neu3D [22], LLFF [25], Tanks & Temples [20], and BungeeNeRF [39] datasets, respectively.&lt;/img&gt;

Figure 6. **Visualization results on four representative tasks.** We present the rendered results of the coffee-martini, fern, Ignatius, and bilbao scenes from the Neu3D [22], LLFF [25], Tanks & Temples [20], and BungeeNeRF [39] datasets, respectively.

Table 2. **Quantitative results of dynamic scene reconstruction.** Our method achieves an average 2.84× training speed-up.

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Method</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓ FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NeRF-DS [41]</td>
      <td>Deformable-3DGS [42]</td>
      <td>7.86</td>
      <td>23.83</td>
      <td>0.851</td>
      <td>0.180</td>
      <td>0.13M</td>
      <td>103</td>
    </tr>
    <tr>
      <td></td>
      <td>+Ours</td>
      <td>2.75</td>
      <td>23.90</td>
      <td>0.854</td>
      <td>0.176</td>
      <td>0.03M</td>
      <td>484</td>
    </tr>
    <tr>
      <td>Neu3D [22]</td>
      <td>Deformable-3DGS [42]</td>
      <td>26.16</td>
      <td>26.74</td>
      <td>0.888</td>
      <td>0.168</td>
      <td>0.21M</td>
      <td>69</td>
    </tr>
    <tr>
      <td></td>
      <td>+Ours</td>
      <td>9.29</td>
      <td>29.29</td>
      <td>0.908</td>
      <td>0.146</td>
      <td>0.09M</td>
      <td>161</td>
    </tr>
  </tbody>
</table>

Table 3. **Quantitative results of sparse-view reconstruction.** Our method achieves an average 2.56× training speed-up.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th colspan="5">LLFF [25] (9-view)</th>
    </tr>
    <tr>
      <th></th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓ FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DropGaussian [28]</td>
      <td>1.41</td>
      <td>26.13</td>
      <td>0.874</td>
      <td>0.089</td>
      <td>0.42M</td>
      <td>154</td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td>0.55</td>
      <td>26.14</td>
      <td>0.873</td>
      <td>0.095</td>
      <td>0.21M</td>
      <td>189</td>
    </tr>
  </tbody>
</table>

ration of FastGS builds upon 3DGS-accel [17, 24], detailed in Sec. 11, and incorporates the proposed VCD, VCP, and CB. To achieve extreme training acceleration, the rendering quality of our method is not the highest. Therefore, we provide a variant, **FastGS-Big**, which achieves both the highest rendering quality and the fastest training speed, where densification is executed once every 100 iterations. Further details are in Sec. 9.1 of the supplementary material.

**5.2. Comparison with Fast Optimization Methods**

**Baselines.** We report comparisons with the SOTA fast optimization methods, including DashGaussian [4], Mini-Splatting [8], Speedy-Splat [12], and Taming-3DGS [24], together with the vanilla 3DGS [17] for reference. These methods represent complementary approaches to accelerating training from different perspectives.

**Quantitative Results.** As shown in Tab. 1 and Fig. 5, FastGS achieves the fastest training with comparable rendering quality. A scene can be trained in around 100 seconds, and the fastest case takes only 77 seconds. Taming-3DGS [24] applies weak multi-view consistency constraints, resulting in excessive Gaussians and slower training. Similarly, the pruning strategy of Speedy-Splat [12] leads to a significant drop in rendering quality. The current SOTA, DashGaussian [4], achieves high rendering quality. However, its scene optimization still retains several million Gaussians, which limits the training speed. In contrast, our

&lt;page_number&gt;7&lt;/page_number&gt;

---


## Page 8

Table 4. **Quantitative results of surface reconstruction.** Our method achieves a 2-6× training speed-up.

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Method</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓</th>
      <th>FPS↑</th>
      <th>F1↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Tanks & Temples [20]</td>
      <td>PGSR [3]</td>
      <td>32.28</td>
      <td>24.20</td>
      <td><b>0.857</b></td>
      <td><b>0.149</b></td>
      <td>1.56M</td>
      <td>87</td>
      <td><b>0.57</b></td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td><b>15.74</b></td>
      <td><b>24.37</b></td>
      <td>0.845</td>
      <td>0.190</td>
      <td><b>0.40M</b></td>
      <td><b>210</b></td>
      <td>0.55</td>
    </tr>
    <tr>
      <td rowspan="2">Mip-NeRF 360 [2]</td>
      <td>PGSR [3]</td>
      <td>74.70</td>
      <td>27.22</td>
      <td><b>0.832</b></td>
      <td><b>0.183</b></td>
      <td>3.79M</td>
      <td>45</td>
      <td>-</td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td><b>11.68</b></td>
      <td><b>27.23</b></td>
      <td>0.813</td>
      <td>0.240</td>
      <td><b>0.49M</b></td>
      <td><b>202</b></td>
      <td>-</td>
    </tr>
  </tbody>
</table>

Table 5. **Quantitative results of large-scale reconstruction.** Our method achieves an average 2.19× training speed-up.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th colspan="5">BungeeNeRF [39]</th>
    </tr>
    <tr>
      <th></th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓</th>
      <th>FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Octree-GS [31]</td>
      <td>21.18</td>
      <td><b>28.04</b></td>
      <td><b>0.917</b></td>
      <td><b>0.093</b></td>
      <td>0.99M</td>
      <td>141</td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td><b>9.68</b></td>
      <td><b>28.04</b></td>
      <td>0.910</td>
      <td>0.102</td>
      <td><b>0.74M</b></td>
      <td><b>162</b></td>
    </tr>
  </tbody>
</table>

variant FastGS-Big surpasses DashGaussian [4] by more than 0.2 dB in rendering quality, reduces training time by 43.6%, and cuts the number of Gaussians by more than half. These results demonstrate the superiority of our multi-view consistent densification and pruning strategies.

### 5.3. Generality of FastGS

**Baselines.** Deformable-3DGS [42], PGSR [3], DropGaussian [28], OctreeGS [31], and Photo-SLAM [16] are selected as the backbones for dynamic scene reconstruction, surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and SLAM, respectively.

**Enhancing Various Tasks.** We further test several SOTA methods combined with our framework across these tasks. As shown in Tab. 2, Tab. 3, Tab. 4, and Tab. 5, our method improves the training speed of all baselines by 2-6x while preserving rendering quality. We visualize rendered results in Fig. 6, there is no degradation in rendering quality across multiple tasks. This improvement demonstrates the strong generality of our approach. We argue that this benefits from the multi-view consistency, which is fundamental to various reconstruction tasks. More details and results are provided in Sec. 9.2 of the supplementary material.

**Enhancing Backbone.** Our framework is simple, which can be easily applied to other 3DGS backbones with different representation primitives [23], or additional filters [44]. As shown in Tab. 6, our method achieves 3-8× faster training while maintaining the same rendering quality.

### 5.4. Ablation Study

We adopt 3DGS-accel [17, 24] as our baseline, which preserves the vanilla 3DGS [17] pipeline and integrates per-splat parallel backpropagation and accelerated SH optimization from Taming-3DGS [24], along with optimizer scheduling. We then systematically evaluate the contribution of each proposed module based on this baseline. By adding each component individually, we analyze its impact on both reconstruction quality and training efficiency.

**Multi-View Consistent Densification.** We first evaluate the effect of the densification strategy VCD. As shown in Tab. 7, VCD achieves over 2× faster training without any loss in reconstruction quality. This is because our Gaussian addition is guided by stricter multi-view consistency, which prevents the addition of redundant Gaussians. Tab. 7 further validates this by showing that with VCD, the number of Gaussians is reduced by 80%. This ablation study demonstrates the effectiveness of VCD for accelerating training.

**Multi-View Consistent Pruning.** Next, we evaluate the effectiveness of VCP. As shown in Tab. 7, adding VCP shortens the training time by 25% and reduces the number of Gaussians by 26%, without sacrificing rendering quality. This is because our method effectively removes redundant Gaussians while preserving those critical for scene reconstruction. The strict multi-view consistency evaluation for each deleted Gaussian ensures this effectiveness, demonstrating that VCP is highly effective.

**Compact Box.** Finally, we evaluate the effectiveness of compact box. As shown in Tab. 7, adding CB shortens the training time by 14% while achieving comparable rendering quality. This demonstrates that CB can accelerate training without degrading reconstruction quality.

### 5.5. Discussions and Limitations

Our method performs optimally within training from sparse point clouds. However, it faces challenges when applied to the post-training of the popular feed-forward 3DGS. Since the output Gaussians from these methods are very dense, our approach struggles to prune a massive number of points.

Table 6. **Quantitative results of accelerating various backbones.** Our method achieves a 3-8× training speed-up.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th colspan="5">Mip-NeRF 360 [2]</th>
    </tr>
    <tr>
      <th></th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓</th>
      <th>FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mip-Splatting [44]</td>
      <td>26.20</td>
      <td>27.89</td>
      <td><b>0.837</b></td>
      <td><b>0.176</b></td>
      <td>3.98M</td>
      <td>224</td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td><b>2.98</b></td>
      <td><b>27.95</b></td>
      <td>0.828</td>
      <td>0.208</td>
      <td><b>0.83M</b></td>
      <td><b>606</b></td>
    </tr>
    <tr>
      <td>Scaffold-GS [23]</td>
      <td>18.37</td>
      <td><b>27.70</b></td>
      <td><b>0.812</b></td>
      <td>0.226</td>
      <td>0.57M</td>
      <td>194</td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td><b>5.06</b></td>
      <td>27.68</td>
      <td>0.809</td>
      <td><b>0.220</b></td>
      <td><b>0.30M</b></td>
      <td><b>281</b></td>
    </tr>
  </tbody>
</table>

Table 7. **Ablation studies over the proposed methods of FastGS.** Experiments are performed on the Mip-NeRF 360 dataset [2] with 3DGS-accel [17, 24] as the baseline.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓</th>
      <th>FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS-accel</td>
      <td>7.10</td>
      <td>27.46</td>
      <td>0.810</td>
      <td>0.226</td>
      <td>2.64M</td>
      <td>182</td>
    </tr>
    <tr>
      <td>+VCD.</td>
      <td>3.53</td>
      <td>27.69</td>
      <td>0.798</td>
      <td>0.259</td>
      <td>0.53M</td>
      <td>222</td>
    </tr>
    <tr>
      <td>+VCP.</td>
      <td>5.32</td>
      <td><b>27.70</b></td>
      <td><b>0.812</b></td>
      <td>0.228</td>
      <td>1.96M</td>
      <td>285</td>
    </tr>
    <tr>
      <td>+CB.</td>
      <td>6.13</td>
      <td>27.44</td>
      <td>0.810</td>
      <td><b>0.223</b></td>
      <td>2.78M</td>
      <td>303</td>
    </tr>
    <tr>
      <td>Full</td>
      <td><b>1.93</b></td>
      <td>27.56</td>
      <td>0.797</td>
      <td>0.261</td>
      <td><b>0.40M</b></td>
      <td><b>579</b></td>
    </tr>
  </tbody>
</table>

&lt;page_number&gt;8&lt;/page_number&gt;

---


## Page 9

effectively within just a few thousand iterations while maintaining rendering quality, making it difficult to achieve extreme acceleration. In our tests, even a short post-training of 3K iterations still requires approximately 20 seconds.

## 6. Conclusion

This paper presents a novel, simple, and general 3DGS acceleration framework FastGS. We propose multi-view consistent densification and pruning strategies that prevent redundant Gaussians. Extensive experiments demonstrate the effectiveness of our view-consistency design. Our method achieves the fastest training speed among all SOTA methods while maintaining comparable rendering quality. The results also demonstrate the strong generality of FastGS, greatly reducing training time across various tasks.

## References

[1] Muhammad Salman Ali, Maryam Qamar, Sung-Ho Bae, and Enzo Tartaglione. Trimming the fat: Efficient compression of 3d gaussian splats through pruning. *arXiv preprint arXiv:2406.18214*, 2024. 3, 5
[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 5470–5479, 2022. 2, 6, 7, 8, 3, 4, 5
[3] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. *IEEE Transactions on Visualization and Computer Graphics*, 2024. 8, 2
[4] Youyu Chen, Junjun Jiang, Kui Jiang, Xiao Tang, Zhihao Li, Xianming Liu, and Yinyu Nie. Dashgaussian: Optimizing 3d gaussian splatting in 200 seconds. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 11146–11155, 2025. 2, 3, 4, 7, 8, 5
[5] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised nerf: Fewer views and faster training for free. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 12882–12891, 2022. 6, 5
[6] Xiaobin Deng, Changyu Diao, Min Li, Ruohan Yu, and Duanqing Xu. Efficient density control for 3d gaussian splatting. *arXiv preprint arXiv:2411.10133*, 2024. 2, 3, 4
[7] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. *Advances in neural information processing systems*, 37: 140138–140158, 2024. 5
[8] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In *European Conference on Computer Vision*, pages 165–181. Springer, 2024. 2, 3, 5, 7
[9] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Boni Hu, Linning Xu, Zhilin Pei, Hengjie Li, et al. Flashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 26652–26662, 2025. 2, 3
[10] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. Eagles: Efficient accelerated 3d gaussians with lightweight encodings. In *European Conference on Computer Vision*, pages 54–71. Springer, 2024. 2, 3, 5
[11] Hao Gui, Lin Hu, Rui Chen, Mingxiao Huang, Yuxin Yin, Jin Yang, Yong Wu, Chen Liu, Zhongxu Sun, Xueyang Zhang, et al. Balanced 3dgs: Gaussian-wise parallelism rendering with fine-grained tiling. *arXiv preprint arXiv:2412.17378*, 2024. 2
[12] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein. Speedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse primitives. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 21537–21546, 2025. 2, 3, 4, 5, 7, 1
[13] Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayawardhana, Matthias Zwicker, and Tom Goldstein. Pup 3d-gs: Principled uncertainty pruning for 3d gaussian splatting. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 5949–5958, 2025. 2, 3
[14] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow. Deep blending for free-viewpoint image-based rendering. *ACM Transactions on Graphics (ToG)*, 37(6):1–15, 2018. 6, 7, 3
[15] Lukas Höllein, Aljaž Božič, Michael Zollhöfer, and Matthias Nießner. 3dgs-lm: Faster gaussian-splatting optimization with levenberg-marquardt. *arXiv preprint arXiv:2409.12892*, 2024. 3
[16] Huajian Huang, Longwei Li, Cheng Hui, and Sai-Kit Yeung. Photo-slam: Real-time simultaneous localization and photorealistic mapping for monocular, stereo, and rgb-d cameras. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024. 8, 2
[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42 (4), 2023. 1, 2, 3, 4, 5, 6, 7, 8
[18] Sieun Kim, Kyungjin Lee, and Youngki Lee. Color-cued efficient densification method for 3d gaussian splatting. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 775–783, 2024. 2, 3
[19] Diederik P Kingma. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014. 6, 2, 3
[20] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. *ACM Transactions on Graphics (ToG)*, 36 (4):1–13, 2017. 6, 7, 8, 3, 4, 5
[21] Lei Lan, Tianjia Shao, Zixuan Lu, Yu Zhang, Chenfanfu Jiang, and Yin Yang. 3dgs2: Near second-order converging 3d gaussian splatting. In *Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers*, pages 1–10, 2025. 3

&lt;page_number&gt;9&lt;/page_number&gt;

---


## Page 10

[22] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 5521–5531, 2022. 6, 7
[23] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 20654–20664, 2024. 8, 3
[24] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3dgs: High-quality radiance fields with limited resources. In *SIGGRAPH Asia 2024 Conference Papers*, pages 1–11, 2024. 2, 3, 4, 7, 8, 1, 5
[25] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. *ACM Transactions on Graphics (ToG)*, 38(4):1–14, 2019. 6, 7, 2
[26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. *Communications of the ACM*, 65(1):99–106, 2021. 1
[27] Panagiotis Papantonakis, Georgios Kopanas, Bernhard Kerbl, Alexandre Lanvin, and George Drettakis. Reducing the memory footprint of 3d gaussian splatting. *Proceedings of the ACM on Computer Graphics and Interactive Techniques*, 7(1):1–17, 2024. 2, 3
[28] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian: Structural regularization for sparse-view gaussian splatting. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 21600–21609, 2025. 7, 8, 2
[29] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10318–10327, 2021. 6
[30] Lukas Radl, Michael Steiner, Mathias Parger, Alexander Weinrauch, Bernhard Kerbl, and Markus Steinberger. Stopthepop: Sorted gaussian splatting for view-consistent real-time rendering. *ACM Transactions on Graphics (TOG)*, 43(4):1–17, 2024. 3
[31] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians. *arXiv preprint arXiv:2403.17898*, 2024. 8, 2
[32] Samuel Rota Bulò, Lorenzo Porzi, and Peter Kontschieder. Revising densification in gaussian splatting. In *European Conference on Computer Vision*, pages 347–362. Springer, 2024. 3
[33] Muhammad Salman Ali, Sung-Ho Bae, and Enzo Tartaglione. Elmgs: Enhancing memory and computation scalability through compression for 3d gaussian splatting. *arXiv e-prints*, pages arXiv-2410, 2024. 2, 3, 5
[34] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. The Replica dataset: A digital replica of indoor spaces. *arXiv preprint arXiv:1906.05797*, 2019. 6
[35] Xinzhe Wang, Ran Yi, and Lizhuang Ma. Adr-gaussian: Accelerating gaussian splatting with adaptive radius. In *SIGGRAPH Asia 2024 Conference Papers*, pages 1–10, 2024. 2
[36] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4):600–612, 2004. 6
[37] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. *IEEE transactions on image processing*, 13(4):600–612, 2004. 6
[38] Zirui Wu, Tianyu Liu, Liyi Luo, Zhide Zhong, Jianteng Chen, Hongmin Xiao, Chao Hou, Haozhe Lou, Yuantao Chen, Runyi Yang, et al. Mars: An instance-aware, modular and realistic simulator for autonomous driving. In *CAAI International Conference on Artificial Intelligence*, pages 3–15. Springer, 2023. 1
[39] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, and Dahua Lin. Bungeenerf: Progressive neural radiance field for extreme multi-scale scene rendering. In *European conference on computer vision*, pages 106–122. Springer, 2022. 6, 7, 8
[40] Linning Xu, Vasu Agrawal, William Laney, Tony Garcia, Aayush Bansal, Changil Kim, Samuel Rota Bulò, Lorenzo Porzi, Peter Kontschieder, Aljaž Božič, et al. Vr-nerf: High-fidelity virtualized walkable spaces. In *SIGGRAPH Asia 2023 Conference Papers*, pages 1–12, 2023. 1
[41] Zhiwen Yan, Chen Li, and Gim Hee Lee. Nerf-ds: Neural radiance fields for dynamic specular objects. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8285–8295, 2023. 6, 7
[42] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 20331–20341, 2024. 7, 8, 2
[43] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong Dou. Absgs: Recovering fine details in 3d gaussian splatting. In *Proceedings of the 32nd ACM International Conference on Multimedia*, pages 1053–1061, 2024. 4, 1, 2, 3
[44] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 19447–19456, 2024. 8, 2, 3
[45] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of

&lt;page_number&gt;10&lt;/page_number&gt;

---


## Page 11

deep features as a perceptual metric. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 586–595, 2018. 6
[46] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. *arXiv preprint arXiv:1805.09817*, 2018. 1

&lt;page_number&gt;11&lt;/page_number&gt;

---


## Page 12

FastGS: Training 3D Gaussian Splatting in 100 Seconds
Supplementary Material

## 7. Overview
The supplementary material provides the following contents: Sec. 8 gives a detailed description of the proposed compact box. Sec. 9 presents additional experimental details, where Sec. 9.1 provides implementation details, and Secs. 9.2 and 9.3 describe the execution details of integrating FastGS into different tasks and backbones. Sec. 10 reports the computational overhead, Sec. 11 includes additional ablations, and Sec. 12 provides scene-wise results.

## 8. Details of Compact Box
During the preprocessing stage of rasterization, vanilla 3DGS [17] employs the 3-sigma rule to coarsely obtain effective 2D ellipses, which results in a large number of Gaussian-tile pairs as shown in Fig. 7a, introducing computational redundancy and significantly reducing rendering efficiency. To address this issue, Speedy-Splat [12] proposes a precise tile-intersection method to reduce the number of Gaussian-tile pairs. However, according to our observations, there remains room for further improvement, as some 2D Gaussians still have a negligible impact on the pixels within the numbered tiles in Fig. 7b. Therefore, we propose Compact Box, which builds upon and refines Speedy-Splat's precise tile-intersection strategy to further reduce the effective 2D Gaussian region and eliminate unnecessary Gaussian-tile pairs, as illustrated in Fig. 7c.

Formally, during the $\alpha$-blending process, the alpha value of the $i$-th Gaussian at pixel $p$ is defined as:
$$
\alpha_i(p) = \sigma_i \cdot \exp\left(-\frac{1}{2}(p - \mu_{i2D})\Sigma_{i2D}^{-1}(p - \mu_{i2D})^T\right). \quad (13)
$$
This equation implies that $\alpha_i(p)$ decays exponentially with the Mahalanobis distance:
$$
A(p) = (p - \mu_{i2D})\Sigma_{i2D}^{-1}(p - \mu_{i2D})^T. \quad (14)
$$
Intuitively, pixels closer to the Gaussian center contribute more, while those farther away have negligible influence. Based on this observation, a reasonable criterion can be established: Gaussian-tile pairs corresponding to pixels with large Mahalanobis distances can be safely discarded, as their effect on the final rendering is minimal.

To prune Gaussian-tile pairs whose contributions are negligible, we define a threshold for the Mahalanobis distance as:
$$
(p - \mu_{i2D})\Sigma_{i2D}^{-1}(p - \mu_{i2D})^T = \beta\left(2\ln\frac{\sigma_i}{\tau_\alpha}\right), \quad (15)
$$
where $\beta$ is a scaling factor. By adjusting $\beta$, the effective support region of each 2D Gaussian can be flexibly controlled. A smaller $\beta$ yields a tighter ellipse around the mean $\mu_{i2D}$, thereby reducing the spatial extent of $(p - \mu_{i2D})$ and limiting the number of pixels influenced by the Gaussian $G_j$. This selective suppression of marginal Gaussian contributions effectively reduces redundant Gaussian-tile pairs and accelerates rasterization.

In implementation, Eq. (15) is integrated into Speedy-Splat [12]'s snugbox, where the parameter $\beta$ further reduces the 2D Gaussian footprint and shrinks its intersection region with tiles, thus forming our CB. We then obtain the Gaussian-tile pairs using Speedy-Splat [12]'s accutile method. The comparison between snugbox [12] and our CB can be found in Tab. 16. We sincerely appreciate the excellent work of Speedy-Splat [12].

&lt;img&gt;
(a) 3DGS
(b) Speedy-Splat
(c) Compact Box (ours)
&lt;/img&gt;

| Method | Tile Intersection Strategy | Visual Representation |
| :--- | :--- | :--- |
| (a) 3DGS | 3-sigma rule | Large number of Gaussian-tile pairs with computational redundancy |
| (b) Speedy-Splat | Precise tile-intersection | Reduced Gaussian-tile pairs, but some tiles (1-6) have negligible impact |
| (c) Compact Box (ours) | Refined precise tile-intersection | Further reduced effective 2D Gaussian region and eliminated unnecessary pairs |
Figure 7. **Compact box.** Compared with vanilla 3DGS [17] and Speedy-Splat [12], incorporating CB leads to a reduced number of Gaussian-tile pairs.

## 9. More Details
### 9.1. Implementation Details
Our FastGS integrate VCD, VCP, and CB into 3DGS-accel [17, 24] and adopt the widely used absolute gradients [43] to ensure more accurate densification, ultimately achieving an average training time of around 100 seconds per scene. The core of our method lies in strictly controlling the number of Gaussians throughout training via VCD and VCP, maintaining it at a very low level and thereby enabling significant acceleration, as shown in Fig. 2.

In practice, our VCD and VCP introduce a multi-view consistency constraint that strictly regulates Gaussian densification and pruning, preventing the creation of redundant Gaussians. For densification, newly added Gaussians are required not only to satisfy the gradient-based criteria but also to have an importance score greater than $\tau_d$, which is set to 5 in our experiments. We follow the vanilla 3DGS [17] gradient for cloning, while adopting the absolute gradient

&lt;page_number&gt;1&lt;/page_number&gt;

---


## Page 13

Table 8. **Quantitative results of sparse-view reconstruction.** We present the results under the 3-view and 6-view settings.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">LLFF [25] 3-view</th>
      <th colspan="4">LLFF [25] 6-view</th>
    </tr>
    <tr>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>NGS ↓</th>
      <th>FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>NGS ↓</th>
      <th>FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DropGaussian [28]</td>
      <td>0.73</td>
      <td>20.43</td>
      <td>0.707</td>
      <td>0.202</td>
      <td>0.08M</td>
      <td>183</td>
      <td>0.88</td>
      <td>24.67</td>
      <td>0.836</td>
      <td>0.116</td>
      <td>0.18M</td>
      <td>174</td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td>0.30</td>
      <td>20.58</td>
      <td>0.708</td>
      <td>0.217</td>
      <td>0.03M</td>
      <td>206</td>
      <td>0.37</td>
      <td>24.68</td>
      <td>0.834</td>
      <td>0.131</td>
      <td>0.09M</td>
      <td>199</td>
    </tr>
  </tbody>
</table>

from AbsGS [43] for splitting. For pruning, before 15k iterations, we follow 3DGS [17] but retain VCP by sampling half of the candidate Gaussians according to their pruning scores. After 15k iterations, pruning is performed every 3k iterations by removing Gaussians with opacity below 0.1 or pruning score exceeding 0.9, ensuring multi-view consistency.

Our baseline, 3DGS-accel [17, 24], preserves the vanilla 3DGS [17] pipeline while integrating the per-splat parallel backpropagation and accelerated SH optimization from Taming-3DGS [24], along with an optimizer schedule that updates the optimizer every 32 iterations from 15,000 to 20,000 iterations and every 64 iterations thereafter. This schedule is inspired by the SH optimization strategy in Taming-3DGS [24]. We do not consider it a conceptual contribution of our method, so instead of discussing it in the main paper, we incorporate it directly into the baseline configuration. As shown in Tab. 15, this scheduling strategy provides acceleration comparable to sparse Adam [24] while fully preserving rendering quality.

We sincerely thank Taming-3DGS [24] for providing a strong baseline, upon which our work builds. By further integrating the proposed VCD, VCP, and CB components, together with the absolute gradients from AbsGS [43], our method achieves significant acceleration. This improvement largely stems from the strict control of Gaussian count enforced by VCD and VCP, ensuring that the number of Gaussians remains as low as possible throughout training. Other modules contribute only marginally to this aspect, which we further analyze in the ablation study.

**9.2. Generalizing FastGS to Other Tasks**

**Dynamic Scene Reconstruction:** Deformable-3DGS [42] adopts the same ADC strategy as vanilla 3DGS [17], while additionally predicting per-Gaussian deformation parameters. This design remains fully compatible with our framework. Building on our accelerated 3DGS backbone, we employ VCD and VCP to precisely regulate densification and pruning, ensuring that the number of Gaussians remains low throughout training. CB is further integrated to speed up rendering, and the optimizer is Adam [19].

**Surface Reconstruction:** PGSR [3] uses an ADC strategy similar to vanilla 3DGS [17]. Based on our accelerated 3DGS backbone, we replace ADC with VCD and VCP to strictly control Gaussian growth and elimination. CB is

Table 9. **Quantitative results of SLAM.** Our method achieves an average 2.70× training speed-up.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="6">Replica RGB-D</th>
    </tr>
    <tr>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>NGS ↓</th>
      <th>FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Photo-SLAM [16]</td>
      <td>5.03</td>
      <td>37.01</td>
      <td>0.961</td>
      <td>0.026</td>
      <td>0.33M</td>
      <td>744</td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td>1.86</td>
      <td>37.01</td>
      <td>0.957</td>
      <td>0.042</td>
      <td>0.11M</td>
      <td>2700</td>
    </tr>
  </tbody>
</table>

integrated into the rasterization stage to further improve efficiency. Adam [19] is used as the optimizer.

**Sparse-view Reconstruction:** DropGaussian [28] differs from vanilla 3DGS [17] in that it randomly sets the opacity of a subset of Gaussians to zero during rendering, eliminating their contribution. Based on our accelerated 3DGS backbone, we apply VCD and VCP to precisely control densification and pruning, keeping the number of Gaussians low throughout training. CB is incorporated to accelerate rendering, and Adam [19] is used as the optimizer. Additional experimental results are presented in Tab. 8.

**Large-scale Reconstruction:** Octree-GS [31] uses an anchor-based parameterization where each anchor generates multiple Gaussians. Based on our accelerated 3DGS backbone, we apply VCD to constrain anchor expansion, requiring the importance score of associated Gaussians to exceed 5. VCP is not applied since pruning is performed at the anchor level. CB is integrated into rasterization. The optimizer is Adam [19].

**SLAM:** Photo-SLAM [16] follows vanilla 3DGS [17]'s ADC strategy. Based on our accelerated 3DGS backbone, we integrate VCD and VCP for effective densification and pruning, and incorporate CB to speed up rendering. Adam [19] is used as the optimizer. Results are presented in Tab. 9.

**9.3. Equipping FastGS to Backbones**

**Mip-Splatting [44]:** Mip-Splatting [44] introduces a filtering mechanism for anti-aliasing while following an ADC strategy similar to vanilla 3DGS [17]. Based on our accelerated 3DGS backbone, we replace its ADC pipeline with VCD and VCP, which precisely control Gaussian densification and pruning to keep the number of Gaussians low throughout training. CB is also integrated into the rasterization stage to further accelerate rendering. Adam [19] is used as the optimizer. Additional experimental results are

&lt;page_number&gt;2&lt;/page_number&gt;

---


## Page 14

Table 10. Quantitative results of accelerating various backbones.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">Deep Blending [14]</th>
      <th colspan="4">Tanks & Temples [20]</th>
    </tr>
    <tr>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>NGS ↓</th>
      <th>FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>NGS ↓</th>
      <th>FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mip-Splatting [44]</td>
      <td>23.94</td>
      <td>29.35</td>
      <td>0.899</td>
      <td>0.241</td>
      <td>3.48M</td>
      <td>219</td>
      <td>14.57</td>
      <td>23.77</td>
      <td>0.856</td>
      <td>0.158</td>
      <td>2.36M</td>
      <td>300</td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td>1.74</td>
      <td>29.68</td>
      <td>0.899</td>
      <td>0.274</td>
      <td>0.20M</td>
      <td>698</td>
      <td>2.01</td>
      <td>24.18</td>
      <td>0.843</td>
      <td>0.200</td>
      <td>0.36M</td>
      <td>729</td>
    </tr>
    <tr>
      <td>Scaffold-GS [23]</td>
      <td>13.31</td>
      <td>30.09</td>
      <td>0.905</td>
      <td>0.256</td>
      <td>0.18M</td>
      <td>307</td>
      <td>9.83</td>
      <td>24.09</td>
      <td>0.851</td>
      <td>0.175</td>
      <td>0.26M</td>
      <td>261</td>
    </tr>
    <tr>
      <td>+Ours</td>
      <td>2.82</td>
      <td>30.00</td>
      <td>0.900</td>
      <td>0.267</td>
      <td>0.08M</td>
      <td>423</td>
      <td>3.77</td>
      <td>24.15</td>
      <td>0.849</td>
      <td>0.180</td>
      <td>0.14M</td>
      <td>332</td>
    </tr>
  </tbody>
</table>

Table 11. Quantitative comparison of computational overhead. We report the mean GPU memory usage (GB), peak GPU memory usage (GB), and storage size (MB).

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="3">MipNeRF-360 [2]</th>
      <th colspan="3">Deep Blending [14]</th>
      <th colspan="3">Tanks&Temples [20]</th>
    </tr>
    <tr>
      <th>mean GPU mem.↓</th>
      <th>peak GPU mem.↓</th>
      <th>Storage↓</th>
      <th>mean GPU mem.↓</th>
      <th>peak GPU mem.↓</th>
      <th>Storage↓</th>
      <th>mean GPU mem.↓</th>
      <th>peak GPU mem.↓</th>
      <th>Storage↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS [17]</td>
      <td>7.70</td>
      <td>9.89</td>
      <td>652</td>
      <td>5.97</td>
      <td>8.10</td>
      <td>610</td>
      <td>3.47</td>
      <td>4.73</td>
      <td>389</td>
    </tr>
    <tr>
      <td>Mini-Splatting [8]</td>
      <td>5.21</td>
      <td>7.44</td>
      <td>132</td>
      <td>4.16</td>
      <td>6.20</td>
      <td>138</td>
      <td>2.51</td>
      <td>4.63</td>
      <td>75</td>
    </tr>
    <tr>
      <td>Speedy-splat [12]</td>
      <td>4.97</td>
      <td>7.03</td>
      <td>74</td>
      <td>3.82</td>
      <td>5.03</td>
      <td>61</td>
      <td>2.19</td>
      <td>2.66</td>
      <td>45</td>
    </tr>
    <tr>
      <td>Taming-3DGS [24]</td>
      <td>4.78</td>
      <td>5.88</td>
      <td>170</td>
      <td>3.39</td>
      <td>4.01</td>
      <td>73</td>
      <td>1.94</td>
      <td>2.43</td>
      <td>79</td>
    </tr>
    <tr>
      <td>DashGaussian [4]</td>
      <td>6.71</td>
      <td>9.96</td>
      <td>595</td>
      <td>4.79</td>
      <td>7.75</td>
      <td>482</td>
      <td>2.81</td>
      <td>4.49</td>
      <td>301</td>
    </tr>
    <tr>
      <td>FastGS (Ours)</td>
      <td>4.58</td>
      <td>5.21</td>
      <td>99</td>
      <td>3.34</td>
      <td>3.77</td>
      <td>54</td>
      <td>1.91</td>
      <td>2.27</td>
      <td>60</td>
    </tr>
    <tr>
      <td>FastGS-Big (Ours)</td>
      <td>5.37</td>
      <td>6.63</td>
      <td>208</td>
      <td>3.81</td>
      <td>4.65</td>
      <td>114</td>
      <td>2.22</td>
      <td>2.83</td>
      <td>86</td>
    </tr>
  </tbody>
</table>

Table 12. Ablation studies over the proposed methods. Experiments are performed on the Mip-NeRF 360 dataset [2] with 3DGS-accel [17, 24] as the baseline.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>NGS ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS-accel</td>
      <td>7.10</td>
      <td>27.46</td>
      <td>0.810</td>
      <td>0.226</td>
      <td>2.64M</td>
    </tr>
    <tr>
      <td>+Abs grad</td>
      <td>6.85</td>
      <td>27.60</td>
      <td>0.817</td>
      <td>0.216</td>
      <td>2.29M</td>
    </tr>
    <tr>
      <td>+CB.</td>
      <td>6.13</td>
      <td>27.44</td>
      <td>0.810</td>
      <td>0.223</td>
      <td>2.78M</td>
    </tr>
    <tr>
      <td>+VCD.</td>
      <td>3.53</td>
      <td>27.69</td>
      <td>0.798</td>
      <td>0.259</td>
      <td>0.53M</td>
    </tr>
    <tr>
      <td>+VCP.</td>
      <td>5.32</td>
      <td>27.70</td>
      <td>0.812</td>
      <td>0.228</td>
      <td>1.96M</td>
    </tr>
    <tr>
      <td>Full</td>
      <td>1.93</td>
      <td>27.56</td>
      <td>0.797</td>
      <td>0.261</td>
      <td>0.40M</td>
    </tr>
  </tbody>
</table>

presented in Tab. 10.
**Scaffold-GS [23]:** Scaffold-GS [23] adopts an anchor-based Gaussian representation. On our accelerated 3DGS backbone, we apply VCD to control densification and maintain a low number of Gaussians. VCP is not applicable because pruning is performed at the anchor level. CB is integrated into the rasterization stage to further accelerate rendering. Adam [19] is used as the optimizer. Additional experimental results are presented in Tab. 10.

## 10. Computational Overhead

We report computational resource consumption in Tab. 11. As shown, our method requires relatively low GPU memory, making it suitable for devices with limited resources.

## 11. Additional Ablation

In this section, we perform more comprehensive ablations based on 3DGS-accel [17, 24] to further demonstrate that the proposed multi-view consistency-based densification and pruning strategies, VCD and VCP, contribute most significantly to the overall acceleration.
**Component-wise Ablation.** We examine the effects of VCD, VCP, CB, and the absolute gradients from Ab-sGS [43] in Tab. 12. The ablation results indicate that neither absolute gradients nor CB effectively reduce the number of Gaussians, and their contribution to acceleration is limited. In contrast, our proposed VCD and VCP achieve significantly greater speed-up, as they strictly control the Gaussian count, keeping it low throughout the entire training process, as shown in Fig. 2.
**VCD Threshold $\tau_d$.** We study the effect of the densification threshold $\tau_d$ in VCD on Tanks & Temples [20], as shown in Tab. 13. A smaller $\tau_d$ allows more Gaussians to be densified, leading to slightly higher rendering quality but at the cost of increased training time and Gaussian count. Conversely, a larger $\tau_d$ reduces the number of Gaussians, accelerating training while slightly degrading quality. Our default choice of $\tau_d = 5$ achieves a balanced trade-off between efficiency and rendering fidelity.
**Number of sampled views K.** We study the effect of the number of sampled views on both training efficiency and rendering quality on the Mip-NeRF 360 [2] dataset. As shown in Tab. 14, using too few views slightly degrades quality, while sampling more views increases training time with minimal improvement. Our default choice of $K = 10$ achieves a good balance.

## 12. Scene-wise Results

We present the quantitative results in Tab. 17, Tab. 18, and Tab. 19, and provide the qualitative comparisons in Fig. 8,

&lt;page_number&gt;3&lt;/page_number&gt;

---


## Page 15

Table 13. Ablation study on thresh $\tau_d$ on Tanks & Temples [20].

<table>
  <thead>
    <tr>
      <th>$\tau_d$</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1.44</td>
      <td>24.17</td>
      <td><b>0.841</b></td>
      <td><b>0.205</b></td>
      <td>0.30M</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.42</td>
      <td><b>24.18</b></td>
      <td><b>0.841</b></td>
      <td>0.207</td>
      <td>0.28M</td>
    </tr>
    <tr>
      <td>5(ours)</td>
      <td>1.32</td>
      <td>24.15</td>
      <td>0.839</td>
      <td>0.210</td>
      <td>0.24M</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1.30</td>
      <td>23.98</td>
      <td>0.834</td>
      <td>0.218</td>
      <td>0.21M</td>
    </tr>
    <tr>
      <td>20</td>
      <td>1.23</td>
      <td>23.84</td>
      <td>0.829</td>
      <td>0.226</td>
      <td>0.17M</td>
    </tr>
    <tr>
      <td>50</td>
      <td>1.15</td>
      <td>23.54</td>
      <td>0.819</td>
      <td>0.241</td>
      <td>0.13M</td>
    </tr>
    <tr>
      <td>100</td>
      <td><b>1.12</b></td>
      <td>23.26</td>
      <td>0.809</td>
      <td>0.252</td>
      <td><b>0.11M</b></td>
    </tr>
  </tbody>
</table>

Table 14. Ablation study on the number of sampled views $K$ on Mip-NeRF 360 [2]. “all” indicates using all training views.

<table>
  <thead>
    <tr>
      <th>$K$</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td><b>1.91</b></td>
      <td>27.47</td>
      <td>0.795</td>
      <td>0.265</td>
      <td><b>0.36M</b></td>
    </tr>
    <tr>
      <td>10(ours)</td>
      <td>1.93</td>
      <td><b>27.56</b></td>
      <td><b>0.797</b></td>
      <td><b>0.261</b></td>
      <td>0.40M</td>
    </tr>
    <tr>
      <td>20</td>
      <td>2.03</td>
      <td>27.55</td>
      <td><b>0.797</b></td>
      <td><b>0.261</b></td>
      <td>0.43M</td>
    </tr>
    <tr>
      <td>50</td>
      <td>2.10</td>
      <td>27.55</td>
      <td><b>0.797</b></td>
      <td>0.262</td>
      <td>0.43M</td>
    </tr>
    <tr>
      <td>all</td>
      <td>2.29</td>
      <td>27.54</td>
      <td>0.796</td>
      <td>0.263</td>
      <td>0.42M</td>
    </tr>
  </tbody>
</table>

Table 15. Comparison of different optimization strategies.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Optimizer</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FastGS</td>
      <td>Sparse Adam [24]</td>
      <td><b>1.93</b></td>
      <td>27.37</td>
      <td>0.792</td>
      <td>0.270</td>
      <td><b>0.37M</b></td>
    </tr>
    <tr>
      <td></td>
      <td>Optimizer schedule</td>
      <td><b>1.93</b></td>
      <td><b>27.56</b></td>
      <td><b>0.797</b></td>
      <td><b>0.261</b></td>
      <td>0.40M</td>
    </tr>
  </tbody>
</table>

Table 16. Comparison of rasterization methods. Experiments are performed on the Mip-NeRF 360 dataset [2] with 3DGS-accel [17, 24] as the baseline.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓</th>
      <th>N<sub>GS</sub> ↓</th>
      <th>FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS-accel</td>
      <td>7.10</td>
      <td><b>27.46</b></td>
      <td>0.810</td>
      <td>0.226</td>
      <td><b>2.64M</b></td>
      <td>182</td>
    </tr>
    <tr>
      <td>+snugbox [12]</td>
      <td>6.50</td>
      <td><b>27.46</b></td>
      <td><b>0.811</b></td>
      <td>0.224</td>
      <td>2.69M</td>
      <td>288</td>
    </tr>
    <tr>
      <td>+CB.</td>
      <td><b>6.13</b></td>
      <td>27.44</td>
      <td>0.810</td>
      <td><b>0.223</b></td>
      <td>2.78M</td>
      <td><b>303</b></td>
    </tr>
  </tbody>
</table>

Fig. 9 and Fig. 10.

&lt;page_number&gt;4&lt;/page_number&gt;

---


## Page 16

Table 17. Scene-wise quantitative results over the Mip-NeRF 360 dataset [2].

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">bicycle</th>
      <th colspan="4">flowers</th>
      <th colspan="4">garden</th>
    </tr>
    <tr>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS [17]</td>
      <td>27.97</td>
      <td>25.14</td>
      <td>0.748</td>
      <td>0.242 4.71M 80</td>
      <td>18.98</td>
      <td>21.30</td>
      <td>0.586</td>
      <td>0.360 2.82M 162</td>
      <td>26.78</td>
      <td>27.34</td>
      <td>0.857</td>
      <td>0.122 4.19M 103</td>
    </tr>
    <tr>
      <td>Mini-Splatting [8]</td>
      <td>16.17</td>
      <td>25.23</td>
      <td>0.764</td>
      <td>0.241 0.59M 564</td>
      <td>17.22</td>
      <td>21.43</td>
      <td>0.614</td>
      <td>0.341 0.63M 511</td>
      <td>15.97</td>
      <td>27.36</td>
      <td>0.806</td>
      <td>0.215 0.67M 487</td>
    </tr>
    <tr>
      <td>Speedy-Splat [12]</td>
      <td>15.87</td>
      <td>24.79</td>
      <td>0.704</td>
      <td>0.333 0.58M 460</td>
      <td>13.38</td>
      <td>21.21</td>
      <td>0.560</td>
      <td>0.418 0.34M 526</td>
      <td>15.73</td>
      <td>26.69</td>
      <td>0.814</td>
      <td>0.214 0.52M 474</td>
    </tr>
    <tr>
      <td>Taming-3DGS [24]</td>
      <td>5.65</td>
      <td>24.72</td>
      <td>0.693</td>
      <td>0.332 0.81M 199</td>
      <td>4.97</td>
      <td>21.10</td>
      <td>0.552</td>
      <td>0.416 0.58M 233</td>
      <td>9.82</td>
      <td>27.42</td>
      <td>0.851</td>
      <td>0.138 2.08M 177</td>
    </tr>
    <tr>
      <td>DashGaussian [4]</td>
      <td>9.93</td>
      <td>25.31</td>
      <td>0.763</td>
      <td>0.222 4.70M 105</td>
      <td>7.05</td>
      <td>21.78</td>
      <td>0.604</td>
      <td>0.341 2.82M 158</td>
      <td>8.27</td>
      <td>27.57</td>
      <td>0.857</td>
      <td>0.131 3.37M 153</td>
    </tr>
    <tr>
      <td>FastGS</td>
      <td>1.92</td>
      <td>24.84</td>
      <td>0.714</td>
      <td>0.310 0.54M 582</td>
      <td>1.95</td>
      <td>21.21</td>
      <td>0.560</td>
      <td>0.406 0.49M 555</td>
      <td>2.47</td>
      <td>27.20</td>
      <td>0.836</td>
      <td>0.174 0.74M 538</td>
    </tr>
    <tr>
      <td>FastGS-Big</td>
      <td>2.59</td>
      <td>25.26</td>
      <td>0.755</td>
      <td>0.245 1.55M 463</td>
      <td>3.22</td>
      <td>21.60</td>
      <td>0.602</td>
      <td>0.341 1.14M 468</td>
      <td>6.50</td>
      <td>27.56</td>
      <td>0.864</td>
      <td>0.110 2.64M 332</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">stump</th>
      <th colspan="4">treehill</th>
      <th colspan="4">room</th>
    </tr>
    <tr>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS [17]</td>
      <td>21.77</td>
      <td>26.64</td>
      <td>0.768</td>
      <td>0.244 4.05M 130</td>
      <td>20.33</td>
      <td>22.59</td>
      <td>0.636</td>
      <td>0.347 3.01M 136</td>
      <td>18.78</td>
      <td>31.71</td>
      <td>0.927</td>
      <td>0.197 1.25M 164</td>
    </tr>
    <tr>
      <td>Mini-Splatting [8]</td>
      <td>16.52</td>
      <td>26.80</td>
      <td>0.839</td>
      <td>0.161 0.67M 521</td>
      <td>17.05</td>
      <td>22.76</td>
      <td>0.656</td>
      <td>0.326 0.63M 487</td>
      <td>18.00</td>
      <td>31.48</td>
      <td>0.928</td>
      <td>0.190 0.39M 506</td>
    </tr>
    <tr>
      <td>Speedy-Splat [12]</td>
      <td>13.77</td>
      <td>26.67</td>
      <td>0.765</td>
      <td>0.288 0.46M 480</td>
      <td>12.90</td>
      <td>22.48</td>
      <td>0.590</td>
      <td>0.462 0.32M 548</td>
      <td>12.05</td>
      <td>30.83</td>
      <td>0.903</td>
      <td>0.258 0.11M 617</td>
    </tr>
    <tr>
      <td>Taming-3DGS [24]</td>
      <td>3.93</td>
      <td>26.05</td>
      <td>0.729</td>
      <td>0.324 0.48M 280</td>
      <td>5.37</td>
      <td>22.92</td>
      <td>0.628</td>
      <td>0.395 0.79M 214</td>
      <td>3.88</td>
      <td>31.64</td>
      <td>0.917</td>
      <td>0.227 0.23M 230</td>
    </tr>
    <tr>
      <td>DashGaussian [4]</td>
      <td>6.57</td>
      <td>27.17</td>
      <td>0.783</td>
      <td>0.229 3.42M 164</td>
      <td>8.20</td>
      <td>22.94</td>
      <td>0.640</td>
      <td>0.333 3.42M 134</td>
      <td>4.00</td>
      <td>31.81</td>
      <td>0.924</td>
      <td>0.205 1.04M 182</td>
    </tr>
    <tr>
      <td>FastGS</td>
      <td>1.72</td>
      <td>26.65</td>
      <td>0.756</td>
      <td>0.297 0.39M 576</td>
      <td>1.72</td>
      <td>22.94</td>
      <td>0.612</td>
      <td>0.429 0.38M 568</td>
      <td>1.62</td>
      <td>31.98</td>
      <td>0.920</td>
      <td>0.217 0.21M 632</td>
    </tr>
    <tr>
      <td>FastGS-Big</td>
      <td>2.88</td>
      <td>27.18</td>
      <td>0.786</td>
      <td>0.240 1.06M 489</td>
      <td>2.78</td>
      <td>22.83</td>
      <td>0.632</td>
      <td>0.378 1.01M 507</td>
      <td>2.38</td>
      <td>32.20</td>
      <td>0.929</td>
      <td>0.189 0.57M 577</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">counter</th>
      <th colspan="4">kitchen</th>
      <th colspan="4">bonsai</th>
    </tr>
    <tr>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS [17]</td>
      <td>17.58</td>
      <td>29.16</td>
      <td>0.915</td>
      <td>0.183 1.05M 171</td>
      <td>21.30</td>
      <td>31.54</td>
      <td>0.932</td>
      <td>0.116 1.53M 144</td>
      <td>14.90</td>
      <td>32.37</td>
      <td>0.946</td>
      <td>0.180 1.07M 224</td>
    </tr>
    <tr>
      <td>Mini-Splatting [8]</td>
      <td>9.83</td>
      <td>28.65</td>
      <td>0.911</td>
      <td>0.181 0.41M 590</td>
      <td>10.23</td>
      <td>31.05</td>
      <td>0.930</td>
      <td>0.120 0.44M 614</td>
      <td>11.02</td>
      <td>31.24</td>
      <td>0.943</td>
      <td>0.177 0.36M 661</td>
    </tr>
    <tr>
      <td>Speedy-Splat [12]</td>
      <td>12.10</td>
      <td>28.22</td>
      <td>0.876</td>
      <td>0.259 0.10M 606</td>
      <td>13.25</td>
      <td>30.09</td>
      <td>0.895</td>
      <td>0.195 0.11M 608</td>
      <td>11.37</td>
      <td>31.16</td>
      <td>0.925</td>
      <td>0.228 0.13M 652</td>
    </tr>
    <tr>
      <td>Taming-3DGS [24]</td>
      <td>4.60</td>
      <td>29.20</td>
      <td>0.909</td>
      <td>0.200 0.31M 221</td>
      <td>3.48</td>
      <td>31.84</td>
      <td>0.929</td>
      <td>0.128 0.48M 209</td>
      <td>4.60</td>
      <td>32.40</td>
      <td>0.942</td>
      <td>0.193 0.41M 227</td>
    </tr>
    <tr>
      <td>DashGaussian [4]</td>
      <td>3.95</td>
      <td>29.11</td>
      <td>0.911</td>
      <td>0.191 0.85M 162</td>
      <td>5.52</td>
      <td>31.69</td>
      <td>0.927</td>
      <td>0.129 1.18M 135</td>
      <td>3.95</td>
      <td>32.15</td>
      <td>0.945</td>
      <td>0.180 0.82M 193</td>
    </tr>
    <tr>
      <td>FastGS</td>
      <td>1.83</td>
      <td>29.15</td>
      <td>0.907</td>
      <td>0.204 0.21M 596</td>
      <td>2.42</td>
      <td>31.87</td>
      <td>0.929</td>
      <td>0.127 0.38M 543</td>
      <td>1.83</td>
      <td>32.19</td>
      <td>0.942</td>
      <td>0.191 0.28M 622</td>
    </tr>
    <tr>
      <td>FastGS-Big</td>
      <td>2.62</td>
      <td>29.57</td>
      <td>0.917</td>
      <td>0.177 0.47M 522</td>
      <td>5.15</td>
      <td>32.17</td>
      <td>0.938</td>
      <td>0.105 1.18M 395</td>
      <td>3.22</td>
      <td>32.97</td>
      <td>0.953</td>
      <td>0.161 0.85M 498</td>
    </tr>
  </tbody>
</table>

Table 18. Scene-wise quantitative results over the Deep Blending dataset [5].

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">playroom</th>
      <th colspan="4">drjohnson</th>
    </tr>
    <tr>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS [17]</td>
      <td>16.75</td>
      <td>30.14</td>
      <td>0.904</td>
      <td>0.243 1.85M</td>
      <td>22.78</td>
      <td>29.28</td>
      <td>0.902</td>
      <td>0.239 3.07M</td>
      <td>126</td>
    </tr>
    <tr>
      <td>Mini-Splatting [8]</td>
      <td>12.30</td>
      <td>30.47</td>
      <td>0.908</td>
      <td>0.241 0.51M</td>
      <td>14.40</td>
      <td>29.51</td>
      <td>0.905</td>
      <td>0.246 0.60M</td>
      <td>629</td>
    </tr>
    <tr>
      <td>Speedy-Splat [12]</td>
      <td>9.70</td>
      <td>29.77</td>
      <td>0.898</td>
      <td>0.274 0.18M</td>
      <td>11.80</td>
      <td>29.07</td>
      <td>0.898</td>
      <td>0.269 0.31M</td>
      <td>633</td>
    </tr>
    <tr>
      <td>Taming-3DGS [24]</td>
      <td>3.35</td>
      <td>29.96</td>
      <td>0.901</td>
      <td>0.264 0.40M</td>
      <td>2.77</td>
      <td>29.04</td>
      <td>0.888</td>
      <td>0.292 0.19M</td>
      <td>349</td>
    </tr>
    <tr>
      <td>DashGaussian [4]</td>
      <td>4.70</td>
      <td>30.17</td>
      <td>0.909</td>
      <td>0.243 2.38M</td>
      <td>3.62</td>
      <td>29.13</td>
      <td>0.903</td>
      <td>0.250 1.51M</td>
      <td>182</td>
    </tr>
    <tr>
      <td>FastGS</td>
      <td>1.22</td>
      <td>30.57</td>
      <td>0.905</td>
      <td>0.266 0.19M</td>
      <td>1.35</td>
      <td>29.50</td>
      <td>0.898</td>
      <td>0.275 0.25M</td>
      <td>700</td>
    </tr>
    <tr>
      <td>FastGS-Big</td>
      <td>1.93</td>
      <td>30.55</td>
      <td>0.909</td>
      <td>0.239 0.60M</td>
      <td>2.07</td>
      <td>29.69</td>
      <td>0.905</td>
      <td>0.247 0.70M</td>
      <td>595</td>
    </tr>
  </tbody>
</table>

Table 19. Scene-wise quantitative results over the Tanks & Temples dataset [20].

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">truck</th>
      <th colspan="4">train</th>
    </tr>
    <tr>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
      <th>Time↓</th>
      <th>PSNR↑</th>
      <th>SSIM↑</th>
      <th>LPIPS↓ NGS ↓ FPS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3DGS [17]</td>
      <td>12.50</td>
      <td>25.39</td>
      <td>0.881</td>
      <td>0.142 2.05M</td>
      <td>186</td>
      <td>10.18</td>
      <td>22.03</td>
      <td>0.818</td>
      <td>0.198 1.09M</td>
      <td>203</td>
    </tr>
    <tr>
      <td>Mini-Splatting [8]</td>
      <td>9.08</td>
      <td>25.32</td>
      <td>0.879</td>
      <td>0.139 0.32M</td>
      <td>716</td>
      <td>9.03</td>
      <td>21.60</td>
      <td>0.809</td>
      <td>0.223 0.28M</td>
      <td>795</td>
    </tr>
    <tr>
      <td>Speedy-Splat [12]</td>
      <td>7.22</td>
      <td>25.18</td>
      <td>0.863</td>
      <td>0.192 0.26M</td>
      <td>648</td>
      <td>5.42</td>
      <td>21.59</td>
      <td>0.768</td>
      <td>0.292 0.11M</td>
      <td>733</td>
    </tr>
    <tr>
      <td>Taming-3DGS [24]</td>
      <td>2.38</td>
      <td>25.27</td>
      <td>0.865</td>
      <td>0.187 0.27M</td>
      <td>409</td>
      <td>3.03</td>
      <td>22.50</td>
      <td>0.802</td>
      <td>0.240 0.37M</td>
      <td>348</td>
    </tr>
    <tr>
      <td>DashGaussian [4]</td>
      <td>4.25</td>
      <td>25.80</td>
      <td>0.886</td>
      <td>0.150 1.43M</td>
      <td>257</td>
      <td>4.32</td>
      <td>22.19</td>
      <td>0.819</td>
      <td>0.206 1.00M</td>
      <td>222</td>
    </tr>
    <tr>
      <td>FastGS</td>
      <td>1.30</td>
      <td>25.73</td>
      <td>0.872</td>
      <td>0.178 0.25M</td>
      <td>666</td>
      <td>1.33</td>
      <td>22.57</td>
      <td>0.805</td>
      <td>0.242 0.23M</td>
      <td>644</td>
    </tr>
    <tr>
      <td>FastGS-Big</td>
      <td>2.13</td>
      <td>26.09</td>
      <td>0.886</td>
      <td>0.140 0.63M</td>
      <td>579</td>
      <td>1.93</td>
      <td>22.68</td>
      <td>0.824</td>
      <td>0.210 0.46M</td>
      <td>558</td>
    </tr>
  </tbody>
</table>

&lt;page_number&gt;5&lt;/page_number&gt;

---


## Page 17

&lt;img&gt;
<table>
  <tr>
    <td>&lt;img&gt;Image of a bicycle in a garden&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a bicycle in a garden&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a bicycle in a garden&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a bicycle in a garden&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>&lt;img&gt;Image of a garden table&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a garden table&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a garden table&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a garden table&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>&lt;img&gt;Image of a stump&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a stump&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a stump&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a stump&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>&lt;img&gt;Image of a living room&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a living room&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a living room&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a living room&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>&lt;img&gt;Image of a kitchen counter&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a kitchen counter&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a kitchen counter&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a kitchen counter&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>&lt;img&gt;Image of a room with a toy bulldozer&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a room with a toy bulldozer&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a room with a toy bulldozer&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a room with a toy bulldozer&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>&lt;img&gt;Image of a bonsai tree&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a bonsai tree&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a bonsai tree&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a bonsai tree&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>&lt;img&gt;Image of a playroom&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a playroom&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a playroom&lt;/img&gt;</td>
    <td>&lt;img&gt;Image of a playroom&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>DashGaussian</td>
    <td>Speedy-Splat</td>
    <td>FastGS(ours)</td>
    <td>FastGS-Big(ours)</td>
    <td>Ground Truth</td>
  </tr>
</table>
&lt;/img&gt;
Figure 8. Additional visual comparisons on the bicycle, garden, stump, room, counter, kitchen, bonsai, and playroom scenes.

&lt;page_number&gt;6&lt;/page_number&gt;

---


## Page 18

&lt;img&gt;
<table>
  <tr>
    <td>&lt;img&gt;Deformable-3DGS&lt;/img&gt;</td>
    <td>&lt;img&gt;+Ours&lt;/img&gt;</td>
    <td>&lt;img&gt;Ground Truth&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>&lt;img&gt;Deformable-3DGS&lt;/img&gt;</td>
    <td>&lt;img&gt;+Ours&lt;/img&gt;</td>
    <td>&lt;img&gt;Ground Truth&lt;/img&gt;</td>
  </tr>
  <tr>
    <td colspan="3">Dynamic Scene Reconstruction</td>
  </tr>
  <tr>
    <td>&lt;img&gt;PGSR&lt;/img&gt;</td>
    <td>&lt;img&gt;+Ours&lt;/img&gt;</td>
    <td>&lt;img&gt;Ground Truth&lt;/img&gt;</td>
  </tr>
  <tr>
    <td>&lt;img&gt;PGSR&lt;/img&gt;</td>
    <td>&lt;img&gt;+Ours&lt;/img&gt;</td>
    <td>&lt;img&gt;Ground Truth&lt;/img&gt;</td>
  </tr>
  <tr>
    <td colspan="3">Surface Reconstruction</td>
  </tr>
</table>
&lt;/img&gt;
Figure 9. Additional visual comparisons on different tasks, including the flame_salmon, cut_roasted_beef, Caterpillar, Meetingroom scenes.

&lt;page_number&gt;7&lt;/page_number&gt;

---


## Page 19

&lt;img&gt;
  &lt;img&gt;Figure 10. Additional visual comparisons on different tasks, including the trex, leaves, amsterdam, and office0 scenes.&lt;/img&gt;
&lt;/img&gt;
&lt;page_number&gt;8&lt;/page_number&gt;