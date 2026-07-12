# Phân Tích Pipeline Spec-FastGS — Visual Quality Tại Vùng Specular

> **Codebase**: [`spec-fastgs`](file:///home/baodinh/baodinh_thesis/spec-fastgs)  
> **Câu hỏi trung tâm**: Với bài toán cải thiện visual quality tại vùng specular, pipeline hiện tại đã làm tốt và chưa làm tốt điều gì?

---

## Tổng Quan — 4 Trục Ảnh Hưởng

Visual quality tại vùng specular phụ thuộc vào **4 trục chính**, xếp theo mức độ ảnh hưởng:

| # | Trục | Mô tả ngắn | Mức quan trọng |
|---|------|------------|----------------|
| 1 | **Geometric Coverage** | Có đủ Gaussian ở đúng vị trí specular không? | ★★★★★ |
| 2 | **Representation Capacity** | MLP/ASG có đủ năng lực biểu diễn specular theo góc nhìn không? | ★★★★☆ |
| 3 | **Supervision Signal** | Loss có đủ "kéo" model về vùng specular không? | ★★★☆☆ |
| 4 | **Normal Quality** | Normal Gaussian có đủ chính xác để tính reflection direction không? | ★★★☆☆ |

---

## Trục 1 — Geometric Coverage

### Bản chất vấn đề

Specular highlight là hiện tượng **cực kỳ cục bộ** (vài pixel đến vài chục pixel) và thay đổi nhanh theo góc nhìn. Nếu không có đủ Gaussian tại vùng đó thì dù MLP tốt đến đâu cũng không render được chính xác.

Pipeline 3DGS gốc dùng cơ chế `densify_and_prune` dựa trên **gradient viewspace**. Với vùng specular, gradient thường nhỏ và không ổn định do:
- Highlight thay đổi mạnh theo góc nhìn → L1/SSIM không nhất quán đa-view → gradient trung bình gần 0
- Vùng specular nhỏ → đóng góp ít vào tổng gradient

### Pipeline đã làm tốt

**Ref Score force densification** ([`fast_utils.py:L86-L95`](file:///home/baodinh/baodinh_thesis/spec-fastgs/utils/fast_utils.py#L86-L95)):

```python
if use_ref_score:
    ref_mask = (my_viewpoint_cam.ref_score.cuda() > 0.5).int()
    metric_map = torch.max(metric_map, ref_mask)  # OR với photometric error
```

Mỗi 500 iteration (nếu còn dưới 400K Gaussians), pipeline **ép buộc** coi vùng specular là "high-error" dù photometric loss thực sự thấp. Điều này đi ngược lại hành vi mặc định của 3DGS gốc và là điểm đóng góp chính của spec-fastgs.

**Prior point cloud initialization** ([`generate_prior_pcd.py`](file:///home/baodinh/baodinh_thesis/spec-fastgs/generate_prior_pcd.py)):
- Space carving 200³ = 8M điểm grid 3D
- Tích lũy ref_score theo từng view cho mỗi điểm 3D sống sót
- Phân bổ 50% trong 100K điểm khởi tạo vào vùng top-K accumulated ref_score

Kết quả: Gaussian được **gieo hạt ngay từ đầu** tại vùng specular thay vì chờ densification tự tìm ra.

### Pipeline chưa làm tốt

**Prior tĩnh, không adaptive**:  
Ref Score được tính từ ảnh gốc **một lần duy nhất** trước khi train. Nếu model ở iter 8000 đã densify đủ Gaussian vào đúng chỗ rồi, iter 8500 vẫn cứ force densify thêm — không có cơ chế phản hồi "đã đủ". Ngược lại, nếu model chưa converge đúng chỗ nhưng budget 400K đã hết → dừng hoàn toàn.

**False positive do model đơn giản**:  
Cả hai thuật toán Tan-Ikeuchi và Shafer-Klinker hoạt động trên pixel đơn lẻ, không có thông tin hình học. Nền trắng, vật liệu diffuse sáng, vùng overexposed đều bị nhầm là specular → lãng phí budget Gaussian vào những chỗ không cần thiết.

```python
# extract_reflection_prior.py — không có geometric context
mask = (Imax > intensity_thresh) & (saturation < sat_thresh)
```

**Budget cap cứng và không theo ngữ cảnh**:  
`max_refscore_gaussians = 400_000` là hằng số toàn cục, không phụ thuộc vào độ phức tạp của scene hay số lượng vùng specular thực sự.

---

## Trục 2 — Representation Capacity

### Bản chất vấn đề

Specular không chỉ cần "MLP lớn hơn". Capacity của trục này phụ thuộc vào 4 điều kiện cùng lúc:

1. **Biến đầu vào đúng vật lý**: highlight nên là hàm của reflection direction, normal, và góc nhìn, không chỉ raw view direction.
2. **Basis/lobe đủ sắc**: ASG phải có đủ lobe, độ nhọn, và hidden capacity để biểu diễn highlight nhỏ, anisotropic.
3. **Phân công vai trò rõ với SH**: SH nên giữ vai trò base/diffuse hoặc view-dependent rất thấp tần; ASG nên giải thích phần residual specular.
4. **ASG nhận đủ gradient đúng Gaussian**: sparse ASG evaluation không được làm mất supervision ở những Gaussian đang visible/specular trong frame hiện tại.

Nếu chỉ tăng `asg_degree` mà không giải quyết role separation và sparse supervision, model có thể vẫn cho `only_asg` yếu vì SH đã hấp thụ highlight trước.

### Pipeline đã làm tốt

**ASG branch tách khỏi SH và có latent per Gaussian**:

- `GaussianModel` lưu riêng `_features_asg` cho từng Gaussian.
- `SpecularModel` có MLP riêng và optimizer riêng.
- `f_rest` của SH bậc cao cũng được tách optimizer, nên về mặt kỹ thuật có thể kiểm soát cạnh tranh SH/ASG mà không đụng `f_dc`, geometry, opacity.
- Bản Representation Capacity patch bổ sung `use_sh_spec_mask` để scale/zero gradient `f_rest` tại Gaussian overlap vùng `ref_score` cao. Mặc định vẫn tắt để giữ baseline, nhưng đã có đường ablation trực tiếp cho role separation.

**Synthetic ASG branch dùng reflection direction** ([`spec_utils.py:L112-L155`](file:///home/baodinh/baodinh_thesis/spec-fastgs/utils/spec_utils.py#L112-L155)):

```python
# ASGRender.forward()
reflect_dir = safe_normalize(reflect(-viewdirs, normal))  # reflection direction
color_feature = self.ree_function(reflect_dir, a, la, mu)  # RenderingEquationEncoding
```

Với `SpecularNetwork` synthetic, `RenderingEquationEncoding` mô hình hóa BRDF dạng Anisotropic Spherical Gaussian trên cầu phương hướng — tốt hơn pure MLP blind vì:
- Encode đúng biến vật lý (reflection direction thay vì raw viewdir)
- Tham số `la`, `mu` kiểm soát độ nhọn của lobe theo từng trục

Mỗi Gaussian có `asg_degree=24` latent features -> linear layer -> `4×8 lobes × 4 params = 128` ASG parameters.

**Hai chế độ phù hợp loại scene**:
- `SpecularNetwork`: synthetic, 128 hidden, 4×8 lobes — cho phép biểu diễn specular phức tạp hơn
- `SpecularNetworkReal`: real scene, 32 hidden, 2×4 lobes — nhẹ hơn, tránh overfit noise

Tuy nhiên cần ghi rõ giới hạn: branch real hiện tại dùng `viewdirs` trực tiếp trong REE, không dùng reflection direction/normal như branch synthetic. Vì vậy không nên mô tả toàn bộ pipeline là physics-informed theo reflection direction; điều đó chỉ đúng hoàn toàn với synthetic branch.

**Sparse ASG evaluation** ([`train.py:L162-L180`](file:///home/baodinh/baodinh_thesis/spec-fastgs/train.py#L162-L180)):  
Chỉ tính MLP cho Gaussian visible trong frame trước → cắt 3–10× chi phí MLP forward/backward mà không ảnh hưởng chất lượng nhiều.

**Diagnostic render đã đúng hướng**: `render.py` lưu `only_sh`, `only_asg`, `residual_real`. Đây là điều kiện cần để kiểm tra ASG có thật sự biểu diễn specular hay chỉ bị SH che mất.

### Pipeline chưa làm tốt

**SH cạnh tranh ASG — chưa có role separation theo không gian**:  
Đây vẫn là điểm yếu quan trọng nhất của Trục 2. SH degree 3 có 16 basis functions, đủ để fit một phần view-dependent color. Khi SH và ASG cùng học từ cùng một photometric loss:

```
loss = L1(SH_color + ASG_color, GT)   # cả hai đều nhận gradient
```

Nếu SH đã fit specular trước khi ASG bắt đầu (`specular_start_iter=3000`), hoặc SH liên tục "ăn" phần residual specular trong suốt training -> `only_asg` yếu, `ASG_Energy_In_Residual` thấp.

Pipeline gốc chỉ có `f_rest_interval_*`, tức **temporal throttle**: giảm tần suất update SH bậc cao trên toàn scene. Bản Representation Capacity patch đã thêm spatial role separation qua `use_sh_spec_mask`, nhưng cơ chế này vẫn cần ablation vì hiệu quả phụ thuộc chất lượng `ref_score`: prior false positive rộng có thể chặn SH ở vùng diffuse sáng.

**`ASG_DEGREE` chưa phải capacity knob đầy đủ**:  
Ablation nhóm C thay đổi `asg_degree`, nhưng trong code hiện tại:

- Synthetic branch vẫn cố định `num_theta=4`, `num_phi=8`, `hidden=128`.
- Real branch vẫn cố định `num_theta=2`, `num_phi=4`, `hidden=32`.
- `asg_degree` chỉ đổi số chiều latent đầu vào trước linear layer, không đổi trực tiếp số lobe ASG hay độ rộng MLP đầu ra.

Vì vậy nếu `C2: ASG 32` không cải thiện, chưa thể kết luận "ASG capacity không phải bottleneck"; có thể bottleneck thật nằm ở lobe grid/hidden width/real-branch design.

**Real-scene branch bỏ normal/reflection direction**:  
`ASGRenderReal.forward()` gọi `self.ree_function(viewdirs, a, la, mu)`. Điều này giảm phụ thuộc vào normal nhiễu, nhưng cũng làm mất cấu trúc vật lý quan trọng của specular reflection. Với object glossy/curved, highlight phụ thuộc mạnh vào normal; viewdir-only branch có thể học được một phần nhưng dễ underfit view-dependent reflectance đúng nghĩa.

**Sparse ASG dùng visibility của frame trước, không phải frame hiện tại**:  
Code hiện tại có xử lý đúng trường hợp densification làm đổi số Gaussian: nếu `prev_vis_mask.shape[0] != n_gs`, nó fallback sang full ASG. Vì vậy "Gaussian mới bị bỏ sót do count mismatch" không phải bug lớn.

Gap thật sự là khác: camera train được sample ngẫu nhiên, nên Gaussian visible ở camera hiện tại có thể không nằm trong `prev_vis_mask` của camera trước. Những Gaussian này render bằng SH-only trong iteration đó, không nhận gradient ASG ở frame đang cần specular. Đây là trade-off tốc độ/chất lượng của sparse ASG; `full_asg_interval=0` nghĩa là không có refresh định kỳ để giảm nhiễu này.

**Specular regularization chỉ nên là optional ablation**:  
Trong baseline cũ, `arguments/__init__.py` có `lambda_spec_reg = 0.01`, và comment trong `train.py` nhắc `spec_reg`, nhưng loss thực tế chỉ là:

```python
loss = photometric_loss
```

Sau patch, `lambda_spec_reg` được nối vào loss nhưng default đổi về `0.0` để không làm đổi baseline ngầm. Đây là L2 penalty chống ASG output quá lớn/leak rộng; nó không chống collapse về zero, và nếu quá mạnh có thể làm `only_asg` yếu hơn.

`only_asg = clamp(full - sh, min=0)` trong render vẫn chỉ là diagnostic dương, có thể bỏ sót các correction âm mà ASG đang học.

**Thiếu logging capacity/role state**:  
`train_info.json` đã log `asg_degree`, `specular_start_iter`, `full_asg_interval`, và lịch `f_rest`, nhưng chưa log các đại lượng giúp đọc nguyên nhân:

- Tỉ lệ Gaussian bị xem là specular-overlap.
- Tỉ lệ gradient `f_rest` bị mask/scale nếu thêm role separation.
- Năng lượng ASG trong train loop.
- Số Gaussian được ASG evaluate mỗi iteration.

---

## Trục 3 — Supervision Signal

### Bản chất vấn đề

Specular chiếm diện tích nhỏ trong ảnh (thường 1–10% pixel). Loss pixel-average như L1 + SSIM tối ưu trên toàn ảnh — vùng specular chỉ đóng góp một phần nhỏ gradient. Model dễ bị "bỏ rơi" specular vì tối ưu diffuse trên nền rộng sẽ giảm loss nhanh hơn.

### Pipeline đã làm tốt

**Specular-specific metrics được định nghĩa** ([`ablation_ver.md:L53-L68`](file:///home/baodinh/baodinh_thesis/spec-fastgs/ablation_ver.md#L53-L68)):

```
Spec_L1      — L1 lỗi trên vùng specular proxy
Spec_PSNR    — PSNR riêng tại vùng specular
ASG_Energy_In_Residual  — % năng lượng ASG rơi vào vùng residual specular
ASG_Residual_IoU        — overlap giữa vùng ASG sáng và vùng residual
```

Việc **đo được vấn đề** là bước đầu tiên để giải quyết nó — đây là cơ sở cho ablation experiments.

**Ref Score gián tiếp tăng supervision** tại vùng specular: bằng cách force densify → nhiều Gaussian hơn tại vùng specular → nhiều gradient hơn chảy qua những vùng đó trong các forward pass thông thường.

### Pipeline chưa làm tốt

**Specular-weighted loss đã có nhưng mặc định tắt**:  
Baseline vẫn là L1 + SSIM toàn ảnh:

```python
# train.py L218-L225
Ll1 = l1_loss(image, gt)
ssim_val = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))
photometric_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
loss = photometric_loss
```

Sau patch, `lambda_spec_l1_weight > 0` sẽ dùng normalized weighted L1 theo `cam.ref_score`. Tuy nhiên nó mặc định `0.0` vì nếu bật khi chưa có role separation, gradient tăng thêm vẫn có thể bị SH hấp thụ.

**Spec_reg là công cụ phụ, không phải lời giải role separation**:  
`lambda_spec_reg` đã được nối vào loss nhưng mặc định `0.0`. Nó giúp kiểm tra ASG output quá lớn/leak rộng, nhưng không tự kéo ASG vào đúng vùng specular và không giải quyết cạnh tranh SH/ASG.

**Không có perceptual loss trên vùng specular**:  
LPIPS tốt hơn L1 trong việc đánh giá quality của highlight sắc nét, nhưng không được dùng trong training loss (chỉ dùng để evaluate).

---

## Trục 4 — Normal Quality

### Bản chất vấn đề

`ASGRender` tính `reflect_dir = 2(v·n)n - v`. Nếu normal sai → reflection direction sai → MLP fit sai hàm từ đầu. Normal của Gaussian được tính từ **minimum scaling axis** — trục ngắn nhất của ellipsoid:

```python
# gaussian_model.py
def get_normal_axis(self, dir_pp_normalized=None):
    normal_axis = self.get_minimum_axis       # proxy heuristic, không phải normal thật
    normal_axis, positive = flip_align_view(normal_axis, dir_pp_normalized)
    return normal_axis / normal_axis.norm(dim=1, keepdim=True)
```

### Pipeline đã làm tốt

**`flip_align_view`**: Flip normal để luôn hướng về phía camera, tránh backface ambiguity (Gaussian dẹt có 2 mặt → chọn đúng mặt).

**`normal.detach()`** khi truyền vào MLP:

```python
# train.py L172
spec_sparse = specular_mlp.step(
    asg_feat[vis_indices],
    viewdir[vis_indices],
    normal[vis_indices].detach(),   # gradient không chảy ngược qua normal
)
```

Tránh MLP kéo geometry (scaling/rotation của Gaussian) theo hướng sai chỉ để fit màu.

### Pipeline chưa làm tốt

**Không có normal smoothness / consistency regularization**:  
Không có gì buộc normal của các Gaussian lân cận phải smooth và nhất quán với nhau. Vùng specular có bề mặt cong → normal cần chính xác cục bộ; nhưng một Gaussian lớn phủ vùng cong sẽ có một normal duy nhất cho toàn bộ vùng đó.

**Gaussian to, tròn = normal nhiễu**:  
Gaussian ở vùng diffuse/background thường lớn → minimum axis không ổn định → nếu ASG được eval trên những Gaussian này thì reflection direction rất nhiễu. Vùng specular thường có Gaussian nhỏ hơn (nhờ ref score densification), nhưng không đảm bảo.

**Normal là proxy, không phải normal thật của bề mặt**:  
Với các object có bề mặt phức tạp (metal, glass, curved mirror), minimum axis của Gaussian ellipsoid không đủ chính xác để tính reflection direction đúng. Các phương pháp như [GaussianShader](https://arxiv.org/abs/2311.17977) hay [Relightable 3DGS](https://arxiv.org/abs/2311.16043) dùng thêm một *learned normal delta* per Gaussian để tinh chỉnh — hiện tại `spec-fastgs` chưa có.

---

## Bảng Tổng Hợp

| Trục | Pipeline làm được | Pipeline chưa làm tốt | Mức quan trọng |
|------|-------------------|----------------------|----------------|
| **Geometric Coverage** | Ref Score force densify mỗi 500 iter; prior PCD init 50% specular | Prior tĩnh; false positive nền trắng; budget cap cứng 400K | ★★★★★ |
| **Representation Capacity** | ASG latent/MLP riêng; synthetic branch dùng reflection dir; optimizer riêng cho `f_rest`/`f_asg`; optional `use_sh_spec_mask`; ASG architecture knobs; sparse eval tiết kiệm chi phí | Role separation mặc định tắt và cần ablation; real branch vẫn viewdir-only mặc định; sparse eval lệch visibility theo camera | ★★★★☆ |
| **Supervision Signal** | Specular metrics đo được; ref score tăng gradient gián tiếp; optional normalized specular L1; optional `lambda_spec_reg` | Weighted loss/spec_reg đều mặc định tắt để giữ baseline; chưa có perceptual loss vùng specular | ★★★☆☆ |
| **Normal Quality** | flip_align_view; normal.detach() | Không có normal smoothness loss; normal là proxy (min axis), không phải normal bề mặt thật | ★★★☆☆ |

---

## Điểm Yếu Then Chốt

> [!IMPORTANT]
> **Vấn đề cốt lõi cần ablate: role separation SH/ASG (Trục 2)**
>
> Dù Geometric Coverage và ASG capacity đều được cải thiện, nếu SH đã "ăn" phần specular trước hoặc trong suốt quá trình train, thì `only_asg` sẽ yếu và mọi cải thiện ở Trục 1 trở nên kém ý nghĩa. Bản patch đã thêm `use_sh_spec_mask` để chặn/giảm gradient `f_rest` tại Gaussian overlap vùng specular; đây là proof-of-concept cần chạy ablation đầu tiên.

> [!TIP]
> **Hướng cải tiến có tác động cao nhất theo thứ tự ưu tiên**:
> 1. **Gaussian-level SH gradient masking/scaling** — dùng `ref_score` + rasterizer `accum_metric_counts` để giảm gradient `f_rest` ở Gaussian overlap vùng specular, buộc ASG nhận residual view-dependent.
> 2. **ASG architecture knobs thật sự** — tách `asg_degree` khỏi `asg_num_theta`, `asg_num_phi`, `specular_hidden`, `specular_layers`, và thêm lựa chọn real branch dùng reflection direction.
> 3. **Specular-weighted residual supervision** — boosting gradient tại vùng ref_score/residual cao là hỗ trợ cần thiết, nhưng thuộc trục Supervision Signal; nên đi sau hoặc đi cùng role separation để gradient không tiếp tục bị SH hấp thụ.
> 4. **Optional ASG regularization** — nối `lambda_spec_reg` vào loss với default `0.0`, chỉ bật khi cần kiểm tra ASG explode/leak; không dùng nó như default vì có thể làm `only_asg` yếu hơn.

---

## Liên Hệ Với Ablation Plan

| Nhóm Ablation | Trục liên quan | Câu hỏi đang trả lời |
|---------------|---------------|----------------------|
| A1 vs A2 (ref score on/off) | Trục 1 | Ref Score densification đóng góp bao nhiêu? |
| B0–B3 (prior method) | Trục 1 | Prior algorithm nào giảm false positive nhất? |
| C0–C2 (ASG degree) | Trục 2 | Capacity ASG cần bao nhiêu để đủ expressiveness? |
| F0–F2 (f_rest schedule) | Trục 2 | Giảm SH update có giúp ASG chiếm phần specular không? |
| E0–E2 (specular_start_iter) | Trục 2 | Timing bắt đầu ASG ảnh hưởng thế nào đến role separation? |
| D0–D2 (num_score_cameras) | Trục 3 | Nhiều camera → signal densification ổn định hơn? |
