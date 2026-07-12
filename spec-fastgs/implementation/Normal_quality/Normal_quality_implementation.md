# Thiết Kế Normal Quality — Đảm Bảo Tính Chính Xác Của Reflection Direction

Dựa trên `Pipeline_analysis.md`, ASG sử dụng reflection direction, mà hướng này phụ thuộc hoàn toàn vào Normal của Gaussian (`n`). Hiện tại, normal được tính xấp xỉ bằng **trục ngắn nhất (minimum axis)** của ellipsoid.

Với các Gaussian ở vùng diffuse hoặc background, hình dáng của chúng có thể rất lớn và ngẫu nhiên, dẫn đến normal bị nhiễu. Nếu normal bị nhiễu, reflection direction sai → ASG học sai.

---

## 1. Normal Smoothness Regularization (Ưu tiên Trung bình - Khả thi)

**Mục tiêu:** Bề mặt của các vật thể bóng (specular) thường nhẵn (smooth) và liên tục. Do đó, normal của các Gaussian nằm cạnh nhau cũng phải mượt mà và tương đồng.

**Thiết kế:**
- Render Normal Map ra thành một ảnh 2D (tương tự render color).
- Áp dụng Total Variation (TV) loss hoặc Edge-aware Smoothness loss lên Normal Map này.
- Loss này sẽ ép quá trình tối ưu hình học (scaling, rotation) của các Gaussian sao cho trục ngắn nhất của chúng "xếp hàng" thẳng lối với nhau.

**Implementation (trong `gaussian_renderer/__init__.py` và `train.py`):**

1. Cần sửa rasterizer hoặc hàm `render_fastgs` để trả về thêm `normal_map`.
   - Normal per-Gaussian: `normal = gaussians.get_normal_axis()`
   - Đưa `normal` vào rasterizer như một color channel (thay vì RGB) để rasterize ra màn hình.
   *(Lưu ý: Diff-Gaussian-Rasterization hiện tại không hỗ trợ render multiple feature cùng lúc một cách dễ dàng, có thể phải gọi rasterizer 2 lần, hoặc chỉnh sửa C++ CUDA code. Nếu không sửa CUDA, có thể render riêng normal bằng cách gán `f_dc` tạm thời thành normal, nhưng rất tốn kém).*

2. Tính TV Loss trên `normal_map` ở `train.py`:
```python
def tv_loss(normal_map):
    # normal_map: [3, H, W]
    dx = torch.mean(torch.abs(normal_map[:, :, 1:] - normal_map[:, :, :-1]))
    dy = torch.mean(torch.abs(normal_map[:, 1:, :] - normal_map[:, :-1, :]))
    return dx + dy

# Trong train loop
loss = loss + opt.lambda_normal_smooth * tv_loss(normal_map)
```

**Rủi ro:** Sửa rasterizer phức tạp. Có thể làm giảm tốc độ render.

---

## 2. Learned Normal Delta (Ưu tiên Trung bình - Nâng cao)

**Mục tiêu:** Trục ngắn nhất (minimum axis) chỉ là xấp xỉ. Cho phép model học thêm một lượng sai số (delta) nhỏ để bù đắp, giúp normal chính xác hơn (Tương tự ý tưởng của GaussianShader / Relightable 3DGS).

**Thiết kế:**
Thêm một tham số học được (learnable parameter) `_normal_delta` cho mỗi Gaussian.

**Implementation (trong `gaussian_model.py`):**

```python
# 1. Khởi tạo
self._normal_delta = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 3), device="cuda").requires_grad_(True))

# 2. Thêm vào optimizer
{'params': [self._normal_delta], 'lr': training_args.normal_lr, "name": "normal_delta"}

# 3. Tính normal mới
def get_normal_axis(self, dir_pp_normalized=None):
    base_normal = get_minimum_axis(self.get_scaling, self.get_rotation)
    
    # Cộng thêm delta
    normal = base_normal + self._normal_delta
    normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-6)
    
    normal, positive = flip_align_view(normal, dir_pp_normalized)
    return normal

# 4. Regularization (Quan trọng)
# Phải ép _normal_delta nhỏ để nó không bị lạm dụng phá hỏng cấu trúc ellipsoid
# Trong train.py:
normal_delta_loss = (gaussians._normal_delta ** 2).mean()
loss = loss + opt.lambda_normal_delta_reg * normal_delta_loss
```

**Tác động:** ASG sẽ nhận được reflection direction chính xác hơn. Cần kết hợp với L1 Weighted Loss ở trên để normal delta nhận được gradient từ vùng specular.

---

## 3. Depth-Normal Consistency (Ưu tiên Thấp - Đòi hỏi độ phức tạp cao)

**Mục tiêu:** Normal của bề mặt phải là đạo hàm (gradient) của Depth map.

**Thiết kế:**
- Render Depth Map.
- Tính Pseudo-Normal từ Depth Map: $N_{depth} = \nabla Depth$.
- Ép Normal Map (render từ minimum axis) phải giống với $N_{depth}$.
- Loss: $L_{consistency} = 1 - \cos(N_{render}, N_{depth})$.

**Hạn chế:** Pseudo-normal từ depth thường rất nhiễu ở rìa vật thể. Implementation phức tạp, tốn chi phí render depth. Thường chỉ áp dụng ở các paper tập trung hoàn toàn vào Geometry (như 2DGS, SuGaR).

---

## Lộ trình Ablation (Nhóm N - Normal)

- **N0:** Baseline (Min-axis, không delta).
- **N1:** Learned Normal Delta + Regularization L2. (Nên thử nghiệm phương án 2 trước vì dễ code hơn phương án 1 - không cần đụng vào CUDA rasterizer).
- **N2 (Mở rộng):** Cập nhật ASG để nội suy Normal theo Depth/Smoothness (Chỉ cân nhắc nếu N1 chưa giải quyết triệt để).
