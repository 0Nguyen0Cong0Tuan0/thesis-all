# Thiết Kế Representation Capacity — Role Separation và ASG Capacity

Mục tiêu của trục này không phải chỉ là tăng số tham số cho ASG. Vấn đề cần giải là: **ASG có đủ năng lực biểu diễn highlight không, và nó có thật sự được giao phần specular để học không?**

Nếu SH vẫn được phép hấp thụ highlight ở vùng specular, tăng `asg_degree` hoặc tăng MLP rất dễ không tạo khác biệt rõ trong `only_asg`.

---

## 1. Những thiếu sót cần sửa trong draft cũ

### 1.1. Weighted loss không phải lời giải chính của Representation Capacity

Specular-weighted loss làm gradient ở vùng specular mạnh hơn, nhưng gradient đó vẫn chảy vào cả SH và ASG:

```python
loss = L1(SH_color + ASG_color, GT)
```

Nếu không có role separation, weighted loss có thể chỉ giúp SH học highlight nhanh hơn. Vì vậy weighted loss nên được xem là hỗ trợ thuộc trục **Supervision Signal**, không phải cơ chế chính của Representation Capacity.

### 1.2. `f_rest_interval_*` chỉ throttle theo thời gian, chưa throttle theo vùng

Code hiện tại tách optimizer cho `f_rest`, nhưng lịch update chỉ quyết định "iteration nào SH bậc cao được step". Nó không biết Gaussian nào overlap vùng specular.

Kết luận: cần thêm **spatial role separation**: Gaussian nào đang cover pixel ref_score cao thì giảm hoặc zero gradient `f_rest` ở Gaussian đó.

### 1.3. `ASG_DEGREE` chưa đo hết capacity

`asg_degree` là số chiều latent per Gaussian. Nó không trực tiếp đổi:

- số lobe ASG (`num_theta`, `num_phi`);
- hidden width của MLP;
- số layer MLP;
- việc real branch dùng `viewdir` hay `reflection direction`.

Do đó ablation `ASG_DEGREE=12/24/32` chỉ đo một phần nhỏ của capacity.

### 1.4. Real branch đang viewdir-only

`SpecularNetwork` synthetic dùng reflection direction qua normal. Nhưng `SpecularNetworkReal` hiện gọi REE bằng `viewdirs`, không dùng normal/reflection direction.

Điều này có thể giảm nhiễu do normal proxy kém, nhưng cũng làm branch real yếu hơn với highlight phụ thuộc hình học bề mặt. Thiết kế mới cần biến lựa chọn này thành flag rõ ràng thay vì là hành vi ẩn.

### 1.5. Sparse ASG không hỏng vì Gaussian mới, mà vì lệch visibility giữa camera

Khi densification làm số Gaussian thay đổi, code đã fallback sang full ASG vì `prev_vis_mask.shape[0] != n_gs`. Điểm yếu thật là `prev_vis_mask` thuộc camera trước, trong khi iteration hiện tại dùng camera ngẫu nhiên khác. Gaussian visible/specular trong camera hiện tại nhưng không visible ở camera trước sẽ render bằng SH-only trong iteration đó.

Đây là trade-off tốc độ/chất lượng, nên cần ablation bằng `full_asg_interval` hoặc một mode current-visible đắt hơn.

### 1.6. `lambda_spec_reg` từng là tham số chưa được dùng

Trong baseline trước patch, `lambda_spec_reg` tồn tại trong arguments, nhưng `train.py` chỉ dùng:

```python
loss = photometric_loss
```

Patch Representation Capacity đã nối nó vào loss và log rõ. Không nên đọc các run cũ như thể chúng đã có specular regularization.

Khi nối vào loss, default nên là `lambda_spec_reg = 0.0` để giữ baseline cũ. Giá trị `0.01` trước đây chỉ là tham số chết; bật nó ngay sau khi implement sẽ làm thay đổi kết quả ngầm và dễ gây nhiễu cho ablation.

---

## 2. Thiết kế pipeline mới

Thiết kế đề xuất gồm 4 khối, triển khai theo thứ tự từ ít rủi ro đến nhiều rủi ro.

Trạng thái implementation hiện tại:

- R1 `use_sh_spec_mask`: đã implement, mặc định tắt.
- R2 `lambda_spec_l1_weight`: đã implement, mặc định `0.0`.
- Phase 2b `lambda_spec_reg`: đã nối vào loss, mặc định `0.0`.
- R3 architecture knobs: đã expose `asg_num_theta`, `asg_num_phi`, `specular_hidden`, `specular_layers`, `real_use_reflection_dir`; default `-1/False` giữ kiến trúc cũ.
- R4 `full_asg_interval`: đã có sẵn từ trước, tiếp tục dùng làm ablation sparse ASG.

### R1 — Gaussian-Level SH Gradient Masking

**Mục tiêu**: ngăn `f_rest` học view-dependent high-frequency color tại Gaussian overlap vùng specular, để ASG nhận phần residual đó.

Cơ chế:

1. Với camera train hiện tại, tạo pixel mask từ `cam.ref_score`:

```python
spec_metric_map = (cam.ref_score.cuda() > opt.sh_spec_mask_threshold).reshape(-1).int()
```

2. Truyền mask này vào cùng render pass training bằng `metric_map` và `get_flag=True`.

3. Rasterizer trả về `accum_metric_counts`: Gaussian nào cover pixel specular sẽ có count > 0.

4. Sau `loss.backward()` và trước mọi optimizer `.step()`, scale gradient `f_rest` tại các Gaussian đó:

```python
features_rest.grad[spec_gaussian_mask] *= opt.sh_spec_grad_scale
```

Default đề xuất:

```python
use_sh_spec_mask = False
sh_spec_mask_threshold = 0.7
sh_spec_grad_scale = 0.0
sh_spec_mask_start = specular_start_iter
sh_spec_min_metric_count = 1
```

Ý nghĩa:

- `0.0`: block hoàn toàn `f_rest` ở vùng specular.
- `0.25`: cho SH học một phần nhỏ nếu block mạnh làm diffuse hỏng.
- Chỉ tác động `f_rest`, không đụng `f_dc`, geometry, opacity.

Giới hạn:

- Mask phải nằm ở đầu `GaussianModel.optimizer_step()`, trước mọi `.step()`. Nếu đặt trong `_step_f_rest_optimizer()` thì đúng với `optimizer_type=default`, nhưng sai/không có tác dụng với `sparse_adam` vì khi đó `f_rest` nằm chung trong `self.optimizer`.
- Với `sh_spec_grad_scale=0.0`, việc mask mỗi iteration là đúng mục tiêu POC: mọi gradient SH-rest ở Gaussian specular đều bị chặn trước khi tích lũy/step. Với scale trung gian như `0.25`, mask theo accumulation có thể mạnh hơn "scale từng contribution" tuyệt đối; vì vậy scale trung gian chỉ nên dùng như fallback ablation, không phải default.

### R2 — Specular-Weighted L1, nhưng normalized

**Mục tiêu**: tăng supervision tại pixel specular mà không làm scale loss toàn ảnh thay đổi quá mạnh.

Chỉ nên bật sau R1 hoặc bật cùng R1, vì nếu không SH vẫn có thể hấp thụ gradient tăng thêm.

Pseudocode:

```python
pixel_l1 = torch.abs(image - gt)  # [3, H, W]
ref_w = cam.ref_score.cuda().unsqueeze(0)  # [1, H, W]
weight = 1.0 + opt.lambda_spec_l1_weight * ref_w

# Normalize để mean loss không tăng chỉ vì tổng weight lớn hơn.
Ll1_weighted = (pixel_l1 * weight).sum() / (3.0 * weight.sum().clamp_min(1e-6))
```

Default đề xuất:

```python
lambda_spec_l1_weight = 0.0
```

Ablation đầu tiên:

```python
lambda_spec_l1_weight = 1.0
```

Không nên dùng 2.0 hoặc 3.0 ngay nếu chưa biết `NonSpec_PSNR` có giảm không.

### R3 — ASG Architecture Knobs Thật Sự

**Mục tiêu**: tách latent size khỏi lobe/MLP capacity để ablation có ý nghĩa.

Thêm các tham số model-level:

```python
asg_num_theta = -1      # -1 dùng default theo synthetic/real
asg_num_phi = -1
specular_hidden = -1
specular_layers = -1
real_use_reflection_dir = False
```

Default giữ tương thích:

| Scene mode | `num_theta` | `num_phi` | hidden | layers | direction |
|---|---:|---:|---:|---:|---|
| synthetic | 4 | 8 | 128 | 2 hidden layers | reflection direction |
| real outdoor | 2 | 4 | 32 | 1 hidden layer | viewdir |
| real indoor | 2 | 4 | 32 | 2 hidden layers | viewdir |

Ablation có ý nghĩa hơn:

- Giữ `asg_degree=24`, tăng real branch từ `2×4` lên `4×8`.
- Giữ lobe grid, tăng hidden `32 -> 64`.
- Bật `real_use_reflection_dir=True` để kiểm tra normal/reflection có giúp hay làm nhiễu.

Lưu ý checkpoint/render:

- Các tham số kiến trúc này phải nằm trong `ModelParams` để `cfg_args` lưu lại và `render.py` instantiate đúng architecture khi load `specular.pth`.

### R4 — Sparse ASG Refresh Có Kiểm Soát

**Mục tiêu**: giảm nhiễu do dùng visibility của camera trước.

MVP không cần viết mode mới ngay; repo đã có `full_asg_interval`.

Ablation đề xuất:

```bash
FULL_ASG_INTERVAL=0
FULL_ASG_INTERVAL=3000
FULL_ASG_INTERVAL=1000
```

Nếu `FULL_ASG_INTERVAL=3000` cải thiện `ASG_Residual_IoU` hoặc `Spec_PSNR`, lúc đó mới cân nhắc mode đắt hơn:

1. render SH-only current camera để lấy `radii/current_vis_mask`;
2. evaluate ASG cho current-visible Gaussians;
3. render full.

Mode này thêm một render pass/iteration nên chỉ nên làm nếu sparse visibility thật sự là bottleneck.

---

## 3. Phương án implementation

### Phase 1 — Role Separation MVP

Files cần sửa:

- `arguments/__init__.py`
- `train.py`
- `scene/gaussian_model.py`
- `run_spec-fastgs_big.sh`
- `run_shiny.sh`

Thêm arguments:

```python
self.use_sh_spec_mask = False
self.sh_spec_mask_threshold = 0.7
self.sh_spec_grad_scale = 0.0
self.sh_spec_mask_start = 3000
self.sh_spec_min_metric_count = 1
```

Trong `train.py`, trước render:

```python
collect_spec_mask = (
    opt.use_sh_spec_mask
    and iteration >= opt.sh_spec_mask_start
    and hasattr(cam, "ref_score")
)

metric_map = None
if collect_spec_mask:
    metric_map = (cam.ref_score.cuda() > opt.sh_spec_mask_threshold).flatten().int()
```

Trong render call:

```python
render_pkg = render_fastgs(
    cam,
    gaussians,
    pipe,
    background,
    opt.mult,
    mlp_color=mlp_color,
    get_flag=collect_spec_mask,
    metric_map=metric_map,
)
```

Sau render:

```python
spec_gaussian_mask = None
if collect_spec_mask:
    spec_gaussian_mask = (
        render_pkg["accum_metric_counts"].squeeze()
        >= opt.sh_spec_min_metric_count
    )
```

Trong `GaussianModel.optimizer_step()`, apply mask ngay đầu hàm:

```python
def optimizer_step(
    self,
    iteration,
    skip_sh=False,
    f_rest_grad_mask=None,
    f_rest_grad_scale=0.0,
):
    self._apply_f_rest_grad_mask(f_rest_grad_mask, f_rest_grad_scale)
    ...
```

Helper:

```python
if f_rest_grad_mask is not None and self._features_rest.grad is not None:
    if f_rest_grad_mask.shape[0] == self._features_rest.grad.shape[0]:
        self._features_rest.grad[f_rest_grad_mask] *= f_rest_grad_scale
```

Log thêm vào `train_info.json`:

```python
"use_sh_spec_mask": opt.use_sh_spec_mask,
"sh_spec_mask_threshold": opt.sh_spec_mask_threshold,
"sh_spec_grad_scale": opt.sh_spec_grad_scale,
"sh_spec_mask_start": opt.sh_spec_mask_start,
"sh_spec_min_metric_count": opt.sh_spec_min_metric_count,
```

Rủi ro chính:

- Nếu prior/ref_score false positive rộng, SH bị chặn ở vùng diffuse sáng.
- Nếu scale `0.0` quá mạnh, `NonSpec_PSNR` có thể giảm.

Fallback:

- Tăng threshold `0.7 -> 0.8`.
- Đổi `sh_spec_grad_scale = 0.25`.
- Chỉ bật sau `specular_start_iter + 1000`.

### Phase 2 — Weighted Specular L1

Files cần sửa:

- `arguments/__init__.py`
- `train.py`
- `train_info.json` metadata block
- scripts chạy ablation

Thêm:

```python
self.lambda_spec_l1_weight = 0.0
```

Trong loss:

```python
if (
    opt.lambda_spec_l1_weight > 0
    and hasattr(cam, "ref_score")
):
    pixel_l1 = torch.abs(image - gt)
    ref_w = cam.ref_score.cuda().unsqueeze(0)
    weight = 1.0 + opt.lambda_spec_l1_weight * ref_w
    Ll1 = (pixel_l1 * weight).sum() / (3.0 * weight.sum().clamp_min(1e-6))
else:
    Ll1 = l1_loss(image, gt)
```

SSIM giữ nguyên để tránh làm phức tạp stage đầu.

### Phase 2b — Optional Specular Regularization

Files cần sửa:

- `arguments/__init__.py`
- `train.py`
- `train_info.json` metadata block

Default:

```python
self.lambda_spec_reg = 0.0
```

Trong loss:

```python
loss = photometric_loss
if opt.lambda_spec_reg > 0 and spec_sparse is not None:
    spec_reg_loss = (spec_sparse ** 2).mean()
    loss = loss + opt.lambda_spec_reg * spec_reg_loss
```

Ý nghĩa:

- Đây là L2 penalty để chống ASG output quá lớn hoặc leak rộng.
- Nó không chống collapse về zero; ngược lại nếu quá mạnh có thể làm `only_asg` yếu đi.
- Chỉ nên bật sau khi R1 đã chứng minh ASG nhận đúng vai trò specular.

### Phase 3 — ASG Architecture Knobs

Files cần sửa:

- `arguments/__init__.py`
- `scene/specular_model.py`
- `utils/spec_utils.py`
- `train.py`
- `render.py`

Thiết kế API:

```python
specular_mlp = SpecularModel(
    dataset.asg_degree,
    dataset.is_real,
    dataset.is_indoor,
    asg_num_theta=dataset.asg_num_theta,
    asg_num_phi=dataset.asg_num_phi,
    specular_hidden=dataset.specular_hidden,
    specular_layers=dataset.specular_layers,
    real_use_reflection_dir=dataset.real_use_reflection_dir,
)
```

Cần đảm bảo:

- default architecture load được checkpoint cũ;
- architecture override được ghi trong `cfg_args` và `train_info.json`;
- `render.py` dùng cùng args khi load `specular.pth`.

### Phase 4 — Logging và Ablation Group

Thêm log runtime nhẹ:

- `asg_eval_count`: số Gaussian được ASG evaluate trung bình.
- `sh_spec_mask_ratio`: tỉ lệ Gaussian bị mask trong iteration có mask.
- `sh_spec_grad_scale`: scale đang dùng.
- `spec_reg_loss`: chỉ log khi `lambda_spec_reg > 0`.

Không nhất thiết log mỗi iteration vào file lớn; có thể chỉ EMA và ghi cuối vào `train_info.json`.

Cập nhật `ablation_ver.md` sau khi code xong:

```text
Nhóm R: Representation Capacity / Role Separation
R0: current default
R1: use_sh_spec_mask=True, scale=0.0, threshold=0.7
R2: use_sh_spec_mask=True, scale=0.25, threshold=0.7
R3: R1 + lambda_spec_l1_weight=1.0
R4: R1 + FULL_ASG_INTERVAL=3000
R5: ASG architecture override nếu R1-R4 cho tín hiệu tốt
```

---

## 4. Thứ tự chạy đề xuất

Chỉ chạy trên `counter/images_8` trước, cùng một `reflection_prior`.

### Bước 1 — kiểm tra root cause SH/ASG

```bash
USE_SH_SPEC_MASK=True
SH_SPEC_GRAD_SCALE=0.0
SH_SPEC_MASK_THRESHOLD=0.7
LAMBDA_SPEC_L1_WEIGHT=0.0
FULL_ASG_INTERVAL=0
```

Kỳ vọng tốt:

- `only_asg` rõ hơn;
- `ASG_Energy_In_Residual` tăng;
- `ASG_Residual_IoU` tăng;
- `NonSpec_PSNR` không giảm mạnh.

Nếu `Spec_PSNR` tăng nhưng `NonSpec_PSNR` giảm, chạy lại với:

```bash
SH_SPEC_GRAD_SCALE=0.25
SH_SPEC_MASK_THRESHOLD=0.8
```

### Bước 2 — thêm weighted loss nhẹ

```bash
USE_SH_SPEC_MASK=True
SH_SPEC_GRAD_SCALE=0.0
LAMBDA_SPEC_L1_WEIGHT=1.0
```

Chỉ giữ setting này nếu `Spec_PSNR` tăng mà `NonSpec_PSNR` và LPIPS không xấu rõ.

### Bước 3 — kiểm tra sparse ASG

```bash
FULL_ASG_INTERVAL=3000
```

Nếu tốt, cân nhắc default `3000` cho các run thesis quan trọng; nếu không khác biệt, giữ `0` để tiết kiệm thời gian.

### Bước 4 — mới ablate architecture

Chỉ làm sau khi R1/R2 chứng minh ASG thật sự nhận vai trò specular. Nếu role separation chưa có tác dụng, tăng architecture dễ chỉ tăng cost.

---

## 5. Kết luận thiết kế

Phương án ưu tiên là **R1 Gaussian-level SH Gradient Masking**. Đây là thay đổi giải quyết đúng bottleneck Representation Capacity hiện tại: không phải ASG không có tham số, mà là ASG chưa được bảo vệ khỏi SH ở vùng specular.

Weighted loss và ASG architecture knobs nên có trong kế hoạch, nhưng không nên là bước đầu tiên. Bước đầu tiên phải chứng minh rằng khi SH bị hạn chế đúng vùng, `only_asg` và các metric residual-specular có tăng hay không. Nếu có, các bước tăng supervision/capacity phía sau mới có cơ sở.
