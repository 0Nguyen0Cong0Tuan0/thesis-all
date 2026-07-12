# Đề Xuất Thiết Kế — Cải Thiện Geometric Coverage

> **Vấn đề gốc**: Hai điểm yếu trong cơ chế Ref Score guidance:
> 1. **Prior tĩnh** — `cam.ref_score` được tính một lần từ ảnh gốc, không cập nhật theo progress của model
> 2. **Budget hardcode** — `max_refscore_gaussians = 400_000` là hằng số toàn cục, không tổng quát

---

## Phần 1 — Prior Tĩnh, Không Adaptive

### Chẩn đoán vấn đề

Code hiện tại ([`train.py:L65-L92`](file:///home/baodinh/baodinh_thesis/spec-fastgs/train.py#L65-L92)):

```python
# Chỉ chạy MỘT LẦN khi bắt đầu training
if opt.use_ref_score and os.path.exists(ref_prior_dir):
    for cam in scene.getTrainCameras():
        ref_img = imageio.imread(npath)
        cam.ref_score = torch.tensor(ref_img / 255.0)
        # → CỐ ĐỊNH SUỐT 30K ITERATIONS
```

Hệ quả:
- Iteration 500 (model chưa ra gì): force densify vào vùng prior → hợp lý
- Iteration 14000 (model gần converge): vẫn force densify vào **đúng cùng prior** → có thể đã đủ Gaussian tại đó, hoặc prior sai chỗ so với nơi model thực sự đang thiếu
- Prior tính từ ảnh gốc: bắt được "vùng có khả năng là specular theo màu sắc" — không bắt được "vùng model đang thiếu specular nhất hiện tại"

---

### Đề Xuất A — Residual-Adaptive Prior

**Ý tưởng cốt lõi**: Thay vì dùng prior tĩnh từ ảnh gốc, định kỳ tính lại `cam.ref_score` từ **residual thực tế của model hiện tại**.

Residual `|GT - render|` cao tại vùng specular → đó là nơi model đang thiếu nhất → đây mới là prior thật cần densify.

#### Cơ chế hoạt động

```
Mỗi K iteration (ví dụ K=3000):
  Với mỗi camera trong tập train:
    1. Render ảnh hiện tại (no_grad)
    2. Tính per-pixel L1 residual: R = |GT - render|.mean(dim=0)
    3. Nhân với prior tĩnh gốc (intersection):
       adaptive = R * static_prior    ← chỉ update vùng cả hai đồng ý
    4. Normalize [0,1] và lưu vào cam.ref_score
```

**Tại sao intersection với static prior?**
- Static prior lọc false positive từ residual: model có thể có residual cao ở vùng diffuse khó (texture phức tạp), không phải specular → nhân với prior tĩnh (chỉ vùng bright+unsaturated) để lọc lại
- Không bỏ hoàn toàn static prior: nếu vùng specular đang được render tốt rồi, residual thấp → adaptive score thấp → không force densify thêm (đúng mong muốn)

#### Pseudocode

```python
# Thêm vào train.py, gọi mỗi opt.adaptive_prior_interval iteration

def update_adaptive_ref_scores(scene, gaussians, pipe, background, opt, iteration):
    """Cập nhật cam.ref_score từ residual hiện tại của model."""
    
    if not opt.use_ref_score or not opt.use_adaptive_prior:
        return
    if iteration < opt.adaptive_prior_start:   # Không cập nhật quá sớm
        return
    if iteration % opt.adaptive_prior_interval != 0:
        return

    train_cams = scene.getTrainCameras()
    
    with torch.no_grad():
        for cam in train_cams:
            if not hasattr(cam, 'ref_score_static'):
                continue  # Không có static prior → bỏ qua
            
            # 1. Render base/SH-only, nhất quán với compute_gaussian_score_fastgs().
            #    Không truyền mlp_color → residual phản ánh phần base model chưa giải thích được.
            render_img = render_fastgs(
                cam,
                gaussians,
                pipe,
                background,
                opt.mult,
                mlp_color=None
            )["render"]
            
            # 2. Per-pixel L1 residual [H, W]
            gt = cam.original_image.cuda()
            residual = torch.abs(render_img - gt).mean(dim=0)  # [H, W]
            
            # 3. Normalize residual [0, 1]
            r_max = residual.quantile(0.95)     # robust max (tránh outlier)
            residual_norm = (residual / (r_max + 1e-6)).clamp(0.0, 1.0)
            
            # 4. Intersection với static prior
            adaptive = residual_norm * cam.ref_score_static  # element-wise product
            
            # 5. Re-normalize
            a_max = adaptive.max()
            if a_max > 0:
                adaptive = adaptive / a_max
            
            # 6. Exponential moving average để tránh update quá đột ngột
            alpha = opt.adaptive_prior_ema    # ví dụ: 0.7
            cam.ref_score = alpha * cam.ref_score + (1 - alpha) * adaptive
```

#### Thay đổi cần thiết

**`train.py` — lưu static prior riêng khi load:**

```python
# Trong đoạn load reflection prior (train.py:L65-L92)
cam.ref_score = ref_tensor          # adaptive (sẽ được cập nhật)
cam.ref_score_static = ref_tensor.clone()  # tĩnh, tách memory, không bao giờ ghi đè
```

**`train.py` — gọi update trong training loop:**

```python
# Trong for iteration loop, sau bước OPTIMIZER STEP
update_adaptive_ref_scores(scene, gaussians, pipe, background, opt, iteration)
```

**`arguments/__init__.py` — thêm hyperparameters:**

```python
self.use_adaptive_prior = False          # Master flag
self.adaptive_prior_start = 5000        # Không update quá sớm (model chưa ổn định)
self.adaptive_prior_interval = 3000     # Cập nhật mỗi 3K iter: iter 5K, 8K, 11K, 14K
self.adaptive_prior_ema = 0.7           # Giữ 70% cũ, 30% mới → smooth transition
```

#### Phân tích rủi ro

| Rủi ro | Mức độ | Cách giảm thiểu |
|--------|--------|-----------------|
| Update quá sớm khi model chưa ổn định (residual nhiễu) | Cao | `adaptive_prior_start = 5000` — chờ geometry cơ bản hình thành |
| Residual cao ở vùng diffuse khó (không phải specular) | Trung bình | Intersection với static prior lọc false positive |
| Oscillation: cập nhật → densify → residual thay đổi → cập nhật lại theo hướng khác | Thấp | EMA `alpha=0.7` làm chậm thay đổi đủ để hội tụ |
| Chi phí thêm: render thêm train views mỗi 3K iter | Thấp-Trung bình | Sample `adaptive_prior_num_cameras=20` thay vì toàn bộ train set trong ablation đầu |

Ước lượng chi phí:

- Nếu update toàn bộ 100 train views của `toaster` mỗi 3000 iteration, từ iter 5000 đến 14000 sẽ có khoảng 4 lần update, tức khoảng 400 no-grad renders thêm.
- Nếu dùng `adaptive_prior_num_cameras=20`, chi phí còn khoảng 80 no-grad renders.
- Với `counter/images_8`, nếu 1 no-grad render khoảng 0.05s, update toàn bộ 100 views tốn khoảng 5s mỗi lần; 4 lần update thêm khoảng 20s trên một run khoảng 15 phút, tức cỡ 2%.

---

### So Sánh Phương Án

| Phương án | Độ phức tạp implement | Rủi ro regression | Tác động kỳ vọng |
|-----------|----------------------|-------------------|-----------------|
| **A (Residual-Adaptive)** | Trung bình (thêm ~50 dòng) | Thấp (EMA + late start) | Cao — prior phản ánh đúng chỗ model đang cần |
| Học prior từ NeRF/depth (complex) | Rất cao | Cao | Cao nhưng không thực tế với timeline |
| Tăng interval (không update, chỉ giảm force densify về sau) | Rất thấp | Rất thấp | Thấp — không giải quyết gốc rễ |

**→ Khuyến nghị: Phương án A**, bắt đầu với `adaptive_prior_interval=3000`, `adaptive_prior_start=5000`, `ema=0.7`.

---

## Phần 2 — Budget Hardcode 400K

### Chẩn đoán vấn đề

Code hiện tại ([`fast_utils.py:L90`](file:///home/baodinh/baodinh_thesis/spec-fastgs/utils/fast_utils.py#L90) + [`arguments/__init__.py:L94`](file:///home/baodinh/baodinh_thesis/spec-fastgs/arguments/__init__.py#L94)):

```python
# Cứng: 400,000 — bất kể scene lớn hay nhỏ
self.max_refscore_gaussians = 400000

# Điều kiện kích hoạt
if gaussians.get_xyz.shape[0] < args.max_refscore_gaussians:
    use_ref_score = True
```

Vấn đề với cách tiếp cận này:

1. **Không tổng quát theo scene size**: Scene `counter/images_8` (800×600) và `counter/images_4` (1600×1200) cần số Gaussian rất khác nhau để cover đủ. 400K có thể thừa với ảnh nhỏ hoặc thiếu với ảnh lớn.

2. **Không tổng quát theo độ phức tạp specular**: Scene có nhiều bề mặt reflective (toaster, kitchen) cần budget lớn hơn so với scene chỉ có vài điểm specular nhỏ.

3. **Budget không liên quan đến tình trạng vùng specular**: Có thể đang có 380K Gaussian nhưng toàn bộ đều ở vùng diffuse, vùng specular vẫn trống — 400K cap không phân biệt được điều này.

4. **Binary behavior**: Ngay khi vượt 400K → ref_score bị tắt hoàn toàn. Không có soft transition.

---

### Đề Xuất B1 — Scene-Relative Budget

**Ý tưởng**: Budget tỉ lệ với **số điểm COLMAP khởi tạo** (initial point cloud), vì đây là proxy tốt cho độ phức tạp hình học của scene.

```python
# arguments/__init__.py — thay thế max_refscore_gaussians
self.refscore_budget_multiplier = 10.0   # Budget = initial_points × multiplier
self.refscore_budget_min = 200_000       # Sàn tối thiểu
self.refscore_budget_max = 1_000_000     # Trần tối đa (tránh OOM)
```

```python
# train.py — tính budget sau khi scene được load
initial_gaussians = gaussians.get_xyz.shape[0]
opt.max_refscore_gaussians = int(
    initial_gaussians * opt.refscore_budget_multiplier
)
opt.max_refscore_gaussians = max(opt.max_refscore_gaussians, opt.refscore_budget_min)
opt.max_refscore_gaussians = min(opt.max_refscore_gaussians, opt.refscore_budget_max)
print(f"Ref Score budget: {opt.max_refscore_gaussians:,} Gaussians "
      f"(= {initial_gaussians:,} × {opt.refscore_budget_multiplier})")
```

**Logic**: Scene có 30K điểm COLMAP → budget 300K. Scene có 80K điểm COLMAP → budget 800K (capped at 1M). Đây là cách các paper 3DGS thường scale density target.

---

### Đề Xuất B2 — Density-Aware Throttle (Thay Binary bằng Soft)

**Ý tưởng**: Thay vì tắt hoàn toàn khi vượt budget, giảm dần **cường độ** của ref_score force theo ratio hiện tại/budget.

```python
# fast_utils.py — thay thế block use_ref_score

use_ref_score = False
ref_score_threshold = args.refscore_threshold_min

if (getattr(args, 'use_ref_score', False) 
        and hasattr(my_viewpoint_cam, 'ref_score')
        and iteration is not None 
        and not getattr(args, 'disable_ref_score', False)):
    
    if iteration % args.densification_refscore_interval == 0:
        n_current = gaussians.get_xyz.shape[0]
        n_budget  = args.max_refscore_gaussians
        
        if n_current < n_budget:
            # Soft decay: giảm dần khi tiến gần budget
            # strength = 1.0 khi n_current = 0
            # strength không thấp hơn refscore_min_strength khi vẫn còn dưới budget
            ratio = n_current / n_budget
            raw_strength = (1.0 - ratio) ** args.refscore_decay_power
            ref_score_strength = max(raw_strength, args.refscore_min_strength)
            ref_score_threshold = (
                args.refscore_threshold_min
                + (1.0 - ref_score_strength)
                * (args.refscore_threshold_max - args.refscore_threshold_min)
            )
            use_ref_score = True

if use_ref_score:
    # Khi strength thấp → chỉ vùng ref_score rất cao mới được force
    ref_mask = (my_viewpoint_cam.ref_score.cuda() > ref_score_threshold).int()
    metric_map = torch.max(metric_map, ref_mask)
```

**Hành vi của `ref_score_threshold` với default `decay_power=1.0`, `min_strength=0.15`:**

| `n_current / n_budget` | `ref_score_strength` | `ref_score_threshold` | Ý nghĩa |
|------------------------|-------------------------------|--------------------|----|
| 0% (mới bắt đầu) | 1.0 | 0.50 | Force densify rộng — mọi vùng ref>0.5 |
| 50% | 0.50 | 0.70 | Force densify vừa phải |
| 80% | 0.20 | 0.82 | Chỉ force vào vùng specular rõ |
| 95% | 0.15 | 0.84 | Clamp giữ ref-score không tắt sớm |
| 100%+ | 0.0 | tắt | Không force gì cả |

**Thêm hyperparameter:**

```python
# arguments/__init__.py
self.refscore_decay_power = 2.0  # 1.0=linear, 2.0=quadratic (giảm nhanh hơn khi gần budget)
```

---

### Đề Xuất B3 — Per-Region Budget (Nâng Cao)

**Ý tưởng**: Thay vì một budget toàn cục, theo dõi **mật độ Gaussian trong vùng ref_score** và chỉ dừng force khi mật độ đó đủ.

> [!WARNING]
> Phương án này phức tạp hơn đáng kể (cần spatial hashing hoặc kd-tree) và rủi ro cao hơn. Chỉ nên xét sau khi B1+B2 đã ổn định.

---

### Kết Hợp Đề Xuất B1 + B2

Hai đề xuất **bổ sung cho nhau** và nên được implement cùng:

```
B1 (Scene-Relative Budget): Xác định ĐÍCH ĐẾN phù hợp với từng scene
B2 (Soft Decay):            Kiểm soát CÁCH ĐI đến đích đó mượt mà hơn
```

---

## Tổng Hợp Thay Đổi Cần Implement

### `arguments/__init__.py`

```python
# Thay thế:
self.max_refscore_gaussians = 400000

# Bằng:
self.max_refscore_gaussians = -1           # -1 = auto từ scene size
self.refscore_budget_multiplier = 10.0    # budget = init_points × multiplier
self.refscore_budget_min = 200_000
self.refscore_budget_max = 1_000_000
self.refscore_decay_power = 1.0
self.refscore_min_strength = 0.15        # tránh tắt ref-score trước khi đầy budget
self.refscore_threshold_min = 0.5
self.refscore_threshold_max = 0.9

# Thêm mới cho adaptive prior:
self.use_adaptive_prior = False
self.adaptive_prior_start = 5000
self.adaptive_prior_interval = 3000
self.adaptive_prior_num_cameras = 20
self.adaptive_prior_ema = 0.7
```

### `train.py`

```python
# Sau khi scene được load (sau dòng initial_gaussians = ...):
if opt.max_refscore_gaussians == -1:
    opt.max_refscore_gaussians = int(
        initial_gaussians * opt.refscore_budget_multiplier
    )
    opt.max_refscore_gaussians = max(opt.max_refscore_gaussians, opt.refscore_budget_min)
    opt.max_refscore_gaussians = min(opt.max_refscore_gaussians, opt.refscore_budget_max)
    print(f"[Auto Budget] Ref Score cap: {opt.max_refscore_gaussians:,}")

# Khi load prior: lưu thêm ref_score_static
cam.ref_score = ref_tensor
cam.ref_score_static = ref_tensor.clone()

# Trong training loop: gọi adaptive update
update_adaptive_ref_scores(scene, gaussians, pipe, background, opt, iteration)
```

### `utils/fast_utils.py`

```python
# Thay block use_ref_score bằng soft decay logic (xem Đề Xuất B2)
```

---

## Thứ Tự Ưu Tiên Implement

```
Ưu tiên 1 (Low risk, high value):
  → B1: Scene-Relative Budget
    - Chỉ sửa arguments/__init__.py và 3 dòng trong train.py
    - Không thay đổi logic densification
    - Dễ test: chạy cùng scene với budget mới vs 400K cũ

Ưu tiên 2 (Medium risk, high value):
  → B2: Soft Decay Threshold
    - Sửa fast_utils.py ~10 dòng
    - Cần ablation: decay_power=1 vs 2 vs không có

Ưu tiên 3 (Medium risk, medium value):
  → A: Residual-Adaptive Prior
    - Cần thêm hàm mới ~50 dòng
    - Cần ablation riêng: adaptive_prior_interval và ema
    - Nên chạy sau khi B1+B2 đã ổn định để tách biệt tác động
```

---

## Tóm Tắt So Sánh

| | Hiện tại | Sau cải tiến |
|---|---|---|
| **Prior** | Tĩnh, tính từ ảnh gốc, không đổi suốt 30K iter | Adaptive, cập nhật từ residual thực tế mỗi 3K iter từ iter 5K trở đi |
| **Budget cap** | 400K hardcode, binary on/off | Tự động theo `init_points × 10`, có min/max safeguard |
| **Behavior khi gần budget** | Tắt đột ngột | Giảm dần: chỉ force vùng specular rõ hơn (threshold 0.5 → khoảng 0.84 với default clamp) |
| **Tổng quát hóa** | Chỉ tốt với scenes có ~40K COLMAP points | Tự scale với bất kỳ scene nào |
