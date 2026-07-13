# Ref Score trong Spec-FastGS

## 1. Mục Đích

`RefScore` là bản đồ prior hai chiều được trích xuất từ ảnh đầu vào để chỉ ra các vùng có khả năng chứa phản xạ gương, highlight hoặc vật liệu phản quang. Trong Spec-FastGS, `RefScore` không được xem là ground truth specular mask tuyệt đối. Nó là một tín hiệu định hướng giúp pipeline biết vùng nào cần được ưu tiên trong quá trình khởi tạo, loang Gaussian, cập nhật prior và SH spec mask nhẹ.

Mục đích chính của `RefScore` là giải quyết vấn đề Geometric Coverage: các vùng specular thường nhỏ, sáng cục bộ và thay đổi mạnh theo góc nhìn, nên cơ chế densification dựa thuần trên gradient hoặc photometric error của FastGS/3DGS có thể không sinh đủ Gaussian tại đúng vị trí. Nếu vùng phản quang không có đủ Gaussian, việc dùng ASG hay MLP specular tốt hơn cũng khó cải thiện visual quality, vì model thiếu hạt để biểu diễn vùng đó.

Sau các thí nghiệm Representation Capacity, Supervision Signal, Normal Quality và ASG residual supervision, pipeline cuối cùng tách `RefScore` thành hai vai trò:

- `ref_score`: prior rộng, dùng cho Geometry Coverage và densification.
- `ref_score_conf`: confidence bảo thủ hơn, dùng cho SH spec mask nhẹ trong Representation Capacity.

Thiết kế này tránh lỗi trước đây: dùng cùng một bản đồ pixel-level prior vừa để sinh thêm Gaussian, vừa làm mask gần giống ground truth specular. Một prior rộng có thể tốt cho coverage, nhưng quá nhiễu để ép loss hoặc chặn gradient.

Ở cấu hình đề xuất cuối cùng, `ref_score_conf` chỉ còn được dùng cho SH spec mask nhẹ. Các nhánh Supervision Signal, `real_use_reflection_dir`, Normal Quality và `use_asg_residual_supervision` đã được khảo sát như ablation nhưng không còn thuộc final pipeline vì không cải thiện metrics/visual quality tốt nhất.

## 2. Mục Tiêu Thiết Kế

RefScore được thiết kế với các mục tiêu sau:

1. Tăng độ phủ hình học tại vùng phản quang: những vùng có highlight hoặc phản xạ cần được đưa vào metric map của FastGS để có cơ hội sinh thêm Gaussian.
2. Giữ pipeline tương thích với FastGS: RefScore chỉ bổ sung tín hiệu vào cơ chế ADC/densification hiện có, không thay thế hoàn toàn FastGS.
3. Hạn chế over-densification: có budget, decay và threshold động để RefScore không sinh Gaussian vô hạn.
4. Cho phép adaptive feedback: prior ban đầu có thể được cập nhật dựa trên residual giữa render hiện tại và ảnh ground truth.
5. Tách broad prior và conservative confidence: `ref_score` phục vụ coverage, `ref_score_conf` phục vụ mask SH spec có precision cao hơn.
6. Hỗ trợ ablation rõ ràng: các thành phần còn trong pipeline có flag để bật/tắt hoặc điều chỉnh mức tác động; các nhánh không hiệu quả được giữ lại trong ghi chú thí nghiệm như negative evidence, không còn là knob mặc định.

## 3. Vai Trò Trong Pipeline

### 3.1. Vai trò với Geometry Coverage

Ở trục Geometry Coverage, `ref_score` giúp pipeline trả lời câu hỏi: vùng nào dù photometric error hiện tại chưa cao vẫn nên được xem là quan trọng vì có khả năng là vùng phản quang?

Trong `utils/fast_utils.py`, FastGS vốn tạo `metric_map` từ photometric error:

```python
metric_map = (l1_loss_norm > args.loss_thresh).int()
```

Spec-FastGS bổ sung RefScore bằng phép OR mềm ở cấp pixel:

```python
ref_mask = (my_viewpoint_cam.ref_score.cuda() > ref_score_threshold).int()
metric_map = torch.max(metric_map, ref_mask)
```

Nghĩa là một pixel được chọn nếu nó có lỗi ảnh cao hoặc có `ref_score` đủ cao. RefScore không trực tiếp tạo Gaussian; nó chỉ làm cho rasterizer tích lũy nhiều `accum_metric_counts` hơn tại các Gaussian chiếu lên vùng đó. Sau đó cơ chế densify/prune của FastGS vẫn quyết định Gaussian nào được sinh thêm hoặc giữ lại.

### 3.2. Vai trò với Representation Capacity

Representation Capacity cần phân vai giữa SH và ASG: SH nên biểu diễn phần nền/base color, ASG nên học phần view-dependent/specular. Tuy nhiên nếu dùng `ref_score` rộng để cấm SH, false positive sẽ làm SH bị kìm ở cả vùng không thật sự specular, kéo giảm PSNR tổng thể.

Vì vậy pipeline mới dùng `ref_score_conf` cho SH spec mask:

```python
ref_conf = get_ref_score_confidence(cam, opt)
sh_spec_metric_map = (ref_conf > opt.sh_spec_mask_threshold).reshape(-1).int()
```

Bản đồ pixel này được rasterizer project về Gaussian thông qua `accum_metric_counts`. Gaussian nào thường xuyên chiếu vào vùng confidence cao sẽ nhận `sh_spec_grad_mask`, rồi optimizer scale gradient SH bằng `SH_SPEC_GRAD_SCALE`.

### 3.3. Supervision Signal đã loại bỏ

Supervision Signal từng gồm các cơ chế như specular-weighted L1, specular regularization và ASG residual supervision. Chúng được thiết kế với ý tưởng dùng `ref_score_conf` làm vùng tin cậy hơn để tăng áp lực học ở vùng phản quang. Tuy nhiên kết quả ablation cho thấy các loss phụ này không tạo ra cấu hình tốt nhất: PSNR/Spec_PSNR không vượt được pipeline chỉ dùng Geometry Coverage + SH spec mask nhẹ.

Vì vậy trong final pipeline:

```bash
LAMBDA_SPEC_L1_WEIGHT=0.0
LAMBDA_SPEC_REG=0.0
```

ASG residual supervision đã bị loại khỏi source pipeline. Kết quả R042/R043/R053/R054 cho thấy loss này có thể làm ASG tham gia mạnh hơn hoặc tăng ASG IoU trong một số run, nhưng không cải thiện PSNR/Spec_PSNR tốt nhất so với SH-mask-only R052. Do đó Supervision Signal được xem là negative ablation, không phải thành phần của pipeline hoàn thiện.

### 3.4. Các nhánh đã loại bỏ khỏi final pipeline

Một số nhánh từng dùng hoặc liên quan tới `ref_score_conf`, nhưng đã được loại khỏi pipeline cuối:

- `real_use_reflection_dir`: đổi ASG real branch từ view direction sang reflection direction dựa trên normal.
- Normal Quality: learned normal delta và normal smoothness.
- Supervision Signal: specular-weighted L1, specular regularization và ASG residual supervision.

Kết quả thực nghiệm cho thấy các nhánh này có tác động thật nhưng không tạo ra final metrics tốt hơn. `real_use_reflection_dir=True` làm Spec_PSNR/PSNR giảm vì normal/geometry chưa đủ đáng tin. Normal Quality học được delta trong một số run nhưng không bù được tác hại của reflection direction. Supervision Signal làm ASG energy hoặc IoU tăng trong vài cấu hình nhưng không chuyển thành Spec_PSNR/PSNR tốt hơn.

Vì vậy tài liệu này vẫn ghi nhận chúng như negative evidence, nhưng không xem chúng là thành phần hoàn thiện pipeline.

## 4. Cơ Chế Tạo RefScore

File chính: `extract_reflection_prior.py`.

Input của extractor là ảnh training view sau khi scene loader đã đọc dataset. Mỗi camera tạo ra một hoặc hai file trong:

```text
<dataset>/<scene>/reflection_prior/
```

Với mỗi ảnh `image_name`, pipeline lưu:

```text
image_name_ref_score.png
image_name_ref_conf.png
```

Trong đó:

- `*_ref_score.png`: bản đồ prior rộng, giữ vai trò coverage.
- `*_ref_conf.png`: bản đồ confidence đã hậu xử lý, giữ vai trò mask bảo thủ cho Representation Capacity.

### 4.1. Tan-Ikeuchi style prior

Phương pháp `tan` dựa trên giả định đơn giản: highlight thường sáng và có thành phần màu tối thiểu `Imin` cao.

Với ảnh RGB chuẩn hóa về `[0, 1]`:

```python
Imin = img01.min(axis=-1)
Imax = img01.max(axis=-1)
score = Imin
mask = (score > ti_thresh) & (Imax > ti_bright)
final_score = where(mask, score, 0)
```

Ưu điểm:

- Đơn giản, nhanh.
- Hoạt động tốt với vùng highlight trắng/sáng.
- Là hành vi cũ của pipeline, phù hợp để tái lập baseline tốt như R025.

Nhược điểm:

- Dễ nhầm nền trắng, diffuse sáng hoặc vùng overexposed thành specular.
- Không dùng thông tin hình học hay đa view.

### 4.2. Shafer/Klinker style prior

Phương pháp `shafer` dựa trên dichromatic reflection model: thành phần specular thường sáng và ít bão hòa màu hơn.

```python
Imax = img01.max(axis=-1)
Imin = img01.min(axis=-1)
saturation = 1.0 - Imin / (Imax + 1e-6)
mask = (Imax > sk_intensity) & (saturation < sk_saturation)
score = Imax * (1.0 - saturation)
```

Ưu điểm:

- Có thêm tiêu chí saturation.
- Ít bắt nhầm vật thể màu đậm hơn Tan-Ikeuchi.

Nhược điểm:

- Vẫn là pixel-level heuristic.
- Có thể bỏ sót phản xạ màu hoặc highlight không trắng.

### 4.3. Hybrid confidence prior

Phương pháp `hybrid` được bổ sung để tạo confidence mềm hơn bằng cách kết hợp nhiều cue:

```python
tan_soft = soft_step(Imin, ti_thresh) * soft_step(Imax, ti_bright)
shafer_soft = soft_step(Imax, sk_intensity) * soft_step(sk_saturation - saturation, 0)
local_highlight = normalize(max(Imax - local_mean(Imax), 0))
gray_bright = normalize(Imax * (1 - saturation))

score = 0.35 * tan_soft
      + 0.35 * shafer_soft
      + 0.20 * gray_bright
      + 0.10 * local_highlight
```

Ý tưởng của `hybrid`:

- Tan cue bắt highlight sáng.
- Shafer cue bắt vùng sáng ít bão hòa.
- Gray-bright cue giữ thông tin vùng trắng/xám sáng.
- Local contrast cue ưu tiên vùng sáng nổi bật so với lân cận.

`hybrid` vẫn không phải ground truth specular mask. Nó chỉ là prior mềm hơn, phù hợp hơn khi muốn dùng làm `ref_conf`.

## 5. Hậu Xử Lý Confidence

Sau khi có raw score, pipeline có thể tạo `ref_conf` bằng:

```python
ref_conf = postprocess_score(
    final_score,
    gamma=args.ref_conf_gamma,
    quantile=args.ref_conf_quantile,
    smooth_radius=args.ref_conf_smooth_radius,
)
```

Các tham số:

- `REF_CONF_GAMMA`: gamma tại bước extraction. Giá trị lớn hơn 1 làm confidence sắc hơn, giảm vùng điểm trung bình.
- `REF_CONF_QUANTILE`: chỉ giữ phần score cao hơn một phân vị. Ví dụ `0.85` nghĩa là ưu tiên top 15% pixel theo score.
- `REF_CONF_SMOOTH_RADIUS`: làm mượt box blur trước khi normalize/threshold, giúp giảm nhiễu pixel đơn lẻ.

Quan trọng:

- `REF_CONF_*` tác động khi chạy `extract_reflection_prior.py` và tạo file `*_ref_conf.png`.
- `REFSCORE_CONF_*` trong `train.py` chỉ dùng khi không có file `*_ref_conf.png`; khi file confidence đã tồn tại, train ưu tiên load file đó.
- Nếu muốn baseline cũ không đổi, giữ `REF_CONF_GAMMA=1.0`, `REF_CONF_QUANTILE=0.0`, `REF_CONF_SMOOTH_RADIUS=0`.
- Nếu muốn ablation các nhánh Representation/Supervision/Normal với confidence bảo thủ, nên tạo lại prior với `REF_CONF_QUANTILE` và `REF_CONF_GAMMA` khác identity.

## 6. Cách Load RefScore Trong Train

Trong `train.py`, khi `--use_ref_score` bật và thư mục `reflection_prior` tồn tại, mỗi camera sẽ load:

```python
cam.ref_score = image_name_ref_score.png
cam.ref_score_static = cam.ref_score.clone()
```

Nếu có confidence map:

```python
cam.ref_score_conf = image_name_ref_conf.png
```

Nếu không có, train tự sinh confidence từ `ref_score`:

```python
cam.ref_score_conf = build_ref_score_confidence(ref_tensor, opt)
```

`ref_score_static` và `ref_score_conf_static` được clone riêng để Adaptive Prior có thể update `cam.ref_score` và `cam.ref_score_conf` mà không phá bản prior gốc.

## 7. RefScore Budget và Soft Decay

RefScore không được phép điều khiển densification vô hạn. Hàm `configure_refscore_budget()` thiết lập budget:

```python
if max_refscore_gaussians == -1:
    budget = initial_gaussians * refscore_budget_multiplier
    budget = clamp(budget, refscore_budget_min, refscore_budget_max)
```

Các tham số:

- `MAX_REFSCORE_GAUSSIANS=-1`: tự tính budget theo số Gaussian ban đầu.
- `REFSCORE_BUDGET_MULTIPLIER`: nhân với số Gaussian ban đầu.
- `REFSCORE_BUDGET_MIN`: sàn budget.
- `REFSCORE_BUDGET_MAX`: trần budget.

Trong densification, strength của RefScore giảm dần khi số Gaussian tiến gần budget:

```python
ratio = n_current / n_budget
strength = max((1 - ratio) ** refscore_decay_power, refscore_min_strength)
ref_score_threshold = threshold_min + (1 - strength) * (threshold_max - threshold_min)
```

Ý nghĩa:

- Khi scene còn ít Gaussian, threshold thấp hơn, RefScore can thiệp mạnh hơn.
- Khi số Gaussian gần budget, threshold cao hơn, chỉ vùng RefScore mạnh nhất còn được ưu tiên.
- `refscore_min_strength` giữ lại một mức tác động tối thiểu để prior không biến mất hoàn toàn quá sớm.

## 8. Adaptive Prior

Adaptive Prior cập nhật RefScore bằng residual giữa render base/SH-only và ground truth:

```python
render_img = render_fastgs(..., mlp_color=None)["render"]
residual = abs(render_img - gt).mean(dim=0)
residual_norm = residual / quantile_95(residual)
adaptive = residual_norm * cam.ref_score_static
cam.ref_score = ema(cam.ref_score, adaptive)
```

Với confidence:

```python
adaptive_conf = residual_norm * cam.ref_score_conf_static
cam.ref_score_conf = ema(cam.ref_score_conf, adaptive_conf)
```

Mục tiêu:

- Giữ prior gốc làm vùng tìm kiếm.
- Tăng trọng số ở nơi model hiện tại vẫn chưa giải thích tốt.
- Giảm việc ép densification ở vùng đã được model reconstruct ổn.

Các flag:

- `USE_ADAPTIVE_PRIOR=True/False`
- `ADAPTIVE_PRIOR_START`
- `ADAPTIVE_PRIOR_INTERVAL`
- `ADAPTIVE_PRIOR_NUM_CAMERAS`
- `ADAPTIVE_PRIOR_EMA`

## 9. Framework Luồng Hoạt Động

Luồng tổng quát:

```text
Ảnh training
   |
   v
extract_reflection_prior.py
   |
   +-- tan / shafer / hybrid raw score
   |
   +-- normalize
   |      |
   |      +--> *_ref_score.png  (broad prior)
   |
   +-- postprocess_score(gamma, quantile, smooth)
          |
          +--> *_ref_conf.png   (conservative confidence)

train.py
   |
   +-- load ref_score vào camera
   +-- load/refine ref_score_conf vào camera
   |
   +-- Geometry Coverage
   |      |
   |      +-- compute_gaussian_score_fastgs()
   |      +-- metric_map = photometric_error OR ref_score_mask
   |      +-- rasterizer accum_metric_counts
   |      +-- FastGS densify/prune
   |
   +-- Adaptive Prior
   |      |
   |      +-- render base/SH-only
   |      +-- residual with GT
   |      +-- EMA update ref_score/ref_score_conf
   |
   +-- Representation Capacity
          |
          +-- use ref_score_conf
          +-- SH spec mask
```

## 10. Các Flag Liên Quan

### 10.1. Extraction flags

```bash
EXTRACT_REF_PRIOR=True
BACKUP_REF_PRIOR=True
REF_PRIOR_METHOD=tan        # tan | shafer | hybrid
TI_THRESH=0.35
TI_BRIGHT=0.6
SK_INTENSITY=0.65
SK_SATURATION=0.3
REF_CONF_GAMMA=1.0
REF_CONF_QUANTILE=0.0
REF_CONF_SMOOTH_RADIUS=0
```

### 10.2. Geometry Coverage flags

```bash
USE_REF_SCORE=True
USE_ADAPTIVE_PRIOR=True
MAX_REFSCORE_GAUSSIANS=-1
REFSCORE_BUDGET_MULTIPLIER=10.0
REFSCORE_BUDGET_MIN=200000
REFSCORE_BUDGET_MAX=1000000
REFSCORE_DECAY_POWER=1.0
REFSCORE_MIN_STRENGTH=0.15
REFSCORE_THRESHOLD_MIN=0.5
REFSCORE_THRESHOLD_MAX=0.9
```

### 10.3. Train-side confidence fallback flags

```bash
REFSCORE_CONF_QUANTILE=0.85
REFSCORE_CONF_GAMMA=1.5
REFSCORE_CONF_MIN=0.0
```

Các flag này chỉ có tác dụng khi train không tìm thấy `*_ref_conf.png`.

### 10.4. Nhánh dùng `ref_score_conf` trong pipeline final

```bash
USE_SH_SPEC_MASK=True/False
SH_SPEC_MASK_THRESHOLD=0.75
SH_SPEC_GRAD_SCALE=0.75
SH_SPEC_MASK_START=8000
SH_SPEC_MIN_METRIC_COUNT=2
```

Trong cấu hình đề xuất cuối cùng:

```bash
USE_SH_SPEC_MASK=True
SH_SPEC_GRAD_SCALE=0.75
SH_SPEC_MASK_START=8000
SH_SPEC_MASK_THRESHOLD=0.75
SH_SPEC_MIN_METRIC_COUNT=2
```

Supervision Signal đã tắt trong final pipeline:

```bash
LAMBDA_SPEC_L1_WEIGHT=0.0
LAMBDA_SPEC_REG=0.0
```

Các flag `USE_ASG_RESIDUAL_SUPERVISION`, `REAL_USE_REFLECTION_DIR`, `USE_NORMAL_DELTA` và `NORMAL_SMOOTH_USE_REF_MASK` đã bị loại khỏi source pipeline cuối. Chúng chỉ còn xuất hiện trong run notes như lịch sử ablation.

## 11. Cách Áp Dụng Trong Ablation

### 11.1. Baseline gần R025

Giữ prior cũ và tắt các nhánh phụ:

```bash
REF_PRIOR_METHOD=tan
REF_CONF_GAMMA=1.0
REF_CONF_QUANTILE=0.0
REF_CONF_SMOOTH_RADIUS=0

USE_REF_SCORE=True
USE_ADAPTIVE_PRIOR=True

USE_SH_SPEC_MASK=False
LAMBDA_SPEC_L1_WEIGHT=0.0
LAMBDA_SPEC_REG=0.0
```

Mục tiêu: kiểm tra lại pipeline tốt nhất theo hướng Geometry Coverage mà không bị nhiễu bởi Representation/Supervision/Normal.

### 11.2. Final pipeline đề xuất sau R052

Giữ Geometry Coverage và bật SH spec mask nhẹ:

```bash
REF_PRIOR_METHOD=tan
REF_CONF_GAMMA=1.0
REF_CONF_QUANTILE=0.0
REF_CONF_SMOOTH_RADIUS=0

USE_REF_SCORE=True
USE_ADAPTIVE_PRIOR=True

USE_SH_SPEC_MASK=True
SH_SPEC_GRAD_SCALE=0.75
SH_SPEC_MASK_START=8000
SH_SPEC_MASK_THRESHOLD=0.75
SH_SPEC_MIN_METRIC_COUNT=2

LAMBDA_SPEC_L1_WEIGHT=0.0
LAMBDA_SPEC_REG=0.0
```

Mục tiêu: chốt pipeline cân bằng hơn cho khóa luận. R052 giảm PSNR tổng thể rất nhẹ so với R025 nhưng tăng Spec_PSNR và ASG alignment proxy, phù hợp hơn với mục tiêu cải thiện phục dựng vùng phản quang.

### 11.3. Ablation prior method

Chỉ đổi extractor:

```bash
REF_PRIOR_METHOD=hybrid
REF_CONF_GAMMA=1.0
REF_CONF_QUANTILE=0.0
REF_CONF_SMOOTH_RADIUS=0
```

Mục tiêu: kiểm tra `hybrid` có làm broad RefScore tốt hơn `tan` không, trong khi chưa bật các nhánh phụ.

### 11.4. Ablation conservative confidence

Tạo lại `*_ref_conf.png` bảo thủ hơn:

```bash
REF_PRIOR_METHOD=tan        # hoặc hybrid
REF_CONF_GAMMA=1.5
REF_CONF_QUANTILE=0.85
REF_CONF_SMOOTH_RADIUS=0
```

Sau đó bật SH spec mask:

```bash
USE_SH_SPEC_MASK=True
```

Mục tiêu: kiểm tra liệu SH spec mask có cải thiện khi dùng confidence map hẹp hơn không. Không dùng lại Supervision Signal, ASG residual supervision hoặc Normal Quality trong pipeline cuối vì các hướng này đã có kết luận âm.

## 12. Giới Hạn Hiện Tại

RefScore hiện tại vẫn là image-space heuristic. Nó chưa phải mask specular ground truth và chưa hiểu đầy đủ hình học đa view. Vì vậy:

- Nó có thể bắt nhầm diffuse sáng, nền trắng, vùng overexposed.
- Nó có thể bỏ sót reflection màu, highlight tối hoặc phản xạ trải rộng.
- Nó chưa phân biệt vùng specular thật với vùng model đang reconstruct kém vì lý do khác.
- `ref_score_conf` chỉ giảm false positive bằng hậu xử lý score, chưa giải quyết tận gốc bằng nhận diện vật lý/hình học.

Kết luận thực dụng là: `RefScore` đã có bằng chứng tốt nhất khi dùng cho Geometric Coverage, đặc biệt khi kết hợp budget và Adaptive Prior. Khi dùng cho Representation Capacity, `ref_score_conf` giúp SH spec mask nhẹ trở nên ổn định hơn. Tuy nhiên `ref_score_conf` không đủ để cứu Supervision Signal, ASG residual supervision hoặc Normal Quality đã bị loại bỏ; mask/concept target của các nhánh đó vẫn chưa phù hợp với pipeline hiện tại.

## 13. Hướng Hoàn Thiện

Các hướng tiếp theo nếu muốn cải thiện RefScore:

1. Multi-view consistency: chỉ giữ vùng specular nếu điểm 3D tương ứng được nhiều view xác nhận.
2. Separate broad/confidence ngay từ thuật toán: `ref_score` ưu tiên recall, `ref_conf` ưu tiên precision.
3. Learned specular segmentation: dùng model segmentation hoặc classifier nhỏ để tạo specular confidence map.
4. Geometry-aware filtering: loại vùng nền/trắng/diffuse sáng bằng depth, normal hoặc consistency theo projection.
5. Residual-aware confidence: kết hợp prior ảnh gốc với residual trong quá trình train nhưng có cơ chế decay khi vùng đã reconstruct tốt.

Trong phạm vi pipeline hiện tại, thiết kế `ref_score + ref_score_conf` là bước sửa quan trọng nhất để Geometric Coverage và Representation Capacity không dùng chung một tín hiệu với giả định sai. Pipeline cuối giữ `ref_score` cho coverage và `ref_score_conf` cho SH spec mask nhẹ; các hướng Supervision Signal, ASG residual supervision, real reflection direction và Normal Quality được xem là negative ablations thay vì thành phần đề xuất.
