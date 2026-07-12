# Feedback — Geometric Coverage

## Đánh giá tổng quan

Hướng cải thiện **Geometric Coverage** trong `Geometric Coverage_implementation_plan.md` là hợp lý và phù hợp với mục tiêu hiện tại của pipeline `spec-fastgs`.

Lý do chính: với bài toán specular, vấn đề không chỉ nằm ở MLP/ASG có đủ khả năng biểu diễn hay không, mà trước hết phải có đủ Gaussian tại đúng vùng highlight. Nếu vùng specular không được cover đủ về hình học, ASG dù mạnh hơn cũng không có đủ primitive để render highlight sắc và đúng vị trí.

Pipeline hiện tại đã có cơ chế `ref_score` guidance cho densification, nhưng vẫn còn hai điểm yếu thật:

- `cam.ref_score` được load một lần từ `reflection_prior` rồi giữ tĩnh trong suốt training.
- `max_refscore_gaussians = 400000` là một budget cứng, không scale theo độ phức tạp của từng scene.

Vì vậy, các hướng đề xuất như **Scene-Relative Budget**, **Soft Decay**, và **Residual-Adaptive Prior** đều có cơ sở nghiên cứu tốt. Tuy nhiên, không nên implement nguyên xi toàn bộ ngay lập tức; nên chia theo mức rủi ro và kiểm chứng bằng ablation.

## Mức độ phù hợp với pipeline hiện tại

Phương án này phù hợp với pipeline hiện tại vì `spec-fastgs` đã có sẵn đường đi cho ref-score guidance:

- `train.py` load `reflection_prior` vào `cam.ref_score`.
- `compute_gaussian_score_fastgs()` dùng `cam.ref_score` để mở rộng `metric_map`.
- `densify_and_prune_fastgs()` dùng `importance_score` để quyết định clone/split Gaussian.

Tuy nhiên, cần hiểu đúng rằng `ref_score` không trực tiếp bảo đảm sinh Gaussian tại mọi pixel specular. Nó chỉ OR vào `metric_map`, sau đó đi qua `importance_score`, rồi vẫn bị gate bởi gradient và scale rules trong FastGS densification. Nói cách khác, ref-score hiện tại là **guidance cho ADC/FastGS**, không phải cơ chế spawn Gaussian độc lập.

Vì vậy, mọi chỉnh sửa trên trục Geometry Coverage nên giữ nguyên tinh thần này: cải thiện signal guidance cho FastGS, thay vì thay hoàn toàn logic densification.

## Nhận xét từng đề xuất

### B1 — Scene-Relative Budget

Đây là phần nên implement đầu tiên.

Thay `max_refscore_gaussians = 400000` bằng budget tương đối theo số Gaussian khởi tạo là hợp lý hơn, vì mỗi scene có độ phức tạp khác nhau. Ví dụ `counter` và `toaster` không nên dùng cùng một cap tuyệt đối nếu số điểm khởi tạo, kích thước ảnh, và mức độ specular khác nhau.

Tác động kỳ vọng:

- Ít rủi ro regression.
- Giữ nguyên logic densification hiện tại.
- Dễ ablation với cùng một scene.
- Có thể làm số Gaussian cuối tăng hoặc giảm tùy scene.
- Nếu budget cao hơn 400K, VRAM và thời gian training có thể tăng.

Kết luận: **nên làm trước**.

### B2 — Soft Decay Threshold

Ý tưởng thay behavior bật/tắt nhị phân bằng decay mềm là hợp lý. Hiện tại, khi số Gaussian vượt budget, ref-score guidance bị tắt hoàn toàn. Điều này có thể gây chuyển trạng thái quá đột ngột.

Tuy nhiên, công thức decay trong plan cần cẩn thận. Nếu `ref_score_strength` giảm quá nhanh, ref-score có thể gần như mất tác dụng khi training vẫn còn cần densify vùng specular. Ví dụ nếu ratio gần budget mà strength chỉ còn khoảng 0.04, dynamic threshold sẽ rất cao và chỉ còn vùng prior cực mạnh mới được dùng.

Giải pháp cụ thể nên dùng là thêm **minimum strength clamp**:

```python
raw_strength = (1.0 - ratio) ** refscore_decay_power
ref_score_strength = max(raw_strength, refscore_min_strength)
```

Default khuyến nghị:

- `refscore_decay_power = 1.0` hoặc `1.5` cho ablation đầu tiên.
- `refscore_min_strength = 0.15` để ref-score không chết quá sớm khi vẫn còn dưới budget.
- `refscore_threshold_min = 0.5`.
- `refscore_threshold_max = 0.9`, không nên đẩy lên 0.95 ngay từ đầu vì quá cực đoan.

Tác động kỳ vọng:

- Densification vùng specular mượt hơn.
- Có thể gián tiếp giảm over-densify do prior false positive, nhưng đây chỉ là side effect. Cơ chế trực tiếp của B2 là giảm cường độ guidance khi tiến gần budget, không phải phân biệt false positive tốt hơn.
- Có thể giảm hiệu quả ref-score nếu decay quá mạnh.

Kết luận: **nên làm sau B1**, với ablation riêng cho `refscore_decay_power` hoặc có clamp minimum strength.

### A — Residual-Adaptive Prior

Đây là hướng có tiềm năng cao nhất nhưng cũng rủi ro cao hơn B1/B2.

Ý tưởng cập nhật `cam.ref_score` từ residual hiện tại là đúng: vùng nào model còn sai nhiều, đặc biệt trong vùng static prior xác nhận là specular, thì vùng đó nên tiếp tục được ưu tiên densify.

Tuy nhiên, cần quyết định rõ residual nào được dùng:

- Nếu dùng `GT - render_full`, residual có thể thấp vì SH/ASG đã fit màu tạm thời, dù geometry coverage chưa thật sự tốt.
- Nếu dùng `GT - only_sh` hoặc base render không có ASG, signal sẽ phù hợp hơn với mục tiêu tìm vùng specular mà base representation chưa cover.

Trong pipeline hiện tại, `compute_gaussian_score_fastgs()` render không truyền `mlp_color`, nên score densification đang gần với base/SH render hơn là full ASG render. Vì vậy, nếu implement adaptive prior, nên giữ logic nhất quán với hướng này.

Về chi phí, adaptive prior cần render thêm một số train views định kỳ. Nếu update toàn bộ 100 train views của `toaster` mỗi 3000 iteration, từ 5000 đến 14000 sẽ có khoảng 4 lần update, tức khoảng **400 no-grad renders** thêm. Nếu chỉ sample `adaptive_prior_num_cameras = 20`, chi phí còn khoảng **80 no-grad renders**. Đây là mức hợp lý hơn cho ablation đầu tiên và giúp quyết định interval mà không làm training đội chi phí quá mạnh.

Tác động kỳ vọng:

- Có thể tập trung ref-score vào vùng specular mà model thật sự còn sai.
- Có thể giảm over-densify ở vùng prior tĩnh đã được cover đủ.
- Tốn thêm thời gian vì phải render thêm train cameras định kỳ.
- Có rủi ro nhiễu nếu update quá sớm hoặc residual cao do vùng diffuse/texture khó.

Kết luận: **nên làm sau B1+B2**, và nên dùng residual từ base/only_sh kết hợp static prior.

## Effect lên việc chạy pipeline

Nếu thực hiện các chỉnh sửa trên trục Geometry Coverage, pipeline sẽ thay đổi chủ yếu ở giai đoạn densification:

- Ref-score guidance sẽ scale theo scene tốt hơn.
- Vùng specular có khả năng nhận thêm Gaussian đúng chỗ hơn.
- Số Gaussian cuối có thể thay đổi, đặc biệt ở scene có nhiều highlight như `toaster`.
- Training time và VRAM có thể tăng nếu budget mới cao hơn hoặc adaptive prior render thêm camera định kỳ.
- Metrics như `Spec_PSNR`, `ASG_Residual_IoU`, và visual highlight có thể cải thiện nếu bottleneck hiện tại là thiếu coverage.
- Nếu bottleneck chính là SH cạnh tranh ASG hoặc normal proxy sai, Geometry Coverage chỉ cải thiện một phần.

Khi chạy ablation sau khi sửa, cần ghi rõ thêm các config mới:

- `max_refscore_gaussians`
- `refscore_budget_multiplier`
- `refscore_budget_min`
- `refscore_budget_max`
- `refscore_decay_power`
- `refscore_min_strength`
- `refscore_threshold_min`
- `refscore_threshold_max`
- `use_adaptive_prior`
- `adaptive_prior_start`
- `adaptive_prior_interval`
- `adaptive_prior_num_cameras`
- `adaptive_prior_ema`

Nếu không ghi các thông tin này, các run sau sẽ khó so sánh trực tiếp với ablation cũ.

## Khuyến nghị triển khai

Thứ tự triển khai nên là:

1. **B1 — Scene-Relative Budget**
2. **B2 — Soft Decay Threshold**
3. **A — Residual-Adaptive Prior**

Không nên bắt đầu ngay bằng adaptive residual, vì nó thay đổi signal khá mạnh và khó tách nguyên nhân nếu metric tăng/giảm. B1 là bước gọn nhất để kiểm tra giả thuyết rằng budget cứng 400K đang không phù hợp với một số scene. B2 giúp transition mượt hơn. Sau khi hai phần này ổn định, adaptive prior mới là bước đáng thử để giải quyết vấn đề prior tĩnh.

## Quyết định implement

Implementation nên có đủ flag cho cả B1, B2, và A, nhưng ablation đầu tiên chỉ nên bật hành vi B1/B2. Adaptive prior được thêm vào code nhưng để `use_adaptive_prior=False` mặc định.

Default được chọn:

```python
max_refscore_gaussians = -1
refscore_budget_multiplier = 10.0
refscore_budget_min = 200000
refscore_budget_max = 1000000
refscore_decay_power = 1.0
refscore_min_strength = 0.15
refscore_threshold_min = 0.5
refscore_threshold_max = 0.9
use_adaptive_prior = False
adaptive_prior_start = 5000
adaptive_prior_interval = 3000
adaptive_prior_num_cameras = 20
adaptive_prior_ema = 0.7
```

Adaptive prior nếu được bật sẽ dùng residual từ base/SH render, tức render không truyền `mlp_color`, để nhất quán với `compute_gaussian_score_fastgs()`. Đây là lựa chọn phù hợp hơn cho trục Geometry Coverage so với dùng full ASG render.

## Kết luận

Phương án cải thiện Geometric Coverage là **hợp lý và phù hợp với pipeline hiện tại**, nhưng cần triển khai theo từng bước. Nó nên được xem là cải tiến cho ref-score guided densification, không phải thay thế toàn bộ cơ chế ADC/FastGS.

Kỳ vọng hợp lý là cải thiện coverage tại vùng specular, giúp `Spec_PSNR`, `ASG_Residual_IoU`, và visual quality tốt hơn. Tuy nhiên, đây không phải lời giải đầy đủ cho toàn bộ bài toán specular. Nếu sau khi cải thiện coverage mà `only_asg` vẫn yếu, cần tiếp tục xử lý trục **Representation Capacity / SH-ASG role separation**.
