Đọc feedback rồi. Kết luận ngắn: **feedback đúng hướng**, nhưng điểm 1 bị gọi hơi quá là “bug nghiêm trọng” nếu xét đúng scope MVP mình viết.

**1. Vị trí mask gradient `f_rest`: đúng một phần, nên sửa plan**

Phản biện nhỏ: với `optimizer_type="default"` hiện tại, `f_rest` nằm trong `shoptimizer`, không nằm trong `self.optimizer` ([gaussian_model.py](/home/baodinh/baodinh_thesis/spec-fastgs/scene/gaussian_model.py:259)). Vì vậy nếu MVP chỉ hỗ trợ default như mình đã ghi, mask trong `_step_f_rest_optimizer()` vẫn có tác dụng trước `shoptimizer.step()`.

Nhưng feedback đúng ở điểm quan trọng: nếu sau này dùng `optimizer_type="sparse_adam"`, `f_rest` nằm chung trong `self.optimizer`, còn `shoptimizer=None` ([gaussian_model.py](/home/baodinh/baodinh_thesis/spec-fastgs/scene/gaussian_model.py:263)). Khi đó `self.optimizer.step()` chạy trước, nên mask trong `_step_f_rest_optimizer()` là quá muộn hoặc không chạy luôn. Cách robust hơn là tạo `_apply_f_rest_grad_mask()` và gọi ngay đầu `optimizer_step()`, trước mọi `.step()` ([gaussian_model.py](/home/baodinh/baodinh_thesis/spec-fastgs/scene/gaussian_model.py:285)).

Có một nuance nữa: `f_rest` hiện có thể tích lũy grad qua các iteration không step. Mask ở đầu `optimizer_step()` là hướng an toàn hơn, nhưng nếu muốn “mask đúng từng contribution của từng frame” tuyệt đối thì cần hook hoặc quyết định lại chuyện accumulate grad. Với MVP, mask trước mọi optimizer step là hợp lý.

**2. `lambda_spec_reg`: nhận xét đúng, nhưng cách bật nên cẩn thận**

Đúng là mình đã bắt bệnh `lambda_spec_reg` chưa dùng, nhưng chưa đưa nó vào implementation phase. Hiện arg tồn tại ở [arguments/__init__.py](/home/baodinh/baodinh_thesis/spec-fastgs/arguments/__init__.py:128), còn loss thật chỉ là `loss = photometric_loss` ở [train.py](/home/baodinh/baodinh_thesis/spec-fastgs/train.py:293).

Tuy nhiên mình không nên bật thẳng `lambda_spec_reg=0.01` theo default hiện tại, vì tham số này trước giờ **chưa có hiệu lực**; bật lên sẽ âm thầm đổi baseline. L2 trên `spec_sparse` cũng không “chống collapse về zero”; nó chủ yếu chống explode và khuyến khích ASG nhỏ lại, nên có thể làm `only_asg` yếu nếu bật quá sớm.

Cách đúng hơn: thêm Phase 2b optional, hoặc đổi default thành `lambda_spec_reg=0.0` khi nối vào loss, rồi ablate riêng sau R1. Nếu bật, phải log `spec_reg_loss` vào train metadata.

Tóm lại: feedback của bạn **đúng về hướng sửa**. Mình chỉ chỉnh lại mức độ: item 1 không sai trong scope default-only, nhưng nên sửa để robust; item 2 đúng là thiếu trong plan, nhưng không nên bật reg mặc định một cách âm thầm.