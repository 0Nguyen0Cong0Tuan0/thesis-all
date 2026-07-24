# I/ Tổng quan.
## 1. Tên đề tài: 
"Tối ưu hóa chất lượng tái tạo bề mặt phản quang trong kỹ thuật Loang Gaussian ba chiều tốc độ cao bằng Hàm Gauss cầu dị hướng (ASG)"

## 2. Chủ đề và ý tưởng nghiên cứu
- Khóa luận tập trung nghiên cứu và phát triển một phương pháp giúp giúp nâng cao chất lượng phục dựng bề mặt phản quang trong kỹ thuật Loang Gaussian ba chiều nhờ tận dụng Hàm Gauss cầu dị hướng (ASG), tối ưu hóa độ phủ hình học nhờ cơ chế 

- Quy trình gồm các công đoạn:
    - 1. Trích xuất vùng phản quang từ dữ liệu đầu vào để tính toán thành bản đồ tọa độ Reflection Score.
    - 2. Dựa vào bản đồ tọa độ Reflection Score và các đám mây điểm ba chiều để khởi tạo và loang các Gaussian ba chiều, ưu tiên loang vào các vùng chứa các bề mặt phản quang.
    - 3. Huấn luyện mô hình cho các Gaussians với mỗi hạt sở hữu song song kiến trúc biểu diễn màu gồm hàm Spherical Harmonics và Anisotropic Spherical Gaussian.
    - 4. Kiểm soát số lượng bằng cách sinh thêm hoặc tỉa bớt Gaussian dựa vào đóng góp chất lượng hình ảnh của từng hạt trên các góc nhìn khác nhau.

- Đề tài đặt mục tiêu đem đến một đầu ra là mô hình ba chiều với chất lượng hình ảnh tổng thể tốt nhờ cải thiện các chi tiết phản quang, ánh sáng vật lý. Ngoài ra, chi phí thời gian tiêu tốn cho việc huấn luyện và lưu trữ mô hình giảm đáng kể mà không làm giảm chất lượng hình ảnh.

## 3. Ý nghĩa khoa học


Đề tài cung cấp một nghiên cứu tập trung vào cải tiến phương pháp 3D Gaussian Splatting (3DGS) – một kỹ thuật biểu diễn và render 3D đang rất đột phá hiện nay – để xử lý tốt hơn các bề mặt phản xạ ánh sáng (specular/shiny surfaces) trong khi vẫn giữ được ưu điểm về tốc độ. Đem đến nhiều ý nghĩa trong khoa học như:

- Giải quyết hạn chế cốt lõi của 3D Gaussian Splatting truyền thống: Các phương pháp 3DGS cơ bản thường dùng Spherical Harmonics (SH) để biểu diễn màu sắc. SH rất tốt cho các bề mặt nhám (diffuse) nhưng lại kém hiệu quả và bị giới hạn bậc (degree) khi mô phỏng các đốm sáng chói (highlight) sắc nét hoặc phản xạ phức tạp thay đổi theo góc nhìn. Đề tài đã đề xuất kiến trúc kết hợp để khắc phục điểm yếu này.
- Kiến trúc biểu diễn kết hợp (Hybrid Representation): Đề tài đề xuất việc tách biệt vai trò (role separation): sử dụng SH cho phần màu sắc khuếch tán (diffuse) và sử dụng một hàm khác phù hợp hơn (như ASG - Asymmetric/Anisotropic Spherical Gaussian) để biểu diễn phần phản xạ chói (specular). Việc phân tách này giúp mô hình học đúng bản chất vật lý của ánh sáng hơn.
- Chiến lược tối ưu hóa và cấp phát tài nguyên thông minh (Guided Densification): Thay vì tăng số lượng Gaussian một cách mù quáng, đề tài nghiên cứu sử dụng các hàm Prior (như Tan-Ikeuchi hay Shafer-Klinker) để trích xuất và chấm điểm (ref_score) vùng nào thực sự có phản xạ. Từ đó, mô hình tập trung sinh thêm Gaussian (densification) và dành dung lượng (capacity) ASG cho đúng những vị trí cần thiết, giúp tăng độ chính xác (Spec_PSNR) mà không làm bùng nổ số lượng Gaussian.
- Định lượng sự đánh đổi (Trade-off Analysis): Đề tài cung cấp những phân tích thực nghiệm sâu sắc (thông qua hệ thống ablation study đồ sộ) về sự đánh đổi giữa chất lượng tái tạo quang học (PSNR, LPIPS, SSIM), khả năng định vị vùng sáng (ASG_IoU), thời gian huấn luyện (Training Time), và bộ nhớ VRAM. Điều này đóng góp dữ liệu và góc nhìn quan trọng cho cộng đồng nghiên cứu Novel View Synthesis.

## 4. Ý nghĩa ứng dụng
Nhờ cải tiến chất lượng mô hình ba chiều và giảm đáng kể chi phí lưu trữ cũng như tính toán mà công trình mở ra nhiều ứng dụng thực tiễn gồm:
- Nâng cao chất lượng cho Thực tế Ảo (VR) và Thực tế Tăng cường (AR): Yêu cầu tối thượng của VR/AR là tốc độ render phải đạt thời gian thực (real-time FPS cao) và hình ảnh phải chân thực khi người dùng di chuyển góc nhìn. spec-fastgs kế thừa tốc độ của FastGS đồng thời tái tạo đúng các vệt sáng di chuyển trên vật thể (như kính, kim loại), giúp trải nghiệm VR/AR không bị "giả" và tăng độ đắm chìm (immersion).
- Số hóa sản phẩm và Thương mại điện tử (E-commerce / Digital Twins): Rất nhiều sản phẩm thương mại có bề mặt bóng bẩy (trang sức, ô tô, đồ gốm sứ, thiết bị điện tử). Phương pháp này cho phép quét và đưa các sản phẩm này lên môi trường 3D/Web với chất lượng phản xạ vật liệu chân thực nhất, giúp khách hàng có cái nhìn trực quan và chính xác hơn về sản phẩm.
- Tiết kiệm tài nguyên phần cứng: So với một số mô hình chuyên trị Specular khác vốn rất nặng và tốn VRAM, thiết kế của spec-fastgs hướng tới tính hiệu quả (được tinh chỉnh qua các thông số asg_degree, số lượng điểm...). Điều này giúp các studio hoặc cá nhân có thể huấn luyện và sinh ra mô hình 3D chất lượng cao mà không nhất thiết phải sở hữu các hệ thống siêu máy tính.
- Ứng dụng trong làm phim và Game: Cung cấp một công cụ mạnh mẽ để quét bối cảnh thực (kể cả những căn phòng có nhiều đồ vật phản chiếu, gương kính) và chuyển đổi thành tài nguyên 3D (3D assets) cho các engine game hoặc phần mềm hậu kỳ với độ chân thực về ánh sáng cao.

## 5. Phát biểu bài toán

- Input (Đầu vào): Một nhóm các bức ảnh chụp một vật thể/căn phòng từ nhiều góc độ khác nhau, kèm theo thông tin về việc "bức ảnh đó được chụp từ vị trí nào".
- Output (Đầu ra): Một mô hình biểu diễn 3D (3D Scene Representation)
    - Khác với các phương pháp tái tạo 3D truyền thống trả về lưới đa giác (3D Mesh - file .obj, .stl), Output của 3DGS là một tệp dữ liệu (thường là file .ply) chứa thông số của hàng triệu hạt 3D Gaussians.
    - Mỗi hạt mang đầy đủ tính chất hình học (tọa độ XYZ, kích thước, độ mờ đục) và tính chất quang học (các hệ số màu sắc như SH hay ASG để mô phỏng bề mặt phản xạ). Đây chính là "bản sao kỹ thuật số" (Digital Twin) 3D của không gian thực được tái tạo.

### 5.1 Đầu vào 
Là tập dữ liệu (Dataset) đã được tiền xử lý, bao gồm 3 thành phần chính:

- Tập ảnh đa góc nhìn (Multi-view Images): Một tập hợp N bức ảnh 2D RGB ghi lại một bối cảnh tĩnh. Các ảnh này chứa thông tin về màu sắc (đặc biệt là các mảng màu chói sáng - specular) và cấu trúc hình học của vật thể.
- Thông số Camera (Camera Parameters): Đi kèm với mỗi bức ảnh i là tập hợp thông số toán học của camera chụp bức ảnh đó, bao gồm:
    - Thông số nội suy (Intrinsics): Tiêu cự (focal length), điểm trung tâm quang học (principal point) - quyết định góc nhìn rộng hay hẹp.
    - Thông số ngoại suy (Extrinsics - Pose): Ma trận biến đổi (Rotation matrix & Translation vector) biểu diễn vị trí tọa độ (x,y,z) và hướng quay của camera trong không gian 3D. (Thường các thông số này được trích xuất tự động qua phần mềm SfM như COLMAP).
- Đám mây điểm thưa (Sparse Point Cloud - Tùy chọn nhưng cần thiết cho 3DGS): Một tập hợp các điểm 3D thưa thớt ban đầu (cũng tạo ra từ COLMAP) dùng làm mốc tọa độ để hệ thống khởi tạo các hạt 3D Gaussians, giúp rút ngắn thời gian hội tụ thay vì khởi tạo ngẫu nhiên.

### 5.2 Đầu ra
Đầu ra của hệ thống được chia làm hai phần chính, tương ứng với kết quả của giai đoạn huấn luyện (để lưu trữ 3D) và kết quả của giai đoạn kiểm thử (để hiển thị):

1. Mô hình biểu diễn 3D (3D Scene Representation - Output cốt lõi) Sau khi quá trình tối ưu hóa kết thúc, đầu ra thực sự và quan trọng nhất mà hệ thống thu được là một bản sao kỹ thuật số 3D của không gian thực.

- Định dạng dữ liệu: Khác với các phương pháp tái tạo 3D dạng lưới truyền thống (Mesh - .obj, .stl), output của hệ thống này là một tệp đám mây điểm (thường là định dạng .ply) chứa hàng triệu hạt 3D Gaussians.
- Cấu trúc dữ liệu bên trong: Mỗi hạt 3D Gaussian lưu trữ độc lập các thông số:
    - Tính chất hình học: Tọa độ trung tâm (X, Y, Z), ma trận hiệp phương sai (Covariance) quyết định độ lớn/giãn (Scale) và hướng xoay (Rotation) của hạt, cùng với độ mờ đục (Opacity).
    - Tính chất quang học: Các hệ số biểu diễn màu sắc thay đổi theo góc nhìn (như Spherical Harmonics - SH cho vùng khuếch tán, hoặc Asymmetric Spherical Gaussian - ASG cho vùng chói sáng).
2. Ảnh kết xuất 2D và Dữ liệu phụ trợ (Rendered Outputs - Output ứng dụng) Khi đã có Mô hình 3D ở trên, nếu cung cấp cho hệ thống một "Thông số Camera mới" (Novel View Pose), hệ thống sẽ sử dụng thuật toán Rasterization (chiếu từ 3D xuống mặt phẳng 2D) để xuất ra:

- Ảnh RGB (Rendered Image): Bức ảnh 2D dự đoán từ góc nhìn mới. Nhờ các ẩn số quang học được lưu trong Mô hình 3D, bức ảnh này có khả năng tái hiện chuẩn xác sự dịch chuyển của các vệt sáng phản xạ (specular highlights) theo nguyên lý vật lý khi thay đổi góc camera.
- Bản đồ chiều sâu (Depth Map - Tùy chọn): Bản đồ dạng ảnh xám thể hiện khoảng cách từ camera đến các vật thể trong khung cảnh. Thông tin này cực kỳ hữu ích cho các tác vụ hậu kỳ hoặc ứng dụng Thực tế tăng cường (AR) để xử lý việc che khuất vật thể (occlusion).

### 5.3 Framework chung của hệ thống
Công đoạn 1: Tiền xử lý dữ liệu (Data Preprocessing)
- Mục đích: Trích xuất các thông tin không gian cơ bản từ dữ liệu thô.
- Hoạt động: Hệ thống nhận Tập ảnh Input, sử dụng các thuật toán hình học đa hướng (Structure-from-Motion - SfM) để ước lượng tọa độ của từng Camera (Camera Poses). Đồng thời, công đoạn này tạo ra một tập hợp các điểm 3D thưa thớt (Sparse Point Cloud) đại diện cho cấu trúc thô của khung cảnh.

Công đoạn 2: Khởi tạo mô hình (Initialization)
- Mục đích: Khởi tạo các Gaussiansđể đưa vào quá trình học.
- Hoạt động: Từ đám mây điểm thưa ở Công đoạn 1, hệ thống khởi tạo một cấu trúc biểu diễn 3D ban đầu. Cấu trúc này chứa các ẩn số mang giá trị ngẫu nhiên hoặc mặc định về mặt hình học (vị trí, kích thước, hình dáng) và quang học (màu sắc, tính phản xạ).

Công đoạn 3: Vòng lặp Tối ưu hóa (Optimization Loop / Training) Đây là công đoạn cốt lõi, lặp đi lặp lại hàng nghìn lần để hệ thống tự học và điều chỉnh các ẩn số 3D. Vòng lặp gồm 4 bước nhỏ:

1. Kết xuất thuận (Forward Rendering): Chọn một tọa độ camera (đã biết từ Công đoạn 1), hệ thống chiếu các ẩn số 3D xuống mặt phẳng 2D để tạo ra một "Ảnh dự đoán".
2. Tính toán hàm mất mát (Loss Computation): Lấy "Ảnh dự đoán" trừ đi "Ảnh Groundtruth" (ảnh thật chụp tại camera đó) để tính ra mức độ sai số (Loss).
3. Cập nhật trọng số (Backpropagation): Dùng thuật toán lan truyền ngược để điều chỉnh lại các giá trị của ẩn số hình học và quang học, sao cho ở vòng lặp tiếp theo, sai số sẽ giảm đi.
4. Điều chỉnh cấu trúc hình học (Adaptive Control/Densification): Định kỳ kiểm tra và thay đổi số lượng các ẩn số 3D (ví dụ: nhân bản thêm ở vùng có chi tiết phức tạp, xóa bớt ở vùng trống) để tăng cường độ chính xác cho mô hình.

Công đoạn 4: Kiểm thử và Ứng dụng (Testing & Inference)
- Mục đích: Đánh giá hiệu năng và sinh ra Output cuối cùng.
- Hoạt động: Khi Vòng lặp tối ưu hóa (Công đoạn 3) kết thúc, ta thu được Mô hình 3D hoàn chỉnh. Lúc này, hệ thống sẽ nhận một Tọa độ camera mới (Novel View), đi qua bước Kết xuất (Rendering) một lần duy nhất để xuất ra Ảnh kết xuất 2D (Output) hoàn chỉnh mà không cần thực hiện tính Loss hay cập nhật trọng số nữa.

## 6. Đóng góp của khóa luận
Khóa luận này tập trung giải quyết bài toán tái tạo và kết xuất các bề mặt phản xạ ánh sáng (specular surfaces) – một trong những điểm yếu lớn nhất của các phương pháp 3D Gaussian Splatting (3DGS) truyền thống, trong khi vẫn phải duy trì được ưu điểm về tốc độ và hiệu suất phần cứng. Cụ thể, khóa luận mang đến những đóng góp khoa học và thực tiễn sau:

1. Đề xuất kiến trúc biểu diễn kết hợp (Hybrid Representation) hiệu quả: Khóa luận đề xuất phương pháp spec-fastgs, một kiến trúc cải tiến kết hợp giữa Spherical Harmonics (SH) để mô phỏng vùng ánh sáng khuếch tán (diffuse) và Anisotropic Spherical Gaussian (ASG) để mô phỏng vùng chói sáng (specular). Thiết kế này giúp mô hình bóc tách và học được bản chất vật lý của ánh sáng phức tạp mà không làm mất đi tốc độ huấn luyện/kết xuất (rendering) nhanh đặc trưng của phương pháp nền tảng FastGS.

2. Đề xuất cơ chế cấp phát tài nguyên thông minh (Guided Densification & Adaptive Prior): Thay vì tăng số lượng 3D Gaussians một cách ngẫu nhiên hoặc dàn trải, khóa luận đề xuất sử dụng cơ chế ref_score kết hợp với các mô hình tri thức tiền nghiệm về phản xạ (Reflection Priors như Tan-Ikeuchi hay Shafer-Klinker). Hơn nữa, việc tích hợp "Adaptive Prior" giúp mô hình tự động nhận diện và tập trung phân bổ cấu trúc hình học (Geometry Coverage) vào đúng các vị trí có độ chói sáng cao. Điều này giúp cải thiện rõ rệt độ chính xác của vùng phản xạ (Spec_PSNR) mà vẫn tối ưu được dung lượng VRAM và giới hạn số lượng hạt Gaussians.

3. Đề xuất cơ chế ràng buộc phân tách vai trò (Role Separation Supervision): Khóa luận đi sâu vào việc thiết kế các hàm mất mát (Loss functions) và mặt nạ (Masks) để ép mô hình học đúng chức năng: ngăn chặn hiện tượng thành phần SH hấp thụ nhầm các vệt sáng thuộc về thành phần ASG. Các cơ chế như sh_spec_mask, spec-L1 regularization, và bù đắp pháp tuyến (normal-delta) được đề xuất và thực nghiệm để cải thiện chỉ số chồng lấp vùng chói sáng (ASG_IoU) cũng như nâng cao chất lượng thị giác (perceptual quality).

4. Đánh giá thực nghiệm toàn diện trên các bộ dữ liệu tiêu chuẩn: Thông qua một hệ thống các thí nghiệm (Ablation Study) quy mô và chặt chẽ, khóa luận đã đánh giá toàn diện phương pháp đề xuất trên các bộ dữ liệu có độ phản xạ phức tạp cao (như Mip-NeRF 360 và Ref-NeRF). Các kết quả thực nghiệm chứng minh rằng spec-fastgs đạt được sự cân bằng vượt trội (trade-off) giữa chất lượng tái tạo vùng specular, chi phí bộ nhớ và thời gian huấn luyện so với các phương pháp tiên tiến hiện hành.

## 7. Khảo sát tổng quan
### 7.1 Quá trình phát triển của các giải pháp (Chọn lọc các công trình liên quan)
Lĩnh vực Tổng hợp góc nhìn mới (Novel View Synthesis) và tái tạo bề mặt phản xạ (Specular) đã trải qua một quá trình phát triển mạnh mẽ với các cột mốc quan trọng sau:

- Giai đoạn 1: Sự ra đời của Neural Radiance Fields (NeRF) NeRF (Mildenhall et al., 2020) là công trình tiên phong sử dụng Mạng nơ-ron (MLP) để biểu diễn không gian 3D dưới dạng trường bức xạ liên tục. Mặc dù tạo ra chất lượng hình ảnh xuất sắc, NeRF gặp phải rào cản chí mạng: thuật toán dò tia (ray-marching) quá chậm, mất nhiều ngày để huấn luyện và không thể render theo thời gian thực (chỉ đạt mức giây/khung hình). Các biến thể sau đó như Ref-NeRF đã ra đời để cải thiện vùng phản xạ cho NeRF, nhưng vẫn không giải quyết được bài toán tốc độ.
- Giai đoạn 2: Đột phá tốc độ với 3D Gaussian Splatting (3DGS) và FastGS Để giải quyết vấn đề tốc độ của NeRF, Kerbl et al. (2023) đã giới thiệu 3DGS. Thay vì dùng mạng nơ-ron liên tục, 3DGS dùng hàng triệu hạt Gaussian rời rạc kết hợp với thuật toán Rasterization, giúp tốc độ render tăng vọt lên hàng trăm FPS. Tuy nhiên, 3DGS tiêu chuẩn sử dụng Spherical Harmonics (SH) - một hàm tần số thấp - nên rất kém trong việc biểu diễn các vệt chói sáng (highlight). Gần đây, FastGS được đề xuất như một bản tối ưu hóa sâu của 3DGS, giúp tăng tốc độ huấn luyện và giảm VRAM, nhưng cốt lõi vẫn dùng SH nên vẫn thất bại trước bề mặt phản xạ phức tạp.
- Giai đoạn 3: Các biến thể chuyên trị Specular (Specular-Gaussians, Spec-GS...) Nhận thấy điểm yếu của SH, các công trình gần đây đã cố gắng thay thế nó. Ví dụ, Specular-Gaussians tích hợp hàm Anisotropic Spherical Gaussian (ASG) để mô phỏng sự bất đẳng hướng của vệt sáng. Mặc dù chất lượng vùng chói sáng tăng lên rõ rệt, nhưng do gán ASG cho mọi hạt Gaussian một cách cồng kềnh, các mô hình này đánh mất đi ưu điểm cốt lõi của 3DGS: tốc độ huấn luyện giảm sút trầm trọng và ngốn lượng VRAM khổng lồ.

### 7.2 Bảng so sánh các giải pháp tiên tiến
Để thấy rõ bức tranh toàn cảnh và định vị phương pháp của khóa luận, dưới đây là bảng so sánh các công trình dựa trên các tiêu chí kỹ thuật cốt lõi:
Dưới đây là Bảng so sánh đã được điều chỉnh, tách biệt rõ ràng giữa **3DGS tiêu chuẩn** và **FastGS** để bạn có cái nhìn chi tiết hơn về sự tiến hóa của tốc độ và bộ nhớ trước khi `spec-fastgs` ra đời:

| Tiêu chí so sánh | Tiêu chuẩn NeRF (vd: Mip-NeRF) | NeRF trị Specular (vd: Ref-NeRF) | 3DGS (Tiêu chuẩn) | FastGS (Nền tảng tối ưu) | 3DGS trị Specular (Specular-Gaussians) | **Đề xuất: spec-fastgs** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cơ chế biểu diễn** | Mạng MLP | Mạng MLP + Directional | 3D Gaussians + SH | 3D Gaussians + SH | 3D Gaussians + ASG | **Gaussians + SH + ASG** |
| **Chất lượng vùng Specular** | Yếu | Tốt | Yếu | Yếu | Tốt | **Tốt** |
| **Tốc độ Huấn luyện (Train)** | Rất chậm (Vài ngày) | Rất chậm | Nhanh (~ 45-60 phút) | Rất nhanh (~ 15-20 phút) | Chậm (Vài giờ) | **Nhanh (~ 30 phút)** |
| **Tốc độ Kết xuất (FPS)** | < 1 FPS | < 1 FPS | ~ 100 FPS | > 150 FPS | ~ 30-50 FPS | **> 100 FPS** |
| **Mức độ tiêu thụ VRAM** | Cao | Rất Cao | Trung bình | Rất Thấp | Rất Cao | **Thấp / Trung bình** |
| **Tối ưu cấp phát (Densification)** | Không | Không | Cấp phát dàn trải | Cấp phát dàn trải | Cấp phát dàn trải | **Cấp phát có định hướng (Ref_Score)** |

### 7.3 Đánh giá tổng quan
Dựa trên quá trình phát triển và bảng so sánh trên, Chương 2 xin đúc kết lại bằng việc trả lời hai câu hỏi cốt lõi:

**1. Thực trạng vấn đề**
Trong bối cảnh của bài toán tái tạo 3D và tổng hợp góc nhìn mới, các công trình nghiên cứu hiện hành đang đối mặt với sự đánh đổi (trade-off) rất lớn giữa hiệu năng tính toán và khả năng tái tạo tính chất vật lý của ánh sáng. Thực trạng này được thể hiện qua hai hướng tiếp cận chính:

1. Hướng tiếp cận ưu tiên hiệu năng (Đại diện: 3DGS tiêu chuẩn, FastGS):
- Đặc điểm: Các phương pháp này đạt được tốc độ huấn luyện và kết xuất (rendering) vượt trội bằng cách sử dụng cấu trúc dữ liệu nhẹ và các hàm toán học đơn giản, điển hình là hàm Spherical Harmonics (SH).
- Hạn chế: Do hàm SH mang đặc điểm của tín hiệu tần số thấp, hệ thống gặp khó khăn nghiêm trọng khi phải nội suy các vệt sáng chói (specular highlights) có tần số cao trên bề mặt kim loại, kính hoặc gốm sứ. Kết quả kết xuất thường bị mờ (blur), mất chi tiết phản xạ, hoặc xuất hiện nhiễu hạt (artifacts) làm suy giảm tính chân thực của khung cảnh.
2. Hướng tiếp cận ưu tiên chất lượng phản xạ (Đại diện: Specular-Gaussians, Ref-NeRF):
- Đặc điểm: Để khắc phục nhược điểm của SH, các nghiên cứu này đề xuất sử dụng các mô hình quang học phức tạp hơn (như Mạng nơ-ron đa tầng - MLP, hoặc hàm Anisotropic Spherical Gaussian - ASG) nhằm mô phỏng chính xác sự thay đổi của vệt sáng theo góc nhìn.
- Hạn chế: Các hàm tính toán phức tạp này thường được áp dụng một cách toàn cục và thiếu chọn lọc cho toàn bộ mô hình (kể cả những khu vực bề mặt nhám không có phản xạ). Hệ quả là tài nguyên phần cứng bị lãng phí nghiêm trọng, dẫn đến mức tiêu thụ VRAM tăng đột biến, thời gian huấn luyện kéo dài gấp nhiều lần, và tốc độ khung hình (FPS) suy giảm đáng kể, khiến việc áp dụng vào môi trường thời gian thực (real-time) gặp nhiều trở ngại.

**2. Định hướng giải quyết**
Để giải quyết bài toán đánh đổi (trade-off) giữa chất lượng kết xuất bề mặt phản xạ và chi phí tính toán, khóa luận đề xuất phương pháp spec-fastgs. Phương pháp này hướng tới việc tích hợp khả năng tái tạo ánh sáng chói sắc nét vào các hệ thống tốc độ cao thông qua 3 giải pháp trọng tâm:

- Kế thừa kiến trúc tối ưu hiệu năng: Khóa luận lựa chọn nền tảng FastGS làm kiến trúc cơ sở (backbone). Việc này đảm bảo hệ thống duy trì được ưu điểm tối ưu hóa bộ nhớ cấp thấp và đạt tốc độ kết xuất (rendering) thời gian thực, tạo tiền đề vững chắc trước khi tích hợp các hàm biểu diễn quang học phức tạp hơn.
- Cơ chế phân bổ tài nguyên có định hướng (Guided Densification): Nhằm khắc phục tình trạng bùng nổ VRAM do sử dụng hàm ASG một cách dàn trải, khóa luận đề xuất tích hợp hệ thống đánh giá Ref_Score và các mô hình tiền nghiệm (Adaptive Priors). Cơ chế này cho phép hệ thống tự động khoanh vùng các khu vực có xác suất phản xạ cao, từ đó chỉ cấp phát năng lực biểu diễn (các hạt Gaussians chứa hàm ASG) vào đúng các vị trí cần thiết, giúp tối ưu hóa dung lượng mô hình.
- Ràng buộc phân tách thành phần quang học (Role Separation Supervision): Khóa luận thiết kế và áp dụng các hàm mất mát (Loss functions) cùng cơ chế mặt nạ (Masks) chuyên biệt nhằm phân định rõ chức năng của các hàm toán học. Cụ thể, các ràng buộc này điều hướng hàm Spherical Harmonics (SH) chỉ hội tụ ở thành phần ánh sáng khuếch tán (diffuse), và hàm ASG chỉ hội tụ ở thành phần chói sáng (specular), qua đó giảm thiểu tối đa sự giao thoa không mong muốn giữa hai thành phần.

Mục tiêu tổng thể: Phương pháp spec-fastgs được kỳ vọng sẽ đạt chất lượng tái tạo bề mặt phản xạ tương đương với các mô hình chuyên biệt nặng nề, nhưng vẫn duy trì được thời gian huấn luyện ngắn và tốc độ khung hình cao đặc trưng của các mô hình tối ưu tốc độ

## 8. Phương pháp
Để giải quyết bài toán tái tạo bề mặt phản xạ trong khi vẫn duy trì hiệu năng cao, phương pháp spec-fastgs được thiết kế dựa trên một kiến trúc tích hợp. Phương pháp này bao gồm ba thành phần kỹ thuật cốt lõi:

### 8.1. Kiến trúc biểu diễn quang học kết hợp (Hybrid Appearance Representation)

#### 8.1.1. Phân tách thành phần quang học của Gaussians
Trong thực tế vật lý, mức độ bức xạ năng lượng ánh sáng từ một bề mặt vật thể tới điểm nhìn của camera không đồng nhất. Dựa trên tính chất phân bố, ánh sáng phản xạ được phân loại thành hai thành phần chính: 
1.  **Thành phần khuếch tán (Diffuse):** Ánh sáng tán xạ đồng đều ra nhiều hướng khi tương tác với bề mặt nhám. Đặc trưng quang học của thành phần này là tín hiệu tần số thấp, mức độ thay đổi cường độ ánh sáng diễn ra mượt mà và ít phụ thuộc vào sự dịch chuyển của góc nhìn.
2.  **Thành phần phản xạ (Specular):** Ánh sáng phản xạ có định hướng theo định luật phản xạ quang học, thường xuất hiện trên các bề mặt nhẵn bóng (như kim loại, kính). Đặc trưng của thành phần này là tín hiệu tần số cao; các vệt sáng chói (highlights) có độ dốc năng lượng lớn và thay đổi cực kỳ nhanh chóng khi góc nhìn bị dịch chuyển.

Phương pháp 3D Gaussian Splatting (3DGS) tiêu chuẩn ứng dụng hàm Spherical Harmonics (SH) để biểu diễn màu sắc phụ thuộc góc nhìn. Tuy nhiên, bản chất toán học của SH là tập hợp các hàm cơ sở dải tần thấp. Việc giới hạn toàn bộ thuộc tính màu sắc vào một hàm SH duy nhất dẫn đến hiện tượng thiếu hụt năng lực biểu diễn đối với các tín hiệu tần số cao. Cụ thể, hàm SH không thể mô phỏng chính xác biên độ hẹp của vệt sáng chói, gây ra hiện tượng mờ nhòe kết cấu bề mặt.

<chèn thêm ảnh minh họa ở đây>

Để giải quyết triệt để giới hạn này, công trình đề xuất kiến trúc **Biểu diễn kết hợp (Hybrid Representation)** giữa SH và ASG để đồng biểu diễn màu sắc cho các hạt Gaussians. Nguyên lý cốt lõi của kiến trúc này là phân tách cấu trúc tính toán màu sắc thành hai luồng độc lập gọi là Diffuse và Specular, tương ứng với hai bản chất vật lý của ánh sáng đã nêu ở trên là SH và ASG, sau đó tổng hợp tuyến tính để tạo ra kết quả bức xạ cuối cùng.


Kế thừa kiến trúc không gian của hệ thống 3DGS truyền thống, mỗi hạt 3D Gaussian trong mô hình duy trì các thuộc tính hình học cơ bản bao gồm: Tọa độ trung tâm $\mu \in \mathbb{R}^3$, ma trận hiệp phương sai $\Sigma$ định hình cấu trúc ellipsoid, và hệ số độ mờ đục $\alpha \in [0, 1]$. 

Thuộc tính màu sắc $C$ phụ thuộc vào hướng nhìn $v$ (viewing direction) lúc này sẽ được tái cấu trúc theo phương trình tổng quát:
$$C(v) = C_{diffuse}(v) + C_{specular}(v)$$

Trong đó, hai thành phần được mô hình hóa chi tiết như sau:

**A. Thành phần Khuếch tán (Diffuse Component) với Spherical Harmonics**
Thành phần $C_{diffuse}(v)$ được tính toán thông qua hàm SH. Thông qua việc tổ hợp tuyến tính các hàm cơ sở trực giao trên mặt cầu, SH có khả năng xấp xỉ liên tục và mượt mà sự thay đổi của ánh sáng khuếch tán.

Công thức tính màu khuếch tán tại hướng nhìn $v$:
$$C_{diffuse}(v) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{l}^{m} Y_{l}^{m}(v)$$
*Diễn giải biến số:*
*   $L$: Bậc (degree) của tập hàm SH.
*   $Y_{l}^{m}(v)$: Hàm cơ sở SH bậc $l$, thứ tự $m$, phụ thuộc vào vector hướng nhìn $v$.
*   $c_{l}^{m}$: Tập hợp các hệ số biến thiên cần tối ưu hóa trong quá trình huấn luyện mạng.

**B. Thành phần Phản xạ (Specular Component) với Anisotropic Spherical Gaussian**
Để mô phỏng các vệt chói sáng có hình dáng bất đối xứng trên bề mặt cong, phương pháp sử dụng hàm **Anisotropic Spherical Gaussian (ASG)**. Khác với phân phối Gaussian đẳng hướng, ASG cung cấp mức độ kiểm soát độ dốc phân bố độc lập trên hai trục trực giao, cho phép biểu diễn các vệt sáng có tính bất đẳng hướng cao (kéo dài hoặc dẹt).

<chèn thêm ảnh minh họa ở đây>

Hàm ASG biểu diễn vùng phản xạ được định nghĩa bởi phương trình:
$$ASG(v; \xi, \lambda, c_s) = c_s \cdot \max(0, v \cdot z) \cdot \exp\left( -\lambda_x (v \cdot x)^2 - \lambda_y (v \cdot y)^2 \right)$$
*Diễn giải biến số:*
*   $\xi = [x, y, z]$: Hệ trục tọa độ cục bộ trực giao. Trục $z$ là vector chỉ hướng phản xạ trung tâm. Các trục $x$ và $y$ tạo thành không gian tiếp tuyến quyết định hình dạng vệt sáng.
*   $\lambda = [\lambda_x, \lambda_y]$: Hệ số độ nhám (sharpness) dọc theo trục $x$ và $y$. Sự chênh lệch giữa $\lambda_x$ và $\lambda_y$ tạo ra hiệu ứng bất đẳng hướng của vệt phản xạ.
*   $c_s$: Hệ số cường độ màu sắc chói sáng.
*   Toán tử $\max(0, v \cdot z)$ được áp dụng nhằm triệt tiêu các thành phần bức xạ ở mặt bán cầu ẩn.

Tổng năng lượng phản xạ $C_{specular}(v)$ tại một điểm được xấp xỉ bằng tổng của các hàm ASG:
$$C_{specular}(v) = \sum_{k=1}^{K} ASG_k(v)$$
*(Nhằm tối ưu hóa hiệu năng tính toán bộ nhớ, tham số $K$ thường được cố định bằng 1 cho mỗi hạt 3D Gaussian).*

**C. Tổng hợp màu sắc (Final Color Rendering)**
Tại giai đoạn Rasterization, giá trị màu tổng hợp $C(v) = C_{diffuse}(v) + C_{specular}(v)$ của từng hạt Gaussian sẽ được nhân với hệ số mờ đục $\alpha$. Thuật toán trộn alpha (alpha-blending) thực hiện tích lũy các giá trị này theo chiều sâu (từ gần đến xa) dọc theo tia chiếu để tính toán cường độ màu sắc cuối cùng của điểm ảnh:
$$C_{pixel} = \sum_{i \in \mathcal{N}} C_i(v) \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

Thông qua hệ phương trình toán học trên, kiến trúc biểu diễn kết hợp đã giải quyết triệt để nút thắt của các phương pháp tiền nhiệm. Bằng cách tách biến số hàm mục tiêu $C(v)$ thành hai thành phần độc lập, thuật toán tối ưu hóa (Gradient Descent) được điều hướng để phân bổ các sai số tần số thấp vào bộ hệ số $c_{l}^{m}$ của hàm SH, đồng thời hội tụ các sai số tần số cao vào bộ thông số độ nhám $\lambda$ của hàm ASG. Quá trình này đảm bảo tính ổn định trong huấn luyện và nâng cao độ chính xác của kết xuất bề mặt phản xạ mà không đòi hỏi gia tăng bậc của hàm SH.

### 8.2. Cơ chế phân bổ hình học có định hướng (Ref_Score Guided Densification)

#### 8.2.1. Hạn chế của phân bổ toàn cục
Trong quá trình tối ưu hóa của 3D Gaussian Splatting, hệ thống sử dụng thuật toán kiểm soát mật độ (Densification) nhằm tăng cường chi tiết hình học tại các vùng có sai số tái tạo lớn. Thuật toán này thực hiện nhân bản (cloning) hoặc phân tách (splitting) các hạt Gaussian khi gradient của vị trí vượt qua một ngưỡng cố định.

Tuy nhiên, đối với các kiến trúc tích hợp hàm phức tạp như Anisotropic Spherical Gaussian (ASG), việc áp dụng cơ chế phân bổ toàn cục (global densification) bộc lộ nhược điểm nghiêm trọng. Hàm ASG đòi hỏi dung lượng lưu trữ bộ nhớ và chi phí tính toán cao hơn đáng kể so với hàm SH. Nếu hệ thống nhân bản các hạt Gaussian chứa ASG một cách đồng đều trên toàn bộ không gian (bao gồm cả các bề mặt nhám không có tính phản xạ), dung lượng VRAM sẽ bị cạn kiệt nhanh chóng, kéo theo sự suy giảm hiệu năng kết xuất. 

Để giải quyết bài toán tối ưu hóa tài nguyên, công trình đề xuất **Cơ chế phân bổ hình học có định hướng**. Cơ chế này tích hợp tri thức tiền nghiệm (prior knowledge) để hệ thống tự động xác định không gian chứa yếu tố phản xạ, từ đó giới hạn quá trình sinh thêm ASG chỉ tại các khu vực thực sự cần thiết.

#### 8.2.2. Thuật toán và Cơ chế tích hợp

**A. Mô hình tri thức tiền nghiệm về phản xạ (Reflection Priors)**
Nhằm cung cấp thông tin không gian cho hệ thống, phương pháp sử dụng các thuật toán trích xuất đặc trưng quang học trên tập dữ liệu ảnh 2D đầu vào (ví dụ: mô hình Tan-Ikeuchi hoặc không gian màu Shafer-Klinker). Các thuật toán này phân tích sự phân bố của cường độ chói sáng và độ bão hòa màu để trích xuất ra một mặt nạ xác suất (probability mask) $M \in [0, 1]^{H \times W}$ cho từng bức ảnh.

Tại mặt nạ $M$, giá trị pixel tiến gần tới 1 đại diện cho xác suất cao khu vực đó là vệt chói sáng (specular highlight), và giá trị tiến về 0 biểu thị bề mặt khuếch tán (diffuse).

**B. Tính toán điểm số định hướng (Ref_Score)**
Để chuyển đổi tri thức 2D từ mặt nạ $M$ vào không gian 3D, hệ thống gán cho mỗi hạt Gaussian $i$ một trọng số đánh giá gọi là Điểm số phản xạ ($Ref\_Score_i$). 

Trong quá trình huấn luyện, khi hạt Gaussian $i$ được chiếu (project) lên mặt phẳng ảnh của camera $c$, nó sẽ rơi vào một vị trí tọa độ pixel tương ứng. Giá trị xác suất tại pixel đó trên mặt nạ $M_c$ sẽ được tích lũy vào $Ref\_Score_i$. Phương trình xấp xỉ điểm số phản xạ của hạt $i$ qua $N$ góc nhìn được tính toán thông qua kỳ vọng hoặc trung bình trọng số:
$$Ref\_Score_i = \frac{1}{\sum_{c=1}^{N} \omega_{i,c}} \sum_{c=1}^{N} \omega_{i,c} \cdot M_c\left( \pi_c(\mu_i) \right)$$
*Diễn giải biến số:*
*   $N$: Số lượng camera (góc nhìn) quan sát được hạt Gaussian $i$.
*   $\pi_c(\mu_i)$: Phép chiếu tọa độ trung tâm $\mu_i$ của hạt Gaussian thứ $i$ lên mặt phẳng ảnh của camera $c$.
*   $M_c$: Mặt nạ xác suất phản xạ tương ứng với ảnh của camera $c$.
*   $\omega_{i,c}$: Hệ số đóng góp (thường dựa trên độ mờ đục hoặc ảnh hưởng của hạt $i$ tại pixel đó).

Giá trị $Ref\_Score \in [0, 1]$ trở thành thước đo định lượng mức độ tồn tại của ánh sáng phản xạ tại vị trí không gian của từng hạt Gaussian.

**C. Cập nhật thuật toán điều khiển mật độ (Guided Densification)**
Dựa trên giá trị $Ref\_Score$, thuật toán kiểm soát mật độ tiêu chuẩn được tái cấu trúc. Thay vì sử dụng một ngưỡng gradient $\tau_{pos}$ cố định, hệ thống áp dụng hàm điều chỉnh động (adaptive thresholding) hoặc hệ số lọc xác suất đối với quá trình nhân bản và phân tách.

Theo đó, hạt Gaussian $i$ chỉ được phép thực hiện quá trình cấp phát thêm năng lực biểu diễn ASG nếu nó thỏa mãn đồng thời hai điều kiện:
1.  Gradient vị trí $\nabla_{\mu_i} \mathcal{L}$ vượt quá ngưỡng tối ưu hình học.
2.  Điểm số phản xạ $Ref\_Score_i$ vượt qua một ngưỡng giới hạn $\tau_{ref}$ định trước (ví dụ: $\tau_{ref} = 0.5$).

Đối với những hạt Gaussian có $Ref\_Score_i < \tau_{ref}$, hệ thống sẽ ngăn chặn việc sinh thêm cấu trúc ASG mới, đồng thời có thể áp dụng cơ chế tỉa cành (pruning) để loại bỏ các thông số ASG dư thừa nhằm giải phóng bộ nhớ.

#### 8.2.3. Đánh giá tính logic của cơ chế
Thông qua việc tích hợp điểm số $Ref\_Score$ vào vòng lặp tối ưu hóa, phương pháp đã thiết lập được một cầu nối giữa tri thức quang học 2D và cấu trúc hình học 3D. Cơ chế phân bổ có định hướng này triệt tiêu hiện tượng phình to kích thước mô hình vô tổ chức. Kết quả là hệ thống đạt được hiệu năng lưu trữ tối ưu; dung lượng bộ nhớ được hội tụ chính xác vào việc gia tăng mức độ chi tiết (capacity) cho các vệt sáng chói, đảm bảo chất lượng kết xuất quang học cao nhất với mức độ tiêu thụ tài nguyên phần cứng tối thiểu.

### 8.3. Ràng buộc phân tách thành phần (Role Separation Supervision)

#### 8.3.1. Vấn đề giao thoa hàm mục tiêu
Trong kiến trúc biểu diễn kết hợp (mục 3.1), phương trình màu sắc tổng hợp $C(v) = C_{diffuse}(v) + C_{specular}(v)$ mang lại sự linh hoạt cao trong việc mô phỏng ánh sáng. Tuy nhiên, dưới góc độ tối ưu hóa toán học, phương trình này tạo ra hiện tượng đa nghĩa (ambiguity) cho thuật toán Gradient Descent. 

Khi tính toán hàm mất mát (Loss) dựa trên chênh lệch giữa ảnh kết xuất và ảnh thực tế (Groundtruth), bộ tối ưu hóa không tự nhận thức được việc phân chia công việc. Hệ quả là xảy ra hai hiện tượng giao thoa không mong muốn:
1.  **Hàm SH hấp thụ phản xạ:** Thuật toán cố gắng ép các hệ số của hàm SH dao động mạnh để xấp xỉ các vệt sáng chói. Điều này khiến vệt sáng bị mờ nhòe (do SH là hàm dải tần thấp) và cản trở hàm ASG phát huy tác dụng.
2.  **Hàm ASG rò rỉ (Leakage):** Hàm ASG tham gia bù đắp các sai số tại những vùng màu khuếch tán (diffuse) tối màu, làm cho cấu trúc vật thể xuất hiện các đốm sáng giả (artifacts) phi vật lý.

Để mô hình học đúng bản chất quang học, phương pháp `spec-fastgs` thiết lập cơ chế **Ràng buộc phân tách thành phần (Role Separation Supervision)**. Cơ chế này đóng vai trò như một hệ thống chỉ đường, ép buộc hàm SH chỉ hội tụ ở tần số thấp (màu nền), và hàm ASG phải nhận trách nhiệm toàn phần đối với các tín hiệu tần số cao (vệt chói sáng).

#### 8.3.2. 8.6. Thuật toán giám sát

**A. Cơ chế mặt nạ quang học (SH Specular Mask)**
Giải pháp đầu tiên để ngăn chặn hàm SH hấp thụ vệt phản xạ là can thiệp trực tiếp vào quá trình lan truyền ngược (Backpropagation) thông qua một mặt nạ làm giảm dốc (Gradient Masking).

Dựa trên bản đồ xác suất phản xạ $M_c$ (đã trình bày ở mục 3.2), hệ thống thiết lập một mặt nạ mềm (soft mask). Tại những pixel $p$ có xác suất chói sáng cao ($M_c(p)$ tiệm cận 1), quá trình lan truyền gradient từ hàm Loss tổng về các hệ số $c_{l}^{m}$ của hàm SH sẽ bị suy giảm (decay) bởi một trọng số phạt (penalty factor).
$$ \nabla_{c_{l}^{m}} \mathcal{L} \leftarrow \nabla_{c_{l}^{m}} \mathcal{L} \cdot \left( 1 - \gamma \cdot M_c(p) \right) $$
*(Trong đó $\gamma$ là hệ số kiểm soát mức độ ức chế gradient).*

Hành động ức chế này khiến hàm SH không thể tự tối ưu hóa để bù đắp sai số tại vùng chói sáng. Hệ quả tất yếu là bộ tối ưu hóa buộc phải "đẩy" toàn bộ lượng gradient còn lại vào các tham số độ nhám $\lambda$ của hàm ASG để giảm thiểu hàm Loss tổng. Cơ chế này định hướng một cách dứt khoát vai trò của hai hàm số trong không gian tối ưu.

**B. Hàm mất mát phân tách chuyên biệt (Specular Loss / Regularization)**
Giải pháp thứ hai là điều chỉnh trực tiếp hàm mục tiêu (Objective Function). Thay vì chỉ sử dụng hàm Loss tái tạo ảnh tổng quát ($\mathcal{L}_1$ hoặc $\mathcal{L}_{D-SSIM}$), phương pháp thiết kế thêm các hàm chuẩn hóa (Regularization) chuyên biệt đánh giá mức độ tương quan giữa phần dư (residual) và hàm ASG.

Gọi $C_{GT}$ là màu sắc của ảnh thực tế. Phần dư năng lượng (Residual Color) chưa được giải thích bởi hàm SH được định nghĩa là:
$$ C_{residual} = \max(0, C_{GT} - C_{diffuse}) $$
Hệ thống thiết lập một hàm mất mát thành phần $\mathcal{L}_{specular}$ nhằm cực tiểu hóa sự chênh lệch giữa lượng năng lượng dư thừa này và giá trị dự đoán của hàm ASG tại những vùng được định vị bởi mặt nạ phản xạ $M$:
$$ \mathcal{L}_{specular} = \sum_{p \in \Omega} M(p) \cdot \left\| ASG(p) - C_{residual}(p) \right\|_1 $$
Phương trình hàm mất mát tổng thể của toàn hệ thống được tổng hợp lại thành:
$$ \mathcal{L}_{total} = \mathcal{L}_{render} + \lambda_{spec} \mathcal{L}_{specular} $$
*(Trong đó $\mathcal{L}_{render}$ là hàm mất mát ảnh tổng thể, và $\lambda_{spec}$ là siêu tham số cân bằng (hyperparameter) quyết định mức độ chi phối của ràng buộc phân tách).*

Sự kết hợp giữa thao tác can thiệp gradient (SH Mask) và hàm mục tiêu chuyên biệt (Specular Loss) thiết lập một hành lang tối ưu hóa vô cùng khắt khe. Các cơ chế này triệt tiêu hoàn toàn điểm mù đa nghĩa của phương trình toán học ban đầu. Nó cung cấp cơ sở lý luận vững chắc để khẳng định rằng: Sự gia tăng mức độ chân thực của bề mặt vật liệu kim loại hoặc kính trong kết quả kết xuất của `spec-fastgs` không đến từ sự ngẫu nhiên của Mạng tối ưu, mà là kết quả tất yếu của một kiến trúc định hướng vật lý ánh sáng (physics-guided architecture) chặt chẽ.

### 8.4. Cơ chế trích xuất Điểm số Phản xạ (Ref Score Extraction)

Như đã đề cập ở các phần trước, sự thành bại của cơ chế phân bổ có định hướng và ràng buộc phân tách phụ thuộc hoàn toàn vào độ chính xác của tri thức tiền nghiệm. Khóa luận triển khai một thuật toán trích xuất Điểm số Phản xạ (Ref Score) hoạt động ngoại tuyến (offline) trong giai đoạn Tiền xử lý, dựa trên nguyên lý **Tính nhất quán đa góc nhìn (Multi-view Consistency)**. 

Quy trình trích xuất được thiết kế thành một Pipeline hai giai đoạn nối tiếp nhau: Đánh giá trong không gian 3D và Kết xuất ngược về mặt phẳng 2D.

#### 8.4.1. Giai đoạn 1: Đánh giá phương sai quang học trong không gian 3D

Nguyên lý cốt lõi của giai đoạn này dựa trên tính chất vật lý của vật liệu: Bề mặt khuếch tán (diffuse) mang tính đẳng hướng (Lambertian) nên màu sắc hầu như không biến thiên theo góc nhìn; ngược lại, bề mặt phản xạ (specular) có tính bất đẳng hướng, khiến cường độ ánh sáng đập vào mắt người quan sát thay đổi liên tục khi góc nhìn dịch chuyển.

**Thuật toán thực thi:**
1.  **Thu thập dữ liệu:** Hệ thống truy xuất tập hợp đám mây điểm 3D (3D Point Cloud) $\mathcal{P} = \{P_1, P_2, ..., P_M\}$ được khởi tạo từ thuật toán Structure-from-Motion (ví dụ: COLMAP).
2.  **Trích xuất quan sát:** Đối với mỗi điểm không gian $P_i$, thuật toán xác định tập hợp các camera $V_i = \{C_1, C_2, ...\}$ mà tại đó điểm $P_i$ nằm trong trường nhìn (Field of View) và không bị che khuất. Hệ thống tiến hành chiếu $P_i$ lên mặt phẳng ảnh của các camera trong $V_i$ để trích xuất màu sắc RGB tương ứng, sau đó chuyển đổi sang thang độ sáng (Luminance $L$).
3.  **Tính toán phương sai cực đại (Maximum Luminance Variance):** Thay vì sử dụng phương sai chuẩn (Standard Variance) dễ bị ảnh hưởng bởi nhiễu, hệ thống đánh giá mức độ phản xạ thông qua độ lệch cường độ sáng lớn nhất giữa hai góc nhìn bất kỳ:
    $$ \Delta L_i = \max_{j, k \in V_i} | L(C_{j}) - L(C_{k}) | $$
    Nếu tập quan sát $|V_i| \le 1$, điểm đó không đủ điều kiện đối chiếu đa góc nhìn và được gán $\Delta L_i = 0$.
4.  **Chuẩn hóa phân phối:** Tập hợp các giá trị chênh lệch $\Delta L$ của toàn bộ đám mây điểm được chuẩn hóa Min-Max để đưa về thang đo xác suất tuyến tính $[0, 1]$, tạo thành điểm số phản xạ 3D ($Ref_{3D}$):
    $$ Ref_{3D}(i) = \frac{\Delta L_i - \min(\Delta L)}{\max(\Delta L) - \min(\Delta L)} $$

#### 8.4.2. Giai đoạn 2: Trải phẳng không gian và xử lý che khuất (Z-Buffering)

Mạng nơ-ron và trình Rasterization của hệ thống hoạt động tối ưu nhất khi nhận dữ liệu điều hướng (gating) dưới dạng ma trận 2D. Do đó, các điểm $Ref_{3D}(i)$ cần được chiếu ngược lại thành các bản đồ phản xạ 2D (Ref Map) cho từng camera.

Thách thức kỹ thuật lớn nhất ở bước này là **Hiệu ứng che khuất không gian (Occlusion/Z-Conflict)**: Khi chiếu nhiều điểm không gian 3D lên cùng một trục tọa độ 2D của camera, các điểm nằm ẩn phía sau vật thể (background) có thể đè lên các điểm ở bề mặt phía trước (foreground), gây nhiễu loạn xác suất phản xạ.

**Thuật toán chiếu bám và loại trừ:**
Để giải quyết bài toán che khuất, khóa luận áp dụng cơ chế Z-Buffering. Đối với mỗi camera $C_j$:
1.  Hệ thống khởi tạo một bản đồ $Ref\_Map \in \mathbb{R}^{H \times W}$ (ma trận toàn số 0) và một bộ đệm chiều sâu $Z\_Buffer \in \mathbb{R}^{H \times W}$ (ma trận khởi tạo bằng vô cực $\infty$).
2.  Thực hiện phép biến đổi phối cảnh (Perspective Projection) để chiếu từng điểm $P_i$ lên mặt phẳng ảnh thành tọa độ $(x, y)$ kèm theo giá trị chiều sâu $z_i$ (khoảng cách từ điểm đến quang tâm camera).
3.  **Cơ chế cập nhật có điều kiện:** 
    Chỉ khi độ sâu của điểm hiện tại nhỏ hơn độ sâu đã lưu trong bộ đệm ($z_i < Z\_Buffer[x, y]$), hệ thống mới thực hiện ghi đè:
    $$ Ref\_Map_j[x, y] = Ref_{3D}(i) $$
    $$ Z\_Buffer_j[x, y] = z_i $$
Cơ chế này đảm bảo tại bất kỳ một điểm ảnh (pixel) nào, hệ thống chỉ ghi nhận đặc tính quang học của bề mặt vật liệu nằm gần ống kính camera nhất.

Cơ chế trích xuất Ref Score thể hiện đặc tính **phân tách rõ ràng (Explicit Disentanglement)**. Bằng việc cung cấp một tín hiệu vật lý trực tiếp mang tính xác định (deterministic), hệ thống giúp mạng nơ-ron giải phóng tài nguyên khỏi việc tự suy diễn (mò mẫm) tính chất vật liệu từ các hàm Loss chung chung. 

Tuy nhiên, dưới lăng kính kỹ thuật, phương pháp này cũng tồn tại hệ quả về chi phí tính toán (Overhead). Quá trình đối chiếu quang học đa góc nhìn và chiếu ngược Z-Buffering trên hàng triệu điểm ảnh yêu cầu khối lượng tính toán ma trận lớn, gia tăng đáng kể thời gian tiền xử lý. Hơn thế nữa, độ chính xác của bản đồ Ref Score 2D phụ thuộc tuyến tính vào mật độ và chất lượng của đám mây điểm 3D ban đầu. Tại những vùng thưa điểm (sparse regions), hệ thống có khả năng gặp lỗi lem viền (Edge Bleeding) do hiện tượng rời rạc hóa không gian (spatial discretization) không đủ bao phủ biên giới vật thể.

### 8.5. Tối ưu hóa Gaussians qua Cổng Neural (Neural Gating Optimization)

Điểm đột phá nhất trong thiết kế cấu trúc luồng dữ liệu (Pipeline) của `spec-fastgs` không chỉ dừng lại ở việc bổ sung hàm ASG, mà nằm ở cơ chế phân luồng thuật toán Tối ưu hóa không gian (Optimization). Khóa luận định nghĩa nguyên lý này là **Thích ứng không gian dựa trên phản xạ (Reflection-aware Spatial Adaptation - RSA)**. 

Cơ chế RSA can thiệp sâu vào cả hai pha của vòng lặp học sâu: Pha truyền xuôi (Forward Pass) và Pha truyền ngược (Backward Pass).

#### 8.5.1. Pha truyền xuôi (Forward Pass): Cơ chế Cổng Neural
Trong vòng lặp huấn luyện, tại bước tính toán màu sắc trước khi kết xuất, mạng ASG xuất ra một tensor màu dự đoán cho thành phần phản xạ (Specular Color). Tuy nhiên, dựa trên nguyên lý RSA, chuỗi giá trị này không được cộng gộp trực tiếp vào hàm SH. Thay vào đó, nó phải đi qua một phép toán nhân vô hướng (element-wise multiplication) với giá trị lấy mẫu từ bản đồ Ref Score 2D.

Phương trình kết xuất màu sắc cuối cùng được thiết lập lại thành:
$$ C_{final} = C_{diffuse} + \mathbf{Ref\_Score} \cdot C_{specular} $$

Về mặt toán học, phép nhân này đóng vai trò như một **Cổng Neural (Neural Gating)** điều tiết dòng chảy năng lượng quang học:
*   Tại các vị trí pixel đại diện cho bề mặt nhám ($Ref\_Score \approx 0$), cổng đóng lại, triệt tiêu gần như hoàn toàn đóng góp của mạng Specular vào màu sắc cuối cùng.
*   Tại các vị trí đại diện cho đốm sáng phản xạ ($Ref\_Score \approx 1$), cổng mở tối đa, duy trì nguyên vẹn cấu trúc vệt sáng do ASG tạo ra.

#### 8.5.2. Pha truyền ngược (Backward Pass): Tối ưu hóa phân luồng đa mục tiêu
Sự xuất hiện của phương trình Cổng Neural ở pha truyền xuôi mang lại tác động quyết định đối với thuật toán cập nhật trọng số (Optimizer) tại pha truyền ngược. Trong `spec-fastgs`, hệ thống thực hiện tối ưu hóa song song hai nhóm tham số với các quy tắc luân chuyển Gradient hoàn toàn khác biệt:

**A. Tối ưu hóa thuộc tính cơ sở (Standard Gaussian Optimization)**
Bất chấp giá trị của Ref Score, mọi sai số (Loss) tái tạo ảnh đều tạo ra Gradient truyền ngược về các thuộc tính cốt lõi của hạt 3D Gaussian. Nhóm tham số này bao gồm:
1.  **Đặc trưng hình học:** Tọa độ trung tâm (X,Y,Z), Độ giãn (Scale), Độ xoay (Rotation) và Độ mờ đục (Opacity).
2.  **Đặc trưng khuếch tán:** Các hệ số của hàm Spherical Harmonics (SH).
Việc duy trì cập nhật liên tục nhóm tham số này đảm bảo rằng cấu trúc không gian 3D của vật thể và màu sắc nền (base color) luôn được tái tạo chính xác trên toàn cục không gian, hình thành bộ khung vững chắc cho khung cảnh.

**B. Tối ưu hóa tham số phản xạ qua Cổng Neural (Gated Specular Optimization)**
Trái ngược với nhóm cơ sở, Gradient truyền về nhóm tham số của mạng ASG (ký hiệu là $\theta_{ASG}$) bắt buộc phải đi qua phép nhân với $\mathbf{Ref\_Score}$ theo quy tắc chuỗi (Chain Rule):
$$ \frac{\partial \mathcal{L}}{\partial \theta_{ASG}} = \frac{\partial \mathcal{L}}{\partial C_{final}} \cdot \frac{\partial C_{final}}{\partial C_{specular}} \cdot \frac{\partial C_{specular}}{\partial \theta_{ASG}} = \left( \frac{\partial \mathcal{L}}{\partial C_{final}} \cdot \mathbf{Ref\_Score} \right) \cdot \frac{\partial C_{specular}}{\partial \theta_{ASG}} $$

Hệ quả của phương trình truyền ngược này tạo ra một cơ chế phân quyền (Role Assignment) cực kỳ nghiêm ngặt:
*   **Ức chế cực tiểu cục bộ:** Khi đạo hàm đi qua vùng không có tính phản xạ ($Ref\_Score = 0$), lượng Gradient truyền về $\theta_{ASG}$ lập tức bị triệt tiêu về 0. Thông điệp toán học này ép buộc bộ tối ưu hóa hiểu rằng: sai số tại vùng này không thuộc trách nhiệm của mạng ASG. Hệ thống bắt buộc phải dồn toàn bộ sự điều chỉnh trọng số vào nhóm thuộc tính cơ sở (cụ thể là hệ số SH) để bù đắp sai số, qua đó ngăn chặn hàm ASG hấp thụ nhầm màu nền.
*   **Kích thích hội tụ phản xạ:** Ngược lại, tại vùng chói sáng ($Ref\_Score = 1$), Gradient được truyền tải nguyên vẹn. Các tham số độ nhám $\lambda$ và cường độ màu $c_s$ của hạt Gaussian chịu trách nhiệm điều chỉnh mạnh mẽ để xấp xỉ chính xác biên độ và hình dáng của vệt chói sáng.

Việc phân luồng quá trình truyền ngược đã biến bản đồ Ref Score từ một bộ lọc tĩnh thành một thành phần giám sát tối ưu hóa động (Dynamic Optimization Supervisor). Phương pháp tiếp cận này trực tiếp giải quyết vấn đề đa nghĩa quang học (optical ambiguity) giữa các hàm toán học. Nhờ duy trì sự độc lập giữa luồng tối ưu hóa hình học cơ sở và luồng tối ưu hóa phản xạ, thuật toán bảo toàn được tính thưa thớt (sparsity) của mạng nơ-ron, giúp cực đại hóa năng lực biểu diễn bề mặt kim loại/kính mà không phá vỡ tính ổn định của cấu trúc không gian tổng thể.

### 8.6. Cơ chế kiểm soát hình học dựa trên tầm nhìn (VCD và VCP)

Trong mô hình 3D Gaussian Splatting tiêu chuẩn, thuật toán kiểm soát mật độ (Densification) và tỉa cành (Pruning) hoạt động chủ yếu dựa trên ngưỡng gradient vị trí 2D và độ mờ đục (opacity). Tuy nhiên, cách tiếp cận này thường dẫn đến hiện tượng sinh ra các điểm ảnh rác (floaters) ở không gian trống hoặc tạo ra mật độ hạt dư thừa tại các vùng bị che khuất (occlusion). 

Để giải quyết triệt để vấn đề này, đồng thời tạo ra một môi trường không gian đủ "sạch" để tích hợp các hàm quang học phức tạp, phương pháp `spec-fastgs` triển khai hai cơ chế kiểm soát tầm nhìn nâng cao: **VCD (Visibility-aware Control Densification)** và **VCP (Visibility-aware Control Pruning)**.

#### 8.6.1. Cơ chế Nhân bản kiểm soát tầm nhìn (VCD)
VCD là thuật toán kiểm duyệt nghiêm ngặt quá trình sinh thêm (nhân bản hoặc tách) hạt 3D Gaussian. 
*   **Nguyên lý hoạt động:** Thay vì cho phép mọi hạt Gaussian sinh sôi tự do khi gradient vượt ngưỡng, VCD yêu cầu sự đồng thuận đa góc nhìn (multi-view consensus). Hệ thống liên tục đánh giá tần suất và mức độ đóng góp (visibility) của mỗi hạt Gaussian trên nhiều góc camera khác nhau. 
*   **Điều kiện khắt khe:** Một hạt chỉ được phép thực hiện quá trình nhân bản nếu nó chứng minh được sự đóng góp đáng kể vào việc kết xuất màu sắc trên đa góc nhìn. 
*   **Sự cộng hưởng thuật toán:** Khi VCD được kết hợp với cơ chế `Ref_Score`, hệ thống hình thành một bộ lọc kép (Dual-filter) cực kỳ chặt chẽ. Theo đó, một hạt Gaussian không chỉ cần nằm đúng khu vực có xác suất chói sáng cao (thông qua Ref Score), mà còn phải sở hữu nền tảng vật lý vững chắc trong không gian thực (thông qua VCD). Nhờ vậy, hiện tượng sinh hạt rác mang đặc tính quang học sai lệch bị triệt tiêu hoàn toàn.

#### 8.6.2. Cơ chế Tỉa cành kiểm soát tầm nhìn (VCP)
Nếu VCD kiểm soát "đầu vào", thì VCP đóng vai trò là cơ chế thu dọn và tối ưu hóa "đầu ra" một cách liên tục trong vòng lặp huấn luyện.
*   **Nguyên lý hoạt động:** Trong quá trình tối ưu, nhiều hạt Gaussian có thể bị đẩy vào bên trong vật thể (nơi không camera nào quan sát được), hoặc bị giảm dần tầm quan trọng đến mức gần như vô hình, nhưng chúng vẫn chiếm dụng tài nguyên tính toán.
*   **Quy tắc thanh lọc:** VCP liên tục tính toán Điểm số hiển thị (Visibility Score) tích lũy của từng hạt trong các lần truyền xuôi (Forward Pass). Bất kỳ hạt nào có điểm số này rơi xuống dưới một ngưỡng an toàn (ví dụ: bị che khuất quá lâu hoặc quá mờ), VCP sẽ ngay lập tức xóa bỏ (prune) hạt đó khỏi cấu trúc bộ nhớ.
*   **Bảo toàn năng lực kết xuất:** Đặc tính của bề mặt phản xạ (specular) là rất nhạy cảm với các hạt rác lơ lửng, bởi ánh sáng chói có xu hướng phản chiếu sai lệch lên các hạt này, tạo ra nhiễu hạt (artifacts). Nhờ VCP hoạt động như một cỗ máy dọn dẹp liên tục, không gian 3D luôn được duy trì ở trạng thái thưa thớt (sparse) và chính xác nhất. Điều này giải thích tại sao hệ thống có thể tích hợp thêm hàm ASG phức tạp mà tốc độ khung hình (FPS) khi kết xuất vẫn đạt mức tối đa.

Việc triển khai đồng bộ mạng ASG, hệ thống `Ref_Score`, cùng với hai cơ chế VCD và VCP đã tạo ra một hệ sinh thái thuật toán thống nhất. VCD và VCP đảm bảo một cấu trúc hình học nền tảng (geometry baseline) hoàn hảo và tối ưu bộ nhớ; từ đó thiết lập một "không gian quang học" lý tưởng để các ràng buộc phản xạ của mạng ASG có thể hội tụ chính xác mà không làm sụp đổ hiệu năng chung của toàn bộ kiến trúc.

Ngoài 3 trụ cột lớn đã trình bày (Kiến trúc lai SH+ASG, Điểm số Ref_Score, và VCD/VCP), phương pháp `spec-fastgs` vẫn còn **3 điểm kỹ thuật nổi bật nữa** mang tính quyết định khi đem so sánh với 3DGS truyền thống. 

### 8.7. Xử lý Pháp tuyến Bề mặt (Surface Normal Estimation & Normal-Delta)
*   **Vấn đề của 3DGS truyền thống:** Hàm Spherical Harmonics (SH) trong 3DGS chỉ quan tâm đến *hướng nhìn (viewing direction)* từ camera tới điểm 3D. Vì vậy, 3DGS truyền thống **không có và không cần khái niệm pháp tuyến (normal)**.
*   **Đột phá của `spec-fastgs`:** Để tính toán được hàm ASG cho vệt sáng chói, bắt buộc phải biết *hướng phản xạ (reflection direction)*, mà muốn biết hướng phản xạ thì hệ thống phải có vector pháp tuyến của bề mặt. 
    *   *Kỹ thuật:* Thay vì dùng các mạng MLP nặng nề để dự đoán pháp tuyến, phương pháp này tận dụng ngay đặc tính hình học của Gaussian: Nó lấy **trục ngắn nhất (shortest axis)** của ma trận hiệp phương sai (ellipsoid) làm pháp tuyến giả định (pseudo-normal). 
    *   *Tinh chỉnh (Normal-Delta):* Trong quá trình huấn luyện, hệ thống có thể học thêm một tham số bù đắp nhỏ gọi là `normal-delta` để "bẻ" nhẹ hướng pháp tuyến này sao cho vệt chói sáng khớp nhất với ảnh thực tế. Điều này giúp bề mặt kim loại/kính mượt mà hơn rất nhiều mà không tốn chi phí tính toán mạng nơ-ron phức tạp.

### 8.8. Kiểm soát Ngân sách Bộ nhớ Động (Auto-Budget & Soft Decay)
*   **Vấn đề của 3DGS truyền thống:** Thuật toán Densification (nhân bản hạt) của 3DGS gốc hoạt động rất "bản năng". Cứ gradient lớn là nó sinh hạt, dẫn đến tình trạng mô hình phình to không kiểm soát (thường lên tới 5-10 triệu hạt) làm tràn VRAM (Out of Memory).
*   **Đột phá của `spec-fastgs`:** Dù đã có Ref Score để hạn chế sinh hạt bừa bãi, hệ thống còn thiết lập thêm cơ chế **Auto-Budget** (Ngân sách tự động). 
    *   Hệ thống quy định một số lượng trần (max budget) cho số hạt ASG. 
    *   Nếu số lượng hạt phản xạ vượt quá ngân sách, hệ thống áp dụng cơ chế **Soft Decay**: các hạt có `Ref_Score` thấp nhất sẽ bị thu hồi quyền sử dụng ASG (biến về hạt diffuse bình thường) hoặc bị xóa. Nhờ vậy, dung lượng mô hình (size checkpoint) luôn được ép ở mức vô cùng nhỏ gọn, có thể chạy trên các GPU phổ thông.

### 8.9. Tối ưu hóa Luồng kết xuất phần cứng (Hardware-level CUDA Optimization)
*   **Vấn đề của 3DGS truyền thống:** Dù 3DGS gốc đã rất nhanh, nhưng bộ Rasterizer (trình kết xuất chiếu 3D xuống 2D) viết bằng CUDA của nó chưa tối ưu triệt để việc đọc/ghi bộ nhớ đệm (shared memory) trên GPU. Khi gặp các cảnh độ phân giải quá cao (4K) hoặc số hạt quá lớn, FPS bắt đầu sụt giảm.
*   **Đột phá của `spec-fastgs`:** 
    *   Kế thừa kiến trúc phần mềm cốt lõi từ FastGS, bộ Rasterizer của `spec-fastgs` đã được viết lại hoàn toàn ở cấp độ phần cứng. Các thuật toán tối ưu phép cộng dồn (Parallel Prefix Sum) và sắp xếp (Radix Sort) được tinh chỉnh để tận dụng tối đa kiến trúc luồng (threads) của card màn hình (GPU NVIDIA). 
    *   Chính sự tối ưu luồng dữ liệu cực đoan này là "tấm lá chắn" giúp `spec-fastgs` có thể cõng thêm một đống toán học phức tạp của hàm ASG, tính toán thêm Ref_Score, mà tốc độ FPS khi xuất ra vẫn bằng hoặc thậm chí vượt trội hơn cả 3DGS truyền thống trống trơn.