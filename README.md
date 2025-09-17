# 🚀 Hệ Thống Chống Giả Mạo Khuôn Mặt Sử Dụng Mô Hình Sinh Mẫu (DFG & OA-Net)

## 📖 Giới Thiệu Dự Án

Đây là kho mã nguồn cho dự án xây dựng một hệ thống **Chống Giả mạo Khuôn mặt (Face Anti-Spoofing - FAS)** tiên tiến. Mục tiêu chính của dự án là xây dựng một mô hình mạnh mẽ, có khả năng phân biệt giữa các hình ảnh khuôn mặt thật (live) và các hình thức tấn công giả mạo đa dạng (ảnh in, phát lại trên màn hình, mặt nạ 3D, v.v.).

Dự án này triển khai một kiến trúc hai giai đoạn phức tạp, lấy cảm hứng từ phương pháp luận **AG-FAS**, khai thác sức mạnh của các mô hình sinh mẫu (Generative Models) và Vision Transformers với cơ chế Cross-Attention để đạt được độ chính xác và khả năng tổng quát hóa cao.

| Thông Tin      | Chi Tiết                                       |
| -------------- | ---------------------------------------------- |
| **👤 Tác giả**    | Bùi Thị Thanh Vân                              |
| **🎓 Trường**      | Đại học Giao thông Vận tải Thành phố Hồ Chí Minh |
| **📧 Email**      | `thanh.van19062004@gmail.com`                  |
| **📅 Thời gian** | Tháng 09, 2025                                 |

---

## 🛠️ Phương Pháp Luận Cốt Lõi

Hệ thống được xây dựng dựa trên một pipeline hai giai đoạn thông minh, biến bài toán phân loại thành bài toán "phát hiện bằng chứng".

**Luồng hoạt động tổng thể:**
`🖼️ Ảnh Gốc ➡️ 🧠 DFG ➡️ 🔬 Tín hiệu Bất thường ➡️ 🕵️‍♀️ OA-Net ➡️ ✅/❌ Kết quả`

### ### Giai đoạn 1: DFG (De-fake Face Generator) - Cỗ Máy Tạo Bằng Chứng

Giai đoạn đầu tiên sử dụng một **Mô hình Khuếch tán (Diffusion Model - DFG)** được huấn luyện **chỉ trên dữ liệu khuôn mặt thật**. Nhiệm vụ của nó không phải là phân loại, mà là học sâu về sự phân phối của một "khuôn mặt thật".

Đầu ra quan trọng nhất là **"Tín hiệu Bất thường" (Anomalous Cue)**, được tính bằng hiệu số tuyệt đối giữa ảnh gốc và ảnh được DFG tái tạo.
-   **Với ảnh Thật (Live):** Quá trình tái tạo gần như hoàn hảo, tạo ra một cue gần như **tối đen hoàn toàn**.
-   **Với ảnh Giả mạo (Spoof):** DFG sẽ "sửa chữa" các dấu hiệu giả mạo, dẫn đến sự khác biệt lớn và tạo ra một cue **sáng rực, làm nổi bật các vùng giả mạo**.

**Ví dụ Trực quan về Hoạt động của DFG:**

Bảng dưới đây minh họa cách DFG xử lý hai trường hợp thực tế: một ảnh thật ở tư thế khó (out-of-distribution) và một ảnh giả mạo.

| Loại | Ảnh Gốc | Ảnh Tái Tạo bởi DFG | Tín hiệu Bất thường (Anomalous Cue) | Phân Tích |
| :--- | :---: | :---: | :---: | :--- |
| **THẬT (Live)**<br>*(Ca khó)* | ![Live Sample Original](results/readme_assets/live_sample_difficult_case.png) | Mô hình thất bại trong việc tái tạo ảnh có tư thế lạ, cố gắng "ép" nó về một dạng chuẩn. | Kết quả là một cue rất sáng, có nguy cơ bị OA-Net nhận nhầm là giả mạo. | *Điều này cho thấy giới hạn của DFG với dữ liệu ngoài phân phối và là nguyên nhân tiềm tàng gây ra lỗi **BPCER** (nhận nhầm người thật).* |
| **GIẢ MẠO (Spoof)**<br>*(Ca thành công)* | ![Spoof Sample Original](results/readme_assets/spoof_sample_detection_case.png) | DFG "sửa chữa" ảnh giả mạo, tạo ra một phiên bản khuôn mặt trông "thật" hơn. | Sự khác biệt lớn tạo ra một cue rất sáng, làm nổi bật các vùng bất thường. | *Đây là "bằng chứng" rõ ràng để OA-Net học và phát hiện tấn công, giúp giảm lỗi **APCER** (bỏ lọt tấn công giả mạo).* |

### ### Giai đoạn 2: OA-Net (Off-real Attention Network) - Thám Tử Thông Minh

Giai đoạn thứ hai là một mô hình phân loại mạnh mẽ, học hỏi **từ các tín hiệu bất thường**. Dự án này triển khai một kiến trúc mới dựa trên **Vision Transformer (ViT)** cho OA-Net, được tăng cường bởi:
-   **ResNet Cue Encoder:** Một backbone CNN để trích xuất các đặc trưng không gian phong phú từ tín hiệu bất thường.
-   **Vision Transformer Backbone:** Xử lý tín hiệu bất thường để nắm bắt các mối quan hệ toàn cục.
-   **Cơ chế Cross-Attention:** Tại mỗi lớp của ViT, một module cross-attention cho phép mô hình kết hợp thông minh thông tin từ luồng chính của ViT với các đặc trưng chi tiết từ ResNet, giúp tập trung vào những dấu vết giả mạo quan trọng nhất.

---

## 📊 Phân Tích và Trực Quan Hóa Dữ Liệu

Dự án sử dụng kết hợp bộ dữ liệu FFHQ và CelebA-Spoof.

### 1. Nguồn Dữ Liệu Gốc

| Nguồn Dữ Liệu | Loại | Số Lượng |
| :--- | :--- | :--- |
| FFHQ | Live | 2,562 |
| CelebA | Live | 35,000 |
| CelebA | Spoof | 33,433 |
| **Tổng cộng** | | **71,000+** |

![Phân tích Nguồn Dữ liệu Gốc](results/charts/1_raw_data_sources.png)
![Tổng số ảnh Thật và Giả mạo Gốc](results/charts/2_raw_live_vs_spoof.png)

### 2. Sự Đa Dạng của Các Loại Tấn Công Giả Mạo

Bộ dữ liệu bao gồm 10 loại hình tấn công giả mạo khác nhau, đảm bảo mô hình được huấn luyện để chống lại nhiều mối đe dọa.

![Phân bố các Loại Tấn công Giả mạo Gốc](results/charts/4_spoof_type_distribution.png)

### 3. Dữ Liệu Đã Xử Lý và Phân Chia

Sau giai đoạn DFG, dữ liệu được chuyển đổi thành tín hiệu bất thường và cân bằng. Sau đó, tập dữ liệu được chia theo tỷ lệ 80/10/10, đảm bảo không có sự trùng lặp chủ thể (subject) giữa các tập.

![Số lượng Tín hiệu Bất thường Đã Xử lý](results/charts/3_processed_cues_count.png)
![Phân chia Tập Dữ liệu Cuối cùng](results/charts/5_dataset_split_pie_chart.png)

---

## 📈 Hiệu Suất Mô Hình và Đánh Giá

Mô hình OA-Net được đánh giá trên tập kiểm tra (test set) chưa từng thấy.

### 1. Đường Cong Huấn Luyện (Training Curves)

Mô hình được huấn luyện trong 15 epochs, với cơ chế **Dừng Sớm (Early Stopping)** được kích hoạt khi loss trên tập kiểm định bắt đầu tăng. Mô hình tốt nhất đã đạt được tại **Epoch 1**, cho thấy khả năng học cực nhanh của kiến trúc ViT.

![Đường cong Huấn luyện và Kiểm định](results/charts/1_training_curves.png)

### 2. Các Chỉ Số Hiệu Suất trên Tập Test

Bảng dưới đây tổng hợp các chỉ số hiệu suất cuối cùng, là tiêu chuẩn cho lĩnh vực Face Anti-Spoofing.

| Chỉ Số | Giá Trị | Ý nghĩa & Tầm quan trọng |
| :--- | :--- | :--- |
| **Accuracy** | **84.10%** | Tỷ lệ dự đoán đúng trên tổng số mẫu. |
| **APCER** | **6.85%** | **(Lỗi An ninh)** Tỷ lệ tấn công giả mạo bị bỏ lọt (nhận nhầm là thật). **Càng thấp càng tốt.** |
| **BPCER** | **24.67%** | **(Lỗi Trải nghiệm)** Tỷ lệ người dùng thật bị từ chối (nhận nhầm là giả). **Càng thấp càng tốt.** |
| **ACER** | **15.76%** | **(Lỗi Trung bình)** Thước đo cân bằng giữa APCER và BPCER, thể hiện hiệu suất tổng thể. |

### 3. Ma Trận Nhầm Lẫn (Confusion Matrix)

Ma trận nhầm lẫn cung cấp cái nhìn chi tiết về các dự đoán của mô hình trên 742 mẫu trong tập kiểm tra.

![Heatmap Ma trận Nhầm lẫn](results/charts/6_confusion_matrix_heatmap.png)

**Phân tích:**
-   **Đúng:** Mô hình nhận diện đúng **301/377** trường hợp thật và **325/365** trường hợp giả mạo.
-   **Điểm yếu:** Tỷ lệ nhận nhầm người dùng thật (BPCER) còn tương đối cao, cho thấy mô hình DFG có thể quá nhạy cảm với các ảnh thật có chất lượng thấp hoặc góc chụp lạ.

---

## 🚀 Yêu Cầu Hệ Thống và Cài Đặt

### Yêu Cầu
- Python 3.8+
- PyTorch 1.10+
- `transformers`, `diffusers`
- `opencv-python`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

### Cài Đặt
```bash
# Clone kho mã nguồn
git clone https://github.com/vanujiash9/FAS_DFG-OANET_Project.git
cd FAS_DFG-OANET_Project

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
Hướng Dẫn Chạy
Huấn Luyện:
code
Bash
# (Tùy chọn) Chạy toàn bộ pipeline huấn luyện DFG và OA-Net
python3 src/scripts/run_full_pipeline.py
Dự Đoán (Inference):
code
Bash
# Chạy ở chế độ tương tác (upload file vào thư mục /uploads)
python3 src/scripts/predict.py
Tạo Báo Cáo:
code
Bash
# Tái tạo tất cả biểu đồ và báo cáo từ các model và log đã có
python3 src/scripts/generate_full_report.py
🎯 Kết Luận và Hướng Phát Triển Tương Lai
Dự án đã triển khai thành công một hệ thống Chống Giả mạo Khuôn mặt phức tạp và hiện đại. Mô hình cho thấy sự hiểu biết sâu sắc về các dấu hiệu giả mạo thông qua việc sử dụng các tín hiệu bất thường được sinh ra.
Các hướng cải thiện trong tương lai:
Cải thiện BPCER: Huấn luyện DFG trên một tập dữ liệu thật đa dạng hơn (với nhiều điều kiện ánh sáng, góc độ, và biểu cảm) để giảm tỷ lệ nhận diện nhầm người dùng thật.
Đánh giá Chéo Miền (Cross-Domain): Kiểm tra mô hình trên các bộ dữ liệu hoàn toàn khác (ví dụ: OULU-NPU) để đánh giá khả năng tổng quát hóa.
Chống các loại Tấn công Hiện đại: Tích hợp dữ liệu chứa các cuộc tấn công deepfake và bộ lọc AR để tăng cường sức mạnh của mô hình.
Tối ưu hóa cho Triển khai Thực tế: Áp dụng kỹ thuật Chưng cất Mô hình (Model Distillation) để tạo ra một mô hình "học trò" gọn nhẹ, có khả năng triển khai thời gian thực trên các thiết bị di động.
---```
