# Báo Cáo Kỹ Thuật: Hệ Thống Chống Giả Mạo Khuôn Mặt Sử Dụng Mô Hình Khuếch Tán và Vision Transformer

## 1. Giới Thiệu Dự Án

### 1.1. Bối cảnh và Mục tiêu
Dự án này trình bày quá trình nghiên cứu, thiết kế và triển khai một hệ thống **Chống Giả mạo Khuôn mặt (Face Anti-Spoofing - FAS)** tiên tiến. Trong bối cảnh các hệ thống xác thực sinh trắc học ngày càng phổ biến, việc đảm bảo khả năng chống lại các cuộc tấn công giả mạo (spoof attacks) là một yêu cầu an ninh tối quan trọng. Mục tiêu của dự án là xây dựng một mô hình không chỉ đạt độ chính xác cao mà còn có khả năng tổng quát hóa tốt trước các loại hình tấn công đa dạng, bao gồm ảnh in, phát lại trên màn hình, và mặt nạ 3D.

Để đạt được mục tiêu này, dự án triển khai một kiến trúc hai giai đoạn phức tạp, lấy cảm hứng từ phương pháp luận **AG-FAS**, nhưng được cải tiến bằng cách sử dụng **Vision Transformer (ViT)** làm backbone chính, kết hợp với cơ chế **Cross-Attention** để tăng cường khả năng phân tích và nhận diện.

### 1.2. Thông Tin Tác Giả
|               |                                                |
| :------------ | :--------------------------------------------- |
| **Tác giả**   | Bùi Thị Thanh Vân                              |
| **Trường**    | Đại học Giao thông Vận tải Thành phố Hồ Chí Minh |
| **Email**     | `thanh.van19062004@gmail.com`                  |
| **Thời gian** | Tháng 09, 2025                                 |

---

## 2. Môi Trường và Hạ Tầng Thực Nghiệm

Do yêu cầu tính toán cực lớn của các mô hình Diffusion và Transformer, toàn bộ quá trình thực nghiệm của dự án được thực hiện trên một hệ thống máy chủ chuyên dụng thuê ngoài với cấu hình chi tiết như sau:

| Thành Phần        | Chi Tiết Kỹ Thuật                                       |
| :---------------- | :------------------------------------------------------ |
| **GPU**           | 2 x NVIDIA GeForce RTX 4090 (24GB VRAM mỗi card)          |
| **Hệ điều hành**  | Ubuntu 22.04 LTS (Linux)                                |
| **Phiên bản CUDA** | 12.x                                                    |
| **Ngôn ngữ**      | Python 3.10                                             |
| **Frameworks**    | PyTorch, Hugging Face Transformers, Diffusers           |

---

## 3. Dữ Liệu và Quy Trình Tiền Xử Lý

### 3.1. Nguồn Dữ Liệu Gốc
Dự án sử dụng một bộ dữ liệu lớn và đa dạng, được tổng hợp từ hai nguồn chính:
-   **FFHQ (Flickr-Faces-HQ):** Cung cấp một lượng lớn ảnh khuôn mặt thật chất lượng cao.
-   **CelebA-Spoof:** Một bộ dữ liệu tiêu chuẩn trong lĩnh vực FAS, cung cấp cả ảnh thật và 10 loại tấn công giả mạo khác nhau.

### 3.2. Quy Trình Tiền Xử Lý
Tất cả ảnh đầu vào đều trải qua một pipeline tiền xử lý chuẩn hóa để loại bỏ các biến thiên không mong muốn:
1.  **Phát hiện và Căn chỉnh:** Sử dụng mô hình MTCNN để phát hiện khuôn mặt và các điểm mốc (landmarks).
2.  **Cắt (Cropping):** Cắt ảnh để chỉ giữ lại vùng mặt, loại bỏ phần lớn nền.
3.  **Thay đổi kích thước (Resizing):** Resize tất cả các ảnh về kích thước `224x224` pixels để phù hợp với đầu vào của ViT.
4.  **Chuẩn hóa (Normalization):** Chuyển giá trị pixel về khoảng `[-1, 1]`.

### 3.3. Phân Chia Dữ Liệu
Để đảm bảo tính khách quan của kết quả, dữ liệu được chia theo phương pháp **phân chia theo định danh (subject-disjoint)**. Tức là, một người (subject) đã xuất hiện trong tập huấn luyện sẽ không bao giờ xuất hiện trong tập kiểm định hoặc kiểm tra. Tỷ lệ phân chia cuối cùng là **80% Training, 10% Validation, và 10% Test**.

---

## 4. Phương Pháp Luận Chi Tiết Về Mô Hình

### 4.1. Tổng quan kiến trúc
Hệ thống bao gồm hai mô hình hoạt động tuần tự. DFG đóng vai trò là một "bộ tiền xử lý thông minh", biến đổi ảnh đầu vào thành một dạng "bằng chứng" dễ phân tích hơn. OA-Net sau đó sẽ phân tích bằng chứng này để đưa ra kết luận.

`Ảnh Gốc ➡️ DFG ➡️ Tín hiệu Bất thường (Anomalous Cue) ➡️ OA-Net ➡️ Kết quả`

### 4.2. Giai đoạn 1: DFG (De-fake Face Generator) - Mô hình Sinh Tín hiệu Bất thường

-   **Mục tiêu:** DFG không học về ảnh giả mạo. Nó được huấn luyện chỉ trên dữ liệu thật để học một cách sâu sắc về sự phân phối của một "khuôn mặt thật".
-   **Kiến trúc:**
    -   **VAE (Variational Autoencoder):** Mã hóa ảnh sang không gian tiềm ẩn (latent space) và giải mã ngược lại.
    -   **UNet:** Kiến trúc chính của mô hình Diffusion, thực hiện quá trình khử nhiễu.
    -   **Identity Encoder (ArcFace):** Trích xuất vector đặc trưng nhận dạng để đảm bảo ảnh tái tạo vẫn giữ lại danh tính của người trong ảnh gốc.
-   **Quá trình hoạt động:**
    1.  Ảnh gốc được mã hóa thành vector tiềm ẩn `z`.
    2.  Nhiễu được thêm vào `z` để tạo ra `z_t`.
    3.  UNet, với điều kiện là vector nhận dạng từ ArcFace, sẽ khử nhiễu `z_t` qua nhiều bước để tái tạo lại vector tiềm ẩn "sạch" `z'`.
    4.  VAE giải mã `z'` để tạo ra ảnh tái tạo.
    5.  **Tín hiệu Bất thường** được tính bằng `abs(Ảnh Gốc - Ảnh Tái Tạo)`.

### 4.3. Giai đoạn 2: OA-Net (Off-real Attention Network) - Mô Hình Phân Tích Chuyên Sâu

-   **Mục tiêu:** Đây là "bộ não" chính, được thiết kế để phân tích các tín hiệu bất thường do DFG cung cấp.
-   **Kiến trúc chi tiết:**

| Thành Phần | Kiến Trúc Sử Dụng | Mục Đích & Hoạt Động |
| :--- | :--- | :--- |
| **Đầu vào** | Tín hiệu Bất thường | Dữ liệu đầu vào cho cả hai luồng xử lý của OA-Net. |
| **Encoder Phụ** | **ResNet-18** | Trích xuất các đặc trưng không gian (spatial features) cục bộ, chi tiết từ tín hiệu bất thường. Đầu ra được chiếu (project) sang không gian 768 chiều để tương thích với ViT. |
| **Backbone Chính** | **Vision Transformer (ViT-Base)** | Xử lý tín hiệu bất thường để nắm bắt các mối quan hệ và ngữ cảnh toàn cục (global context) thông qua cơ chế self-attention. |
| **Cơ chế Kết hợp** | **12 Lớp Cross-Attention** | Tại mỗi lớp trong số 12 lớp của ViT, đặc trưng từ ViT (đóng vai trò `query`) sẽ "tham khảo" thông tin từ đặc trưng của ResNet (đóng vai trò `key` và `value`). Quá trình này giúp làm giàu thông tin cho luồng ViT. |
| **Bộ Phân Loại** | **Linear Layer** | Đưa ra quyết định cuối cùng (Real/Spoof) dựa trên đặc trưng từ token `[CLS]` của ViT sau khi đã được tăng cường thông tin. |

---

## 5. Kết Quả Thực Nghiệm và Đánh Giá

### 5.1. Quá trình Huấn luyện
Mô hình được huấn luyện trên hệ thống 2 GPU với các kỹ thuật tối ưu hóa `nn.DataParallel` và `torch.amp`. Cơ chế **Dừng Sớm (Early Stopping)** đã hoạt động hiệu quả, kết thúc quá trình huấn luyện sau 15 epochs. Dựa trên loss của tập kiểm định, mô hình tốt nhất đạt được ở **Epoch thứ 4**.

![Đường cong Huấn luyện và Kiểm định](results/charts/1_training_curves.png)

### 5.2. Đánh Giá Hiệu Suất Cuối Cùng

Mô hình tốt nhất được đánh giá trên tập kiểm tra (test set) với 742 mẫu chưa từng thấy.

#### **Ma Trận Nhầm Lẫn (Confusion Matrix)**

![Heatmap Ma trận Nhầm lẫn](results/charts/6_confusion_matrix_heatmap.png)

#### **Bảng Chỉ Số Hiệu Suất**

| Chỉ Số | Giá Trị | Ý nghĩa & Tầm quan trọng |
| :--- | :--- | :--- |
| **Accuracy** | **84.10%** | Tỷ lệ dự đoán đúng trên tổng số mẫu. |
| **APCER** | **6.85%** | **(Lỗi An ninh)** Tỷ lệ tấn công giả mạo bị bỏ lọt (nhận nhầm là thật). **Càng thấp càng tốt.** |
| **BPCER** | **24.67%** | **(Lỗi Trải nghiệm)** Tỷ lệ người dùng thật bị từ chối (nhận nhầm là giả). **Càng thấp càng tốt.** |
| **ACER** | **15.76%** | **(Lỗi Trung bình)** Thước đo cân bằng giữa APCER và BPCER, thể hiện hiệu suất tổng thể. |

**Phân tích:** Kết quả cho thấy mô hình có khả năng phát hiện tấn công giả mạo tốt (APCER thấp). Tuy nhiên, điểm yếu nằm ở tỷ lệ nhận nhầm người dùng thật (BPCER cao), chủ yếu do mô hình DFG quá nhạy cảm với các ảnh thật có chất lượng thấp hoặc góc chụp lạ.

---

## 6. Hướng Dẫn Cài Đặt và Sử Dụng

### 6.1. Yêu Cầu
- Python 3.8+
- PyTorch, Transformers, Diffusers
- OpenCV, Scikit-learn, Pandas, Matplotlib, Seaborn

### 6.2. Cài Đặt
```bash
# Clone kho mã nguồn
git clone https://github.com/vanujiash9/FAS_DFG-OANET_Project.git
cd FAS_DFG-OANET_Project

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
6.3. Hướng Dẫn Chạy
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
7. Kết Luận và Hướng Phát Triển Tương Lai
Dự án đã triển khai thành công một hệ thống Chống Giả mạo Khuôn mặt phức tạp, sử dụng kiến trúc ViT và Cross-Attention hiện đại.
Các hướng cải thiện trong tương lai:
Cải thiện BPCER: Huấn luyện DFG trên một tập dữ liệu thật đa dạng hơn để giảm tỷ lệ nhận diện nhầm người dùng thật.
Đánh giá Chéo Miền (Cross-Domain): Kiểm tra mô hình trên các bộ dữ liệu hoàn toàn khác (ví dụ: OULU-NPU) để đánh giá khả năng tổng quát hóa.
Tối ưu hóa cho Triển khai Thực tế: Áp dụng kỹ thuật Chưng cất Mô hình (Model Distillation) để tạo ra một mô hình "học trò" gọn nhẹ, có khả năng triển khai thời gian thực.
