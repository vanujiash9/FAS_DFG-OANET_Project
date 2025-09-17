# Hệ**   | Bùi Thị Thanh Vân                              |
| **Trường**    | Đại học Giao thông V Thống Chống Giả Mạo Khuôn Mặt Sử Dụng Mô Hình Khuếch Tán và Vision Transformer

## 1. Giới Thiệu Dự Án

Đây là kho mã nguồn cho dự án nghiên cứu và triển khaiận tải Thành phố Hồ Chí Minh |
| **Email**     | `thanh.van19062 một hệ thống **Chống Giả mạo Khuôn mặt (Face Anti-Spoofing - FAS)** tiên004@gmail.com`                  |
| **Thời gian** | Tháng 09, 2025                                 |

---

## 2. Môi Trường và Hạ Tầng Thực Nghiệm

Do yêu cầu tính tiến. Mục tiêu chính của dự án là xây dựng một mô hình mạnh mẽ, có khả năng phân biệt giữa các hình toán cực lớn của các mô hình Diffusion và Transformer, toàn bộ quá trình thực nghiệm của dự án được thực hiện trên một hệ ảnh khuôn mặt thật (live) và các hình thức tấn công giả mạo đa dạng, bao gồm ảnh in, phát lại trên màn hình, và mặt nạ 3D.

Dự án này triển khai một kiến trúc hai thống máy chủ chuyên dụng thuê ngoài với cấu hình chi tiết như sau:

| Thành Phần | Chi Tiết K giai đoạn phức tạp, lấy cảm hứng từ phương pháp luận **AG-FAS**, nhưng được cải tiến bằng cách sửỹ Thuật |
| :--- | :--- |
| **GPU** | 2 x NVIDIA GeForce RTX dụng **Vision Transformer (ViT)** làm backbone chính, kết hợp với cơ chế **Cross-Attention** để tăng cường khả 4090 (24GB VRAM mỗi card) |
| **Hệ điều hành** | Linux (Ubuntu/Debian-based) |
| **CUDA Version** | 12.x |
| ** năng phân tích và nhận diện.

| Thông Tin      | Chi Tiết                                       |
| -------------- | ---------------------------------------------- |
| **Tác giả**    | Bùi Thị Thanh Vân                              |
| **Trường**      Ngôn ngữ** | Python 3.10 |
| **Frameworks** | PyTorch, Hugging Face Transformers, Diffusers |

---

## 3. Phân Tích Dữ Liệu và Quy Trình Tiền Xử Lý| Đại học Giao thông Vận tải Thành phố Hồ Chí Minh |
| **Email**      | `thanh.van

### 3.1. Nguồn Dữ Liệu Gốc

Dự án sử dụng một bộ dữ liệu lớn19062004@gmail.com`                  |
| **Thời gian** | Tháng 09, 2025                                 |

---

## 2. Môi Trường Thực Nghiệm

 và đa dạng, được tổng hợp từ hai nguồn chính:
-   **FFHQ (Flickr-Faces-HQDo yêu cầu tính toán cực kỳ cao của các mô hình Diffusion và Transformer, toàn bộ quá trình thực nghiệm, từ):** Cung cấp một lượng lớn ảnh khuôn mặt thật chất lượng cao.
-   **CelebA- huấn luyện đến xử lý dữ liệu, được thực hiện trên một hệ thống máy chủ chuyên dụng thuê ngoài.

| ThànhSpoof:** Một bộ dữ liệu tiêu chuẩn trong lĩnh vực FAS, cung cấp cả ảnh thật và 10 loại tấn Phần | Chi Tiết Kỹ Thuật |
| :--- | :--- |
| **GPU** |  công giả mạo khác nhau.

| Nguồn Dữ Liệu | Loại | Số Lượng Ảnh Gốc |
|2 x NVIDIA GeForce RTX 4090 (24GB VRAM mỗi card) |
| **CPU** | :--- | :--- | :--- |
| FFHQ | Live | 2,562 |
| Cele (Thông tin CPU của máy chủ) |
| **RAM** | (Thông tin RAM của máy chủ) |
| **Hệ điều hành** | Ubuntu 22.04 LTS (Linux) |
| **Nền tảng** | Python 3.10, PyTorch 1.12+ (CUDA 11bA | Live | 35,000 |
| CelebA | Spoof | 3.6+) |
| **Thư viện chính** | `transformers`, `diffusers`, `opencv-python`, `sc3,433 |
| **Tổng cộng** | | **70,995** |

### ikit-learn`, `pandas` |

---

## 3. Dữ Liệu và Quy Trình Ti3.2. Quy Trình Tiền Xử Lý
Tất cả ảnh đầu vào đều trải qua một pipeline tiền xử lý chuẩn hóa để loại bỏ các biến thiên không mong muốn và đảm bảo tính nhất quán:
1.  ền Xử Lý

### 3.1. Nguồn Dữ Liệu Gốc

Dự án sử dụng kết hợp hai bộ dữ liệu lớn và chất lượng cao:
-   **FFHQ (Flickr-Faces-HQ):**Phát hiện và Căn chỉnh:** Sử dụng mô hình MTCNN để phát hiện khuôn mặt và các điểm mốc (landmarks).
2.  **Cắt (Cropping):** Cắt ảnh để chỉ giữ lại vùng mặt, loại bỏ phần lớn nền có thể gây nhiễu.
3.  **Thay đổi kích thước (** Cung cấp một lượng lớn ảnh khuôn mặt thật, đa dạng, chất lượng cao, chủ yếu dùng để huấnResizing):** Resize tất cả các ảnh về kích thước `224x224` pixels để phù luyện mô hình DFG.
-   **CelebA-Spoof:** Một bộ dữ liệu tiêu chuẩn cho bài hợp với đầu vào của mô hình ViT.
4.  **Chuẩn hóa (Normalization):** Chuyển giá toán FAS, cung cấp cả ảnh thật và 10 loại tấn công giả mạo khác nhau, được gán nh trị pixel về khoảng `[-1, 1]`.

### 3.3. Phân Chia Dữ Liệuãn chi tiết.

### 3.2. Tiền Xử Lý

Tất cả ảnh đầu vào (c
Để đảm bảo tính khách quan của kết quả đánh giá, dữ liệu được chia theo phương pháp **phân chia theo định danh (subject-disjoint)**. Tức là, một người (subject) đã xuất hiện trong tập huấn luyện sẽả live và spoof) đều trải qua một pipeline tiền xử lý đồng bộ để loại bỏ các biến thiên không mong muốn và chuẩn hóa dữ liệu:
1.  **Phát hiện và Căn chỉnh:** Sử dụng mô hình MTCNN để không bao giờ xuất hiện trong tập kiểm định hoặc kiểm tra. Tỷ lệ phân chia cuối cùng là **80% Training, 10% Validation, và 10% Test**.

---

## 4. Phương Pháp tự động phát hiện khuôn mặt và các điểm mốc (landmarks).
2.  **Cắt (Cropping):** C Luận Chi Tiết Về Mô Hình

### 4.1. Tổng quan kiến trúc
Hệ thống baoắt ảnh để chỉ giữ lại vùng mặt, loại bỏ phần lớn nền có thể gây nhiễu.
3.  **Thay gồm hai mô hình hoạt động tuần tự. DFG đóng vai trò là một "bộ tiền xử lý thông minh", biến đổi Kích thước (Resize):** Chuẩn hóa tất cả các khuôn mặt về cùng một kích thước (ví dụ: 224x224) để phù hợp với đầu vào của mô hình.
4.  **Chuẩn đổi ảnh đầu vào thành một dạng "bằng chứng" dễ phân tích hơn. OA-Net sau đó sẽ phân tích bằng hóa (Normalization):** Chuyển giá trị pixel về một khoảng nhất định, là yêu cầu tiêu chuẩn cho các mạng chứng này để đưa ra kết luận cuối cùng.

`🖼️ Ảnh Gốc ➡️ 🧠 DFG  neural.

### 3.3. Phân Chia Dữ Liệu

Để đảm bảo tính khách quan của kết quả➡️ 🔬 Tín hiệu Bất thường (Anomalous Cue) ➡️ 🕵️‍♀️ OA-Net , dữ liệu được phân chia theo nguyên tắc **tách biệt định danh (subject-disjoint)**. Các chủ➡️ ✅/❌ Kết quả`

### 4.2. Giai đoạn 1: DFG (De thể (người) xuất hiện trong tập huấn luyện sẽ không xuất hiện trong tập kiểm định hoặc tập kiểm tra. Điều này ngăn-fake Face Generator) - Mô hình Sinh Tín hiệu Bất thường

-   **Mục tiêu:** D chặn hiện tượng "học vẹt" và đánh giá đúng khả năng tổng quát hóa của mô hình trên những người hoànFG không học về ảnh giả mạo. Nó được huấn luyện chỉ trên dữ liệu thật để học một cách sâu sắc về sự toàn mới.

---

## 4. Kiến Trúc Mô Hình và Phương Pháp Luận Chi Tiết

H phân phối của một "khuôn mặt thật".
-   **Kiến trúc:**
    -   **ệ thống được xây dựng dựa trên một pipeline hai giai đoạn.

**Luồng hoạt động tổng thể:**
VAE (Variational Autoencoder):** Mã hóa ảnh sang không gian tiềm ẩn (latent space) và giải mã ngược`Ảnh Gốc ➡️ DFG ➡️ Tín hiệu Bất thường (Anomalous Cue) ➡️ OA lại.
    -   **UNet:** Kiến trúc chính của mô hình Diffusion, thực hiện quá trình khử nhiễu.-Net ➡️ Kết quả`

### 4.1. Giai đoạn 1: DFG (De
    -   **Identity Encoder (ArcFace):** Trích xuất vector đặc trưng nhận dạng để đảm bảo ảnh-fake Face Generator) - Cỗ Máy Tạo Bằng Chứng

Giai đoạn đầu tiên sử dụng một **M tái tạo vẫn giữ lại danh tính của người trong ảnh gốc.
-   **Quá trình hoạt động:**
    1.ô hình Khuếch tán (Diffusion Model - DFG)**.
-   **Mục tiêu:** DFG được  Ảnh gốc được mã hóa thành vector tiềm ẩn `z`.
    2.  Nhiễu được thêm vào `z` để tạo ra `z_t`.
    3.  UNet, với điều kiện là vector nhận dạng từ huấn luyện **chỉ trên dữ liệu khuôn mặt thật**. Nhiệm vụ của nó không phải là phân loại, mà là học ArcFace, sẽ khử nhiễu `z_t` qua nhiều bước để tái tạo lại vector tiềm ẩn " một cách sâu sắc về sự phân phối của một "khuôn mặt thật".
-   **Thành phần:**sạch" `z'`.
    4.  VAE giải mã `z'` để tạo ra ảnh tái tạo.
     DFG bao gồm một UNet, một VAE, và một Identity Encoder (ArcFace) để bảo toàn đặc tính nhận dạng trong quá trình tái tạo.
-   **Cơ chế hoạt động:** Khi nhận một ảnh đầu5.  **Tín hiệu Bất thường** được tính bằng `abs(Ảnh Gốc - Ảnh Tái Tạo)`.

### 4.3. Giai đoạn 2: OA-Net (Off-real Attention Network) - Mô vào, DFG sẽ cố gắng tái tạo lại một phiên bản "lý tưởng" và "thật" của ảnh đó. **"Tín hiệu Bất thường" (Anomalous Cue)** được tính bằng hiệu số tuyệt đối Hình Phân Tích Chuyên Sâu

-   **Mục tiêu:** Đây là "bộ não" giữa ảnh gốc và ảnh được tái tạo.

**Ví dụ Trực quan về Hoạt động của DFG:**

 chính, được thiết kế để phân tích các tín hiệu bất thường do DFG cung cấp.
-   **Kiến trúc chiBảng dưới đây minh họa cách DFG xử lý hai trường hợp thực tế: một ảnh thật ở tư thế khó (out-of-distribution) và một ảnh giả mạo.

| Loại | Ảnh Gốc | Ảnh Tái Tạo tiết:**

| Thành Phần | Kiến Trúc Sử Dụng | Mục Đích & Hoạt Động |
| :--- | :--- | :--- |
| **Đầu vào** | Tín hiệu Bất thường | Dữ liệu đầu vào cho cả hai luồng xử lý của OA-Net. |
| **Encoder Phụ** | **Res bởi DFG | Tín hiệu Bất thường (Anomalous Cue) | Phân Tích |
|Net-18** | Trích xuất các đặc trưng không gian (spatial features) cục bộ, chi tiết từ tín hiệu bất thường. Đầu ra được chiếu (project) sang không gian 768 chiều để tương thích với ViT :--- | :---: | :---: | :---: | :--- |
| **THẬT (Live. |
| **Backbone Chính** | **Vision Transformer (ViT-Base)** | Xử lý tín)**<br>*(Ca khó)* | ![Live Sample Original](results/readme_assets/live_sample_difficult hiệu bất thường để nắm bắt các mối quan hệ và ngữ cảnh toàn cục (global context) thông qua cơ chế self_case.png) | Mô hình thất bại trong việc tái tạo ảnh có tư thế lạ, cố gắng "ép-attention. |
| **Cơ chế Kết hợp** | **12 Lớp Cross-Attention** |" nó về một dạng chuẩn. | Kết quả là một cue rất sáng, có nguy cơ bị OA-Net nhận nhầm là giả mạo. | *Điều này cho thấy giới hạn của DFG với dữ liệu ngoài phân phối và là Tại mỗi lớp trong số 12 lớp của ViT, đặc trưng từ ViT (đóng vai trò `query`) sẽ nguyên nhân tiềm tàng gây ra lỗi **BPCER** (nhận nhầm người thật).* |
| **GI "tham khảo" thông tin từ đặc trưng của ResNet (đóng vai trò `key` và `value`). QuẢ MẠO (Spoof)**<br>*(Ca thành công)* | ![Spoof Sample Original](results/readme_assets/spoof_sample_detection_case.png) | DFG "sửa chữa" ảnh giả mạo, tạoá trình này giúp làm giàu thông tin cho luồng ViT. |
| **Bộ Phân Loại** | **Linear ra một phiên bản khuôn mặt trông "thật" hơn. | Sự khác biệt lớn tạo ra một cue rất sáng, Layer** | Đưa ra quyết định cuối cùng (Real/Spoof) dựa trên đặc trưng từ token `[ làm nổi bật các vùng bất thường. | *Đây là "bằng chứng" rõ ràng để OA-Net họcCLS]` của ViT sau khi đã được tăng cường thông tin. |

---

## 5. Kết Qu và phát hiện tấn công, giúp giảm lỗi **APCER** (bỏ lọt tấn công giả mạo).* |

###ả Thực Nghiệm và Đánh Giá

### 5.1. Quá trình Huấn luyện
Mô hình được huấn luyện trên hệ thống 2 GPU với các kỹ thuật tối ưu hóa `nn.DataParallel` và `torch.amp 4.2. Giai đoạn 2: OA-Net (Off-real Attention Network) - Mô Hình Ph`. Cơ chế **Dừng Sớm (Early Stopping)** đã hoạt động hiệu quả, kết thúc quá trình huấn luyện sau ân Tích Chuyên Sâu

Đây là "bộ não" của hệ thống, được thiết kế để phân tích các tín hiệu bất thường do DFG cung cấp.

| Thành Phần | Kiến Trúc Sử Dụng | Mục15 epochs. Dựa trên loss của tập kiểm định, mô hình tốt nhất đạt được ở **Epoch thứ 4**.

 Đích |
| :--- | :--- | :--- |
| **Đầu vào** | Tín hiệu Bất thường | Dữ liệu đầu vào duy nhất cho OA-Net, chứa đựng các "bằng chứng" giả mạo. |
| **Encoder Phụ** | **ResNet-18** | Trích xuất các![Đường cong Huấn luyện và Kiểm định](results/charts/1_training_curves.png)

### 5.2. Đánh Giá Hiệu Suất Cuối Cùng

Mô hình tốt nhất được đánh giá trên tập đặc trưng không gian (spatial features) cục bộ, chi tiết từ tín hiệu bất thường. |
| **Backbone Chính** | **Vision Transformer (ViT-Base)** | Xử lý tín hiệu bất thường để nắm bắt kiểm tra (test set) với 742 mẫu chưa từng thấy.

#### **Ma Trận Nhầm L các mối quan hệ và ngữ cảnh toàn cục (global context). |
| **Cơ chế Kết hợp** | **12 Lớp Cross-Attention** | Tại mỗi lớp của ViT, một module `OffRealAttention` cho phépẫn (Confusion Matrix)**

![Heatmap Ma trận Nhầm lẫn](results/charts/6_confusion_matrix_ luồng ViT "tham khảo" thông tin từ luồng ResNet, giúp mô hình tập trung vào các vùng "heatmap.png)

#### **Bảng Chỉ Số Hiệu Suất**

| Chỉ Số | Giá Trị | Ý nghĩa & Tầm quan trọng |
| :--- | :--- | :--- |
| **Accuracy** | **bằng chứng" quan trọng nhất. |
| **Bộ Phân Loại** | **Linear Layer** | Đ84.10%** | Tỷ lệ dự đoán đúng trên tổng số mẫu. |
| **APCER** | **6.85%** | **(Lỗi An ninh)** Tỷ lệ tấn công giả mạoưa ra quyết định cuối cùng (Real/Spoof) dựa trên đặc trưng từ token `[CLS]` của ViT. |

---

## 5. Quy Trình Huấn Luyện và Tối Ưu Hóa bị bỏ lọt (nhận nhầm là thật). **Càng thấp càng tốt.** |
| **BPCER**

-   **Pipeline:** Đầu tiên, mô hình DFG được huấn luyện trên dữ liệu thật. Sau đó, D | **24.67%** | **(Lỗi Trải nghiệm)** Tỷ lệ người dùng thật bị từFG được dùng để sinh ra toàn bộ tập dữ liệu cues. Cuối cùng, OA-Net được huấn luyện trên tập chối (nhận nhầm là giả). **Càng thấp càng tốt.** |
| **ACER** | ** dữ liệu cues này.
-   **Tối ưu hóa Huấn luyện:** Do mô hình rất lớn, hai kỹ15.76%** | **(Lỗi Trung bình)** Thước đo cân bằng giữa APCER và BPC thuật tối ưu hóa hiệu suất đã được áp dụng:
    1.  **Song song Dữ liệu (`nn.DataParallelER, thể hiện hiệu suất tổng thể. |

**Phân tích:** Kết quả cho thấy mô hình có khả năng phát`):** Tận dụng sức mạnh của cả hai GPU để chia sẻ tải công việc, giúp giảm thời gian huấn hiện tấn công giả mạo tốt (APCER thấp). Tuy nhiên, điểm yếu nằm ở tỷ lệ nhận nhầm luyện mỗi epoch.
    2.  **Độ chính xác Hỗn hợp (`torch.cuda.amp người dùng thật (BPCER cao), chủ yếu do mô hình DFG quá nhạy cảm với các ảnh thật có`):** Sử dụng các phép toán 16-bit (FP16) để tăng tốc độ xử lý và giảm đáng kể lượng VRAM sử dụng.
-   **Cơ chế điều khiển:** Quá trình huấn luyện sử dụng scheduler chất lượng thấp hoặc góc chụp lạ.

---

## 6. Hướng Dẫn Cài Đặt và Sử Dụng

### 6.1. Yêu Cầu
- Python 3.8+
- PyTorch, Transformers `ReduceLROnPlateau` để tự động giảm learning rate và cơ chế `Early Stopping` để kết thúc sớm khi mô hình ngừng cải thiện, tránh lãng phí tài nguyên và overfitting.

---

## 6. Kết, Diffusers
- OpenCV, Scikit-learn, Pandas, Matplotlib, Seaborn

### 6.2. Cài Đặt
```bash
# Clone kho mã nguồn
git clone https://github.com/vanuji Quả Thực Nghiệm

### 6.1. Đường Cong Huấn Luyện (Training Curves)

Mash9/FAS_DFG-OANET_Project.git
cd FAS_DFG-OANô hình được huấn luyện trong 15 epochs. Dựa trên loss của tập kiểm định (validation), mô hình tốt nhất đạt được ở **Epoch thứ 4**, trước khi có dấu hiệu overfitting.

![Đường cong Huấn luyện và Kiểm định](results/charts/1_training_curves.png)

### 6.2. Đánh Giá Hiệu Suất Cuối Cùng trên Tập Test

Bảng dưới đây tổng hợp các chỉ số hiệu suất cuối cùng trên tập test.

| Chỉ Số | Giá Trị | Ý nghĩa & Tầm quan trọng |
| :--- | :--- | :--- |
| **Accuracy** | **84.10%** | Tỷ lệ dự đoán đúng trên tổng số mẫu. |
| **APCER** | **6.85%** | **(Lỗi An ninh)** Tỷ lệ tấn công giả mạo bị bỏ lọt (nhận nhầm là thật). **Càng thấp càng tốt.** |
| **BPCER** | **24.67%** |ET_Project

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
Dự án đã triển khai thành công một hệ thống Chống Giả mạo Khu (Lỗi Trải nghiệm) Tỷ lệ người dùng thật bị từ chối (nhận nhầm là giả). Càng thấp càng tốt. |
| ACER | 15.76% | (Lỗi Trung bình) Thước đo cân bằng giữa APCER và BPCER, thể hiện hiệu suất tổng thể. |
![alt text](results/charts/6_confusion_matrix_heatmap.png)
Phân tích: Kết quả cho thấy mô hình có khả năng phát hiện tấn công giả mạo tốt (APôn mặt phức tạp, sử dụng kiến trúc ViT và Cross-Attention hiện đại.
Các hướng cải thiện trong tương lai:
Cải thiện BPCER: Huấn luyện DFG trên một tập dữ liệu thật đa dạng hơn để giảm tỷ lệ nhận diện nhầm người dùng thật.
Đánh giá Chéo Miền (Cross-Domain): Kiểm tra mô hình trên các bộ dữ liệu hoàn toàn khác (ví dụ: OULU-NPU) để đánh giá khả năng tổng quát hóa.
Tối ưu hóa choCER thấp). Tuy nhiên, điểm yếu nằm ở tỷ lệ nhận nhầm người dùng thật (BPCER cao), chủ Triển khai Thực tế: Áp dụng kỹ thuật Chưng cất Mô hình (Model Distillation) để tạo ra một mô hình "học trò" gọn nhẹ, có khả năng triển khai thời gian thực.
