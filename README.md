<img width="2372" height="1764" alt="image" src="https://github.com/user-attachments/assets/cbb8d910-e386-46b9-81c6-61177f4ad9da" />
# Hệ Thống Chống Giả Mạo Khuôn Mặt Sử Dụng Mô Hình Sinh Mẫu (DFG & OA-Net)

## Giới Thiệu Dự Án

Đây là kho mã nguồn cho dự án xây dựng một hệ thống Chống Giả mạo Khuôn mặt (Face Anti-Spoofing - FAS) tiên tiến. Mục tiêu chính của dự án là xây dựng một mô hình mạnh mẽ, có khả năng phân biệt giữa các hình ảnh khuôn mặt thật (live) và các hình thức tấn công giả mạo đa dạng (ví dụ: ảnh in, phát lại trên màn hình, mặt nạ).

Dự án này triển khai một kiến trúc hai giai đoạn phức tạp, lấy cảm hứng từ phương pháp luận AG-FAS, khai thác sức mạnh của các mô hình sinh mẫu (Generative Models) và Vision Transformers với cơ chế Cross-Attention.

**Tác giả:** Bùi Thị Thanh Vân
**Trường:** Đại học Giao thông Vận tải Thành phố Hồ Chí Minh
**Email:** thanh.van19062004@gmail.com
**Thời gian thực hiện:** Tháng 09, 2025

---

## Phương Pháp Luận Cốt Lõi

Hệ thống được xây dựng dựa trên một pipeline hai giai đoạn thông minh:

### Giai đoạn 1: DFG (De-fake Face Generator) - Cỗ Máy Tạo Bằng Chứng

Giai đoạn đầu tiên sử dụng một Mô hình Khuếch tán (Diffusion Model - DFG) được huấn luyện **chỉ trên dữ liệu khuôn mặt thật**. Nhiệm vụ của nó không phải là phân loại, mà là học sâu về sự phân phối của một "khuôn mặt thật" trông như thế nào. Khi nhận một ảnh đầu vào, DFG sẽ cố gắng tái tạo lại một phiên bản "lý tưởng" và "thật" của ảnh đó.

Đầu ra quan trọng nhất của giai đoạn này là **"Tín hiệu Bất thường" (Anomalous Cue)**, được tính bằng hiệu số tuyệt đối giữa ảnh gốc và ảnh được tái tạo.

-   **Với ảnh Thật (Live):** Quá trình tái tạo gần như hoàn hảo, tạo ra một cue gần như **tối đen hoàn toàn**.
-   **Với ảnh Giả mạo (Spoof):** DFG sẽ "sửa chữa" các dấu hiệu giả mạo (ví dụ: nhiễu moiré của màn hình, kết cấu giấy in), dẫn đến sự khác biệt lớn và tạo ra một cue **sáng rực, làm nổi bật các vùng giả mạo**.

### Giai đoạn 2: OA-Net (Off-real Attention Network) - Thám Tử Thông Minh

Giai đoạn thứ hai là một mô hình phân loại mạnh mẽ, học hỏi **từ các tín hiệu bất thường**, chứ không phải từ ảnh gốc. Dự án này triển khai một kiến trúc mới dựa trên Vision Transformer (ViT) cho OA-Net, được tăng cường bởi:

-   **ResNet Cue Encoder:** Một backbone CNN để trích xuất các đặc trưng không gian (spatial features) phong phú từ các tín hiệu bất thường.
-   **Vision Transformer Backbone:** Một mô hình ViT xử lý các tín hiệu bất thường để nắm bắt các đặc trưng và mối quan hệ toàn cục.
-   **Cơ chế Cross-Attention:** Tại mỗi lớp của ViT, một module cross-attention cho phép mô hình kết hợp một cách thông minh thông tin từ luồng chính của ViT với các đặc trưng chi tiết từ ResNet encoder. Điều này giúp mô hình tập trung vào những dấu vết giả mạo quan trọng nhất.

Cách tiếp cận này huấn luyện mô hình trở thành một chuyên gia nhận dạng "bằng chứng" của một cuộc tấn công, thay vì chỉ đơn thuần ghi nhớ các khuôn mặt.

---

## Phân Tích và Trực Quan Hóa Dữ Liệu

Dự án sử dụng kết hợp bộ dữ liệu FFHQ và CelebA-Spoof. Dưới đây là phân tích chi tiết về dữ liệu ở các giai đoạn khác nhau của pipeline.

### 1. Nguồn Dữ Liệu Gốc

Tập dữ liệu ban đầu bao gồm ba nguồn chính. Mô hình được huấn luyện trên một tập hợp đa dạng các khuôn mặt thật để đảm bảo sự hiểu biết sâu sắc về "tính sống".

<img width="2972" height="2064" alt="image" src="https://github.com/user-attachments/assets/228d82c8-b229-4261-a828-da18baa1869c" />


Tóm tắt cấp cao về dữ liệu thô trước khi xử lý:

<img width="2372" height="1764" alt="image" src="https://github.com/user-attachments/assets/0ffd9fc2-6ac9-42fe-9528-971b391ab1ad" />


### 2. Sự Đa Dạng của Các Loại Tấn Công Giả Mạo

Bộ dữ liệu bao gồm nhiều loại hình tấn công giả mạo khác nhau, đảm bảo mô hình được huấn luyện để chống lại nhiều mối đe dọa.

<img width="2560" height="2042" alt="image" src="https://github.com/user-attachments/assets/08d20407-943b-4bc7-99dd-33ed56e2fc46" />


### 3. Dữ Liệu Đã Xử Lý để Huấn Luyện

Sau giai đoạn DFG, các ảnh thô được chuyển đổi thành tín hiệu bất thường. Dữ liệu sau đó được cân bằng để ngăn chặn sự thiên vị của mô hình.

<img width="2371" height="1764" alt="image" src="https://github.com/user-attachments/assets/18b06b1b-d034-42b7-8f21-10675240a0ad" />


### 4. Phân Chia Dữ Liệu Cuối Cùng

Tập dữ liệu tín hiệu bất thường đã được cân bằng được chia thành các tập huấn luyện, kiểm định và kiểm tra theo tỷ lệ 80/10/10, đảm bảo rằng các chủ thể (subject) trong tập kiểm tra hoàn toàn không xuất hiện trong quá trình huấn luyện.

<img width="2369" height="2374" alt="image" src="https://github.com/user-attachments/assets/9c8b579d-7177-44cc-85dd-aa35604ed4f2" />


---

## Hiệu Suất Mô Hình và Đánh Giá

Mô hình OA-Net đã được huấn luyện được đánh giá trên tập kiểm tra (test set) chưa từng thấy. Kết quả cho thấy khả năng phân loại tốt giữa các trường hợp thật và giả mạo.

### 1. Đường Cong Huấn Luyện (Training Curves)

Mô hình được huấn luyện trong 15 epochs, với cơ chế Dừng Sớm (Early Stopping) được kích hoạt khi loss trên tập kiểm định bắt đầu tăng, cho thấy điểm tổng quát hóa tối ưu. Mô hình tốt nhất đã đạt được tại **Epoch 1**.

![Đường cong Huấn luyện và Kiểm định](results/charts/1_training_curves.png)

### 2. Các Chỉ Số Hiệu Suất trên Tập Test

Kết quả đánh giá cuối cùng mang lại các chỉ số hiệu suất sau, là tiêu chuẩn cho lĩnh vực Face Anti-Spoofing:

| Chỉ Số                                                  | Giá Trị    | Mô Tả                                                                           |
| ------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------- |
| **Accuracy (Độ chính xác)**                              | **84.10%** | Tỷ lệ phần trăm tổng thể các dự đoán chính xác.                                  |
| **APCER** (Tỷ lệ Lỗi Phân loại Tấn công Giả mạo)        | **6.85%**  | Tỷ lệ các cuộc tấn công giả mạo bị phân loại nhầm là thật (Sai lầm loại II). Đây là một chỉ số an ninh quan trọng. |
| **BPCER** (Tỷ lệ Lỗi Phân loại Trường hợp Thật)          | **24.67%** | Tỷ lệ các trường hợp thật bị phân loại nhầm là giả mạo (Sai lầm loại I). Đây là một chỉ số quan trọng về trải nghiệm người dùng. |
| **ACER** (Tỷ lệ Lỗi Phân loại Trung bình)                  | **15.76%** | Trung bình của APCER và BPCER, cung cấp một thước đo cân bằng về hiệu suất của mô hình. |

### 3. Ma Trận Nhầm Lẫn (Confusion Matrix)

Ma trận nhầm lẫn cung cấp một cái nhìn chi tiết về các dự đoán của mô hình trên 742 mẫu trong tập kiểm tra.

<img width="1925" height="1644" alt="image" src="https://github.com/user-attachments/assets/8a0a15f0-331a-47ff-b983-62ec18a5e3de" />


**Phân tích:**
-   **True Negatives (Thật -> Thật):** 301
-   **False Positives (Thật -> Giả):** 76
-   **False Negatives (Giả -> Thật):** 40
-   **True Positives (Giả -> Giả):** 325

Kết quả cho thấy hiệu suất phát hiện tấn công giả mạo tốt (True Positives cao) nhưng cũng chỉ ra tiềm năng cải thiện trong việc giảm số lượng báo động giả đối với người dùng thật (False Positives).

---

## Yêu Cầu Hệ Thống và Cài Đặt

### Yêu Cầu
- Python 3.8+
- PyTorch 1.10+
- Transformers, Diffusers
- OpenCV, Scikit-learn, Pandas, Matplotlib, Seaborn

### Cài Đặt
```bash
# Clone kho mã nguồn
git clone https://github.com/vanujiash9/FAS_DFG-OANET_Project.git
cd FAS_DFG-OANET_Project

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
