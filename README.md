
# Face Anti-Spoofing with Generative Models (DFG & ViT-based OA-Net)

Đây là kho mã nguồn cho dự án nghiên cứu và triển khai một hệ thống Chống Giả mạo Khuôn mặt (Face Anti-Spoofing - FAS) tiên tiến, dựa trên kiến trúc hai giai đoạn: **De-fake Face Generator (DFG)** và **Off-real Attention Network (OA-Net)**.

Dự án này khám phá việc sử dụng các mô hình sinh mẫu để tạo ra "tín hiệu bất thường" (Anomaly Cue) từ ảnh đầu vào. Các tín hiệu này sau đó được phân tích bởi một kiến trúc lai (hybrid) kết hợp Vision Transformer và Mạng Tích chập (CNN) để phân loại thật/giả.

| Thông Tin      | Chi Tiết                                       |
| -------------- | ---------------------------------------------- |
| **Tác giả**    | Bùi Thị Thanh Vân                              |
| **Trường**      | Đại học Giao thông Vận tải Thành phố Hồ Chí Minh |
| **Ngành**      | Khoa học dữ liệu                               |
| **Email**      | `thanh.van19062004@gmail.com`                  |

---

## 🏛️ Kiến Trúc và Phương Pháp Luận

Hệ thống được xây dựng theo triết lý **phát hiện bất thường**. Thay vì học các đặc điểm của ảnh giả, mô hình được dạy để hiểu sâu sắc "một khuôn mặt thật trông như thế nào" và coi bất kỳ sai khác nào là dấu hiệu của sự giả mạo.

**Luồng hoạt động tổng thể:**

1.  **Giai đoạn 1: De-fake Face Generator (DFG)**
    -   Sử dụng một **Latent Diffusion Model (LDM)** được huấn luyện chỉ trên dữ liệu khuôn mặt thật.
    -   Mô hình này nhận một ảnh đầu vào và tái tạo lại một phiên bản "chuẩn thật" của khuôn mặt đó, với sự hỗ trợ của **ArcFace** để bảo toàn danh tính.
    -   **Anomaly Cue** được tạo ra bằng cách lấy hiệu số tuyệt đối giữa ảnh gốc và ảnh tái tạo. Cue này sẽ sáng rực ở những vùng có dấu hiệu giả mạo.

2.  **Giai đoạn 2: Off-real Attention Network (OA-Net)**
    -   Đây là mô hình phân loại chính, chỉ nhận đầu vào là **Anomaly Cue** đã được tạo ra.
    -   Sử dụng kiến trúc lai (hybrid) gồm **ViT-Base** (nắm bắt ngữ cảnh toàn cục) và **ResNet-18** (trích xuất đặc trưng cục bộ).
    -   **12 lớp Cross-Attention** kết hợp thông tin từ hai luồng trên, giúp mô hình tập trung vào các vùng bằng chứng quan trọng nhất trước khi đưa ra quyết định cuối cùng.

---

## 📂 Cấu Trúc Thư Mục Dự Án
FAS_project/
├── checkpoints/ # Nơi lưu các file model đã huấn luyện (.pth)
│ ├── dfg/
│ └── oanet/
├── configs/ # Các file cấu hình .yaml
├── data/
│ ├── raw/ # Dữ liệu gốc (FFHQ, CelebA-Spoof)
│ └── processed/ # Dữ liệu đã qua xử lý
│ └── anomalous_cues/ # Thư mục chứa các file cue (.pt)
├── logs/ # Chứa các file log quá trình huấn luyện
├── results/ # Thư mục chứa tất cả kết quả đầu ra
│ ├── charts/ # Các biểu đồ phân tích
│ ├── predictions/ # Kết quả dự đoán từ script predict.py
│ └── ..._report.txt # Các file báo cáo tổng kết
├── src/ # Toàn bộ mã nguồn của dự án
│ ├── data_management/
│ ├── models/
│ ├── scripts/
│ ├── training_utils/
│ └── visualization/
└── uploads/ # Thư mục để "upload" ảnh/video cần dự đoán
code
Code
---

## 🚀 Hướng Dẫn Cài Đặt và Sử Dụng

### 1. Yêu Cầu
-   Python 3.8+
-   PyTorch & Torchvision
-   Hạ tầng GPU với CUDA (khuyến nghị 2 x 24GB VRAM cho huấn luyện)
-   Các thư viện phụ: `pip install transformers diffusers opencv-python scikit-learn pandas matplotlib seaborn`

### 2. Cài Đặt

1.  **Clone kho mã nguồn:**
    ```bash
    git clone https://github.com/vanujiash9/FAS_DFG-OANET_Project.git
    cd FAS_DFG-OANET_Project
    ```
2.  **(Tùy chọn) Tạo môi trường ảo:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```
3.  **Cài đặt các thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Chuẩn bị mô hình tiền huấn luyện:**
    -   Tải các file trọng số (`.pth`) cần thiết.
    -   Đặt chúng vào các thư mục tương ứng trong `checkpoints/`.

5.  **Chuẩn bị dữ liệu:**
    -   Đặt dữ liệu thô vào `data/raw/`.
    -   Chạy các script tiền xử lý và sinh cues, hoặc đặt các file cues đã được tạo sẵn vào `data/processed/anomalous_cues/`.

### 3. Thực Thi

-   **Huấn luyện lại toàn bộ pipeline:**
    ```bash
    python3 src/scripts/run_full_pipeline.py
    ```
-   **Chỉ huấn luyện OA-Net (khi đã có cues):**
    ```bash
    python3 src/scripts/run_training_only.py
    ```
-   **Tạo báo cáo và trực quan hóa kết quả:**
    ```bash
    python3 src/scripts/generate_full_report.py
    ```
-   **Dự đoán trên ảnh/video mới:**
    1.  Kéo/thả file vào thư mục `uploads/`.
    2.  Chạy script ở chế độ tương tác:
        ```bash
        python3 src/scripts/predict.py
        ```

---

## 📊 Kết Quả Thực Nghiệm

Mô hình OA-Net được huấn luyện trên 7,420 mẫu cues và đánh giá trên 742 mẫu test chưa từng thấy.

### 1. Phân Tích Dữ Liệu

| Phân Bố Nguồn Dữ Liệu Gốc | Phân Bố Các Loại Tấn Công | Phân Chia Dữ Liệu Cuối Cùng |
| :---: | :---: | :---: |
| ![Raw Data Sources Breakdown](results/charts/1_raw_data_sources.png) | ![Distribution of Original Spoof Attack Types](results/charts/4_spoof_type_distribution.png) | ![Final Dataset Split for OA-Net Training](results/charts/5_dataset_split_pie_chart.png) |

### 2. Hiệu Suất Mô Hình

Mô hình đạt hiệu suất tốt nhất ở **Epoch 4** trước khi có dấu hiệu overfitting.

#### Bảng Chỉ Số Đánh Giá
| Chỉ Số | Giá Trị |
| :--- | :---: |
| **Accuracy** | **84.10%** |
| **APCER** (Lỗi An ninh) | **6.85%** |
| **BPCER** (Lỗi Trải nghiệm) | **24.67%** |
| **ACER** | **15.76%** |

#### Ma Trận Nhầm Lẫn
![Confusion Matrix on Test Set](results/charts/6_confusion_matrix_heatmap.png)

**Phân tích:** Mô hình có khả năng phát hiện tấn công tốt (APCER thấp), nhưng điểm yếu lớn nhất là tỷ lệ từ chối nhầm người dùng thật (BPCER cao), nguyên nhân chính là do khả năng tổng quát hóa của mô hình DFG trên các ảnh thật có tư thế/điều kiện khó.

---

## 💡 Kết Luận và Hướng Phát Triển

Dự án đã triển khai thành công một pipeline FAS phức tạp, chứng minh hiệu quả của hướng tiếp cận dựa trên tín hiệu bất thường trong việc phát hiện tấn công. Thách thức lớn nhất được xác định là khả năng tổng quát hóa của mô hình DFG.

**Hướng phát triển trong tương lai:**
1.  **Cải thiện DFG:** Huấn luyện lại DFG trên một tập dữ liệu thật đa dạng hơn về góc mặt, biểu cảm và điều kiện ánh sáng để giảm BPCER.
2.  **Tối ưu hóa Tốc độ:** Nghiên cứu kỹ thuật **Chưng cất Kiến thức (Knowledge Distillation)** để tạo ra một phiên bản mô hình gọn nhẹ, có khả năng triển khai thời gian thực.
3.  **Đánh giá Chéo Miền (Cross-Domain):** Kiểm tra mô hình trên các bộ dữ liệu hoàn toàn khác (ví dụ: OULU-NPU, MSU-MFSD) để đánh giá khả năng tổng quát hóa một cách nghiêm ngặt.
