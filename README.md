

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
