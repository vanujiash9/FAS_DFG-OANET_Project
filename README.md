Dưới đây là phiên bản trình bày lại README/report dự án hệ thống FAS (AG-FAS: DFG & OA-Net) rõ ràng, hiện đại, cấu trúc chuyên nghiệp, nhấn mạnh khoa học – đẹp và súc tích đúng chuẩn dự án học thuật hoặc open source.

***

# Face Anti-Spoofing System with Generative Modeling (DFG & OA-Net)

## 🚀 Dự án & Tác giả
- **Tác giả:** Bùi Thị Thanh Vân  
- **Trường:** Đại học Giao thông Vận tải TP.HCM  
- **Email:** thanh.van19062004@gmail.com  
- **Thời gian:** 09/2025  

## 1️⃣ Tổng Quan Mục Tiêu

Hệ thống xây dựng theo hai pha độc đáo AG-FAS:
- **DFG:** Sinh mẫu khuôn mặt thật, phát hiện các vùng bất thường.
- **OA-Net:** Nhận diện spoof nhờ học sâu các cues giả mạo, ứng dụng cơ chế Vision Transformer kết hợp Cross-Attention.

Pipeline này chống lại đa dạng hình thức tấn công: print, replay, mask, deepfake, v.v., đạt hiệu quả vượt trội cho kiểm thử cross-domain.

***

## 2️⃣ Kiến Trúc & Triết Lý

### 🌀 1. De-fake Face Generator (DFG)
- **Huấn luyện 100% trên ảnh live** – học chi tiết “một khuôn mặt người thật” nên có.
- Khi nhận input (live/spoof), DFG tái tạo phiên bản lý tưởng nhất của mặt thật.
- Đầu ra: 
  - Ảnh gốc (input)
  - Ảnh tái tạo (by DFG)
  - **Anomalous Cue:** Hiệu số tuyệt đối – cue tối đen (live), cue sáng rõ vùng giả mạo (spoof).

### 🤖 2. OA-Net & Cross-Attention
- **Nhận đầu vào là cues.** Không dùng raw face.
- **Backbone:** ResNet cue encoder + ViT, với cross-attention từng lớp.
- **Cross-Attention:** Kết hợp thông minh cues spatial (CNN) & global (ViT), tập trung vào vùng bất thường đặc thù giả mạo.
- Huấn luyện tách subject, kiểm tra generalization, đảm bảo không “học vẹt” mà tìm bằng chứng giả mạo thực thụ.[2]

***

## 3️⃣ Phân Tích & Trực Quan Hóa Dữ Liệu

### **Nguồn và Quy mô dữ liệu**
- 33.625 ảnh live (FFHQ, CelebA)
- 33.433 ảnh spoof (10 loại attack như 3D mask, poster, region mask…)
- Dữ liệu sau khi sinh cues và cân bằng: 3.680 live cues – 3.680 spoof cues

### **Phân bố & trực quan hóa**
- Biểu đồ bar: Số lượng từng loại spoof
- Pie chart: Tỉ lệ train/val/test (5936/742/742)
- Minh họa hình ảnh: Mỗi loại spoof, cues thật và giả mạo
- Trực quan hóa pipeline: Ảnh gốc – ảnh DFG – anomalous cue (nhấn mạnh difference vùng spoof)

***

## 4️⃣ Kết Quả – Đánh Giá Hiệu Suất

- Train: 5936 | Val: 742 | Test: 742
- **Ma trận nhầm lẫn – Heatmap:**  
  |           | Dự đoán Real | Dự đoán Spoof |
  |-----------|--------------|---------------|
  | Thực tế Real | 284          | 93            |
  | Thực tế Spoof| 25           | 340           |

- **Chỉ số trên tập test:**
  - Accuracy: **84.10%**
  - APCER (Lỗi nhận nhầm spoof): **6.85%**
  - BPCER (Lỗi nhận nhầm người thật): **24.67%**
  - ACER (Lỗi trung bình): **15.76%**

- Đường cong train/val loss, confusion matrix, các hình cues minh họa đã được lưu trực quan trong `/results/charts`

***

## 5️⃣ Yêu Cầu & Thiết lập nhanh

### **Yêu cầu phụ thuộc**
- Python >=3.8, PyTorch >=1.10, Transformers, Diffusers, OpenCV, Scikit-learn, Pandas, Matplotlib, Seaborn

### **Thiết lập & Chạy thử**
```bash
git clone https://github.com/vanujiash9/FAS_DFG-OANET_Project.git
cd FAS_DFG-OANET_Project
pip install -r requirements.txt
python3 src/scripts/run_full_pipeline.py
```
- Tạo báo cáo, trực quan hóa chỉ cần:
```bash
python3 src/scripts/generate_full_report.py
```

***

## 6️⃣ Hướng phát triển tương lai

- Giảm BPCER: Mở rộng tập live và tách biệt domain ánh sáng/góc mặt
- Đánh giá cross-domain, kiểm thử unseen spoof attack
- Tích hợp tấn công deepfake, AR filter
- Rút gọn mô hình (distillation) để triển khai on-device

***

**Mã nguồn và báo cáo kết quả trực quan sẵn sàng minh bạch, hỗ trợ mọi kiểm thử – thích hợp cho nghiên cứu và sản phẩm thực tế.**
