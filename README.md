Dưới đây là mẫu README chuyên nghiệp cho dự án của bạn, tối ưu để dùng trên GitHub, báo cáo hoặc chia sẻ học thuật.

***

# Hệ Thống Face Anti-Spoofing AG-FAS (DFG & OA-Net)

## 🚀 Giới thiệu

Hệ thống AG-FAS là giải pháp phát hiện giả mạo khuôn mặt hiện đại, áp dụng kiến trúc hai giai đoạn dựa trên mô hình sinh mẫu (DFG) và mạng phân loại attention (OA-Net). Mục đích của dự án là nhận diện hiệu quả mọi dạng tấn công spoof hiện đại, đồng thời giữ tỷ lệ báo động giả thấp cho người dùng thật.

- **Tác giả:** Bùi Thị Thanh Vân  
- **Đơn vị:** Trường ĐH Giao thông Vận tải TPHCM  
- **Email:** thanh.van19062004@gmail.com  
- **Thực hiện:** 09/2025

***

## ⚙️ Kiến trúc chính

### 1. Giai đoạn 1: De-fake Face Generator (DFG)
- **Chỉ học trên ảnh thật:** DFG dựa Latent Diffusion Model, tái tạo lại khuôn mặt thuần túy từ input.
- **Sinh anomalous cue:** So sánh tuyệt đối ảnh gốc – ảnh tái tạo, phát hiện vùng khác biệt (anomaly) – đặc biệt nhạy với spoof.

### 2. Giai đoạn 2: OA-Net (Off-real Attention Network)
- **Input:** Anomaly cue từ DFG  
- **Backbone song song:**  
   - **ViT-Base** (google/vit-base-patch16-224): Thu nhận quan hệ toàn cục từ cues.  
   - **ResNet-18**: Trích xuất chi tiết cục bộ (biên, kết cấu).  
- **Cross-Attention:** Kết hợp tính năng hai nhánh, giúp ViT tập trung vùng bất thường.
- **Classifier:** Phân loại output của token [CLS] thành xác suất Live/Spoof.

***

## 📊 Dữ liệu và pipeline

- **Data nguồn:** CelebA-Spoof (70K live, 35K spoof), FFHQ (20K real).
- **Tiền xử lý:** Cắt, resize 224x224, standardize, sinh cues anomaly.
- **Chia tập:** 80% train, 10% val, 10% test, không trùng subject giữa các tập.
- **Huấn luyện:** AdamW, ReduceLROnPlateau, Early Stopping, multi-GPU & AMP tối ưu tốc độ.

***

## 🏆 Hiệu suất nổi bật

- **Accuracy:** 84.1%
- **APCER:** 6.85%
- **BPCER:** 24.67%
- **ACER:** 15.76%
- **Checkpoint tốt nhất:** Epoch 4
- **Tự động trực quan hóa:** Biểu đồ loss, confusion matrix, ROC, các ảnh cue so sánh live/spoof.

***

## 🔬 Cài đặt & sử dụng

### Yêu cầu
- Python >=3.8, PyTorch >=1.10, Transformers, Diffusers, OpenCV, Pandas, Scikit-learn, Matplotlib

### Hướng dẫn nhanh
```bash
git clone https://github.com/vanujiash9/FAS_DFG-OANET_Project.git
cd FAS_DFG-OANET_Project
pip install -r requirements.txt

# Huấn luyện toàn bộ pipeline
python3 src/scripts/run_full_pipeline.py

# Tạo chỉ báo cáo và trực quan hóa dữ liệu
python3 src/scripts/generate_full_report.py
```

***

## 📈 Trực quan & phân tích

- Biểu đồ các loại spoof
- Loss curve, confusion matrix, ROC/AUC
- So sánh ảnh gốc/tái tạo/cue cho live và spoof

Tất cả kết quả lưu tại thư mục `results/`, thuận tiện kiểm định và báo cáo.

***

## 🌱 Hướng phát triển

- Bổ sung đánh giá cross-domain trên tập OULU-NPU, MSU-MFSD
- Mở rộng nhận diện deepfake, AR-filter
- Distillation, tối ưu hóa mô hình chạy thiết bị thực

***

**Mọi đóng góp, ý kiến hoặc hợp tác nghiên cứu vui lòng liên hệ tác giả!**
