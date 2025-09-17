# File: src/scripts/evaluate_on_test_set.py (Nâng cấp để lưu báo cáo)

import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd # Import pandas để tạo bảng đẹp

# ... (Các import khác giữ nguyên)
from src.configs.config_loader import cfg
from src.data_management.datasets import get_oanet_dataloader
from src.models.oanet.network import OANet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def calculate_metrics(all_labels, all_preds):
    # ... (Hàm này giữ nguyên)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0
    acer = (apcer + bpcer) / 2
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    return accuracy, apcer, bpcer, acer, cm

def evaluate_test_set():
    logger = logging.getLogger('EVALUATION')
    # ... (Phần tải model và dữ liệu giữ nguyên)

    # --- Chạy Inference (Giữ nguyên) ---
    # ...

    # --- Tính toán và Tạo Báo Cáo ---
    accuracy, apcer, bpcer, acer, cm = calculate_metrics(np.array(all_labels), np.array(all_preds))

    # Chuẩn bị nội dung báo cáo
    report_header = "BÁO CÁO ĐÁNH GIÁ HIỆU SUẤT TRÊN TẬP TEST"
    report_content = []
    report_content.append("="*len(report_header))
    report_content.append(report_header)
    report_content.append("="*len(report_header) + "\n")
    report_content.append(f"  - Tổng số mẫu trong tập test: {len(all_labels)}")
    report_content.append(f"  - Ngưỡng phân loại (Threshold): {cfg.oanet.OANET_EVAL_THRESHOLD}\n")
    
    # Bảng Ma trận nhầm lẫn
    df_cm = pd.DataFrame(cm, index=['Actual Real', 'Actual Spoof'], columns=['Predicted Real', 'Predicted Spoof'])
    report_content.append("  MA TRẬN NHẦM LẪN (Confusion Matrix):\n")
    report_content.append(df_cm.to_string() + "\n")

    # Bảng Chỉ số hiệu suất
    metrics_data = {
        'Chỉ Số': ['Accuracy', 'APCER (Spoof -> Real)', 'BPCER (Real -> Spoof)', 'ACER (Trung bình lỗi)'],
        'Giá Trị (%)': [accuracy * 100, apcer * 100, bpcer * 100, acer * 100]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics['Giá Trị (%)'] = df_metrics['Giá Trị (%)'].map('{:.2f}%'.format)
    report_content.append("\n  CÁC CHỈ SỐ HIỆU SUẤT:\n")
    report_content.append(df_metrics.to_string(index=False) + "\n")

    final_report = "\n".join(report_content)
    
    # In ra console
    print(final_report)

    # Lưu báo cáo ra file
    report_dir = Path(cfg.RESULTS_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "test_set_evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    logger.info(f"Đánh giá hoàn tất. Báo cáo chi tiết đã được lưu tại: {report_path}")

if __name__ == "__main__":
    evaluate_test_set()