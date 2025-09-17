# ==============================================================================
# SCRIPT BÁO CÁO TOÀN DIỆN (PHIÊN BẢN SỬA LỖI TRIỆT ĐỂ)
# ==============================================================================

import os
import sys
from pathlib import Path
import random
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import logging
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# --- Cấu hình hệ thống ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.data_management.datasets import get_oanet_dataloader, get_oanet_eval_transforms
from src.models.oanet.network import OANet
from src.models.dfg.generator import DFGGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def generate_comprehensive_report(oanet_model, report_path):
    logger = logging.getLogger('REPORT_GENERATION')
    
    # --- PHẦN 1: THỐNG KÊ DỮ LIỆU ---
    live_dir_train = Path(cfg.data.DFG_REAL_FACES_TRAIN_DIR)
    num_live_initial = len(list(live_dir_train.glob('*.[jp][pn]g')))

    spoof_images_root = Path(cfg.data.PROCESSED_DATA_DIR) / 'oanet_dataset' / 'spoof'
    ignore_dirs = {'train', 'val', 'test'}
    spoof_subfolders = [d for d in spoof_images_root.iterdir() if d.is_dir() and d.name not in ignore_dirs and not d.name.startswith('.')]
    spoof_counts = {folder.name: len(list(folder.glob('*.[jp][pn]g'))) for folder in spoof_subfolders}
    num_spoof_initial = sum(spoof_counts.values())

    splits_file_path = Path(cfg.data.PROCESSED_DATA_DIR) / 'oanet_cues_splits.json'
    with open(splits_file_path, 'r') as f:
        splits_data = json.load(f)
    
    train_count = len(splits_data['train']); val_count = len(splits_data['val']); test_count = len(splits_data['test'])
    all_cues = splits_data['train'] + splits_data['val'] + splits_data['test']
    num_live_cues_balanced = sum(1 for _, label in all_cues if label == 0)

    # --- PHẦN 2: ĐÁNH GIÁ HIỆU SUẤT TRÊN TẬP TEST ---
    device = next(oanet_model.parameters()).device
    oanet_model.eval()
    cfg.data.OANET_SPLITS_PATH = str(splits_file_path)
    test_dataloader = get_oanet_dataloader('test', cfg.oanet.OANET_BATCH_SIZE * 2, shuffle=False)
    
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Evaluating on Test Set"):
            images = images.to(device)
            logits = oanet_model(images)
            probs = torch.sigmoid(logits).squeeze()
            preds = (probs > cfg.oanet.OANET_EVAL_THRESHOLD).int()
            all_labels.extend(labels.cpu().numpy()); all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds); tn, fp, fn, tp = cm.ravel()
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0
    acer = (apcer + bpcer) / 2
    accuracy = (tp + tn) / (tp + tn + fn + fp)

    # --- PHẦN 3: TẠO FILE BÁO CÁO ---
    report_content = []; report_header = "BÁO CÁO TỔNG KẾT DỰ ÁN FACE ANTI-SPOOFING"
    report_content.append("="*len(report_header)); report_content.append(report_header); report_content.append("="*len(report_header) + "\n")
    report_content.append("\n--- PHẦN I: THỐNG KÊ DỮ LIỆU ---\n")
    df_spoof = pd.DataFrame(list(spoof_counts.items()), columns=['Loại Tấn Công (Spoof Type)', 'Số Lượng Ảnh Gốc']).sort_values(by='Loại Tấn Công (Spoof Type)').reset_index(drop=True)
    report_content.append("1.1. Chi tiết số lượng ảnh gốc theo từng loại tấn công:\n"); report_content.append(df_spoof.to_string() + "\n")
    summary_initial = {'Loại Dữ Liệu': ['Live (FFHQ/CelebA)', 'Spoof (Tổng hợp)'], 'Số Lượng Gốc': [num_live_initial, num_spoof_initial]}
    df_summary_initial = pd.DataFrame(summary_initial)
    report_content.append("\n1.2. Tổng hợp dữ liệu gốc:\n"); report_content.append(df_summary_initial.to_string(index=False) + "\n")
    summary_balanced = {'Loại Dữ Liệu': ['Live Cues', 'Spoof Cues'], 'Số Lượng (Sau Cân Bằng)': [num_live_cues_balanced, num_live_cues_balanced]}
    df_summary_balanced = pd.DataFrame(summary_balanced)
    report_content.append("\n1.3. Dữ liệu sau khi sinh cues và cân bằng:\n"); report_content.append(df_summary_balanced.to_string(index=False) + "\n")
    summary_split = {'Tập Dữ Liệu': ['Training', 'Validation', 'Test'], 'Số Lượng Mẫu': [train_count, val_count, test_count]}
    df_summary_split = pd.DataFrame(summary_split)
    report_content.append("\n1.4. Phân chia dữ liệu cuối cùng cho OA-Net:\n"); report_content.append(df_summary_split.to_string(index=False) + "\n")
    report_content.append("\n--- PHẦN II: KẾT QUẢ ĐÁNH GIÁ HIỆU SUẤT TRÊN TẬP TEST ---\n")
    df_cm = pd.DataFrame(cm, index=['Thực Tế: Real', 'Thực Tế: Spoof'], columns=['Dự Đoán: Real', 'Dự Đoán: Spoof'])
    report_content.append("2.1. Ma trận nhầm lẫn (Confusion Matrix):\n"); report_content.append(df_cm.to_string() + "\n")
    metrics_data = {'Chỉ Số': ['Accuracy', 'APCER (Lỗi nhận nhầm Spoof)', 'BPCER (Lỗi nhận nhầm Real)', 'ACER (Lỗi trung bình)'], 'Giá Trị': [f"{accuracy * 100:.2f}%", f"{apcer * 100:.2f}%", f"{bpcer * 100:.2f}%", f"{acer * 100:.2f}%"]}
    df_metrics = pd.DataFrame(metrics_data)
    report_content.append("\n2.2. Các chỉ số hiệu suất chính:\n"); report_content.append(df_metrics.to_string(index=False) + "\n")
    final_report = "\n".join(report_content)
    print(final_report)
    with open(report_path, 'w', encoding='utf-8') as f: f.write(final_report)
    logger.info(f"Báo cáo tổng kết đã được lưu tại: {report_path}")

def visualize_sample_cues(dfg_model, viz_dir, num_samples=5):
    logger = logging.getLogger('VISUALIZATION')
    logger.info("="*50); logger.info("   TRỰC QUAN HÓA MẪU DỮ LIỆU CUES"); logger.info("="*50)
    device = next(dfg_model.parameters()).device
    dfg_model.eval()
    live_dir_val = Path(cfg.data.DFG_REAL_FACES_VAL_DIR)
    live_samples = random.sample(list(live_dir_val.glob('*.[jp][pn]g')), num_samples)
    spoof_images_root = Path(cfg.data.PROCESSED_DATA_DIR) / 'oanet_dataset' / 'spoof'
    all_spoof_images = [p for d in spoof_images_root.iterdir() if d.is_dir() and d.name not in {'train', 'val', 'test'} for p in d.glob('*.[jp][pn]g')]
    spoof_samples = random.sample(all_spoof_images, num_samples)
    all_samples = [(p, 'live') for p in live_samples] + [(p, 'spoof') for p in spoof_samples]
    transform_func = get_oanet_eval_transforms()
    for img_path, label in tqdm(all_samples, desc="Generating Visualizations"):
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform_func(image).unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed_tensor, cue_tensor = dfg_model.generate_reconstruction_and_cue(image_tensor)
        original_img = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
        reconstructed_img = transforms.ToPILImage()(reconstructed_tensor.squeeze(0).cpu())
        cue_tensor_norm = (cue_tensor.squeeze(0) - cue_tensor.min()) / (cue_tensor.max() - cue_tensor.min())
        cue_img = transforms.ToPILImage()(cue_tensor_norm.cpu())
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Loại: {label.upper()} - File: {Path(img_path).name}", fontsize=16)
        axes[0].imshow(original_img); axes[0].set_title('Ảnh Gốc'); axes[0].axis('off')
        axes[1].imshow(reconstructed_img); axes[1].set_title('Ảnh Tái Tạo bởi DFG'); axes[1].axis('off')
        axes[2].imshow(cue_img, cmap='jet'); axes[2].set_title('Anomalous Cue'); axes[2].axis('off')
        save_path = viz_dir / f"{label}_{Path(img_path).stem}.png"
        plt.savefig(save_path)
        plt.close(fig)
    logger.info(f"Đã lưu {num_samples*2} ảnh trực quan hóa vào: {viz_dir}")

def main():
    # SỬA LỖI: Truy cập RESULTS_DIR thông qua nhóm 'oanet' trong config
    results_dir = Path(cfg.oanet.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "comprehensive_project_report.txt"
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    
    print("Đang tải các mô hình cần thiết cho báo cáo...")
    oanet_model_path = os.path.join(cfg.oanet.OANET_CHECKPOINTS_DIR, "best_oanet_vit_model.pth")
    if not os.path.exists(oanet_model_path):
        print(f"Lỗi: Không tìm thấy model OA-Net tại {oanet_model_path}.")
        return
    
    oanet_model = OANet(device=device)
    state_dict = torch.load(oanet_model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    oanet_model.load_state_dict(state_dict)
    oanet_model.to(device)

    dfg_unet_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, "best_dfg_unet.pth")
    if not os.path.exists(dfg_unet_path):
        print(f"Lỗi: Không tìm thấy model DFG tại {dfg_unet_path}.")
        return
    
    dfg_model = DFGGenerator(device=device)
    dfg_model.unet.load_state_dict(torch.load(dfg_unet_path, map_location=device))
    dfg_model.to(device)
    print("Tải mô hình thành công.")

    generate_comprehensive_report(oanet_model, report_path)
    visualize_sample_cues(dfg_model, viz_dir, num_samples=5)
    
    print(f"\n--- BÁO CÁO VÀ TRỰC QUAN HÓA HOÀN TẤT ---")
    print(f"Toàn bộ kết quả đã được lưu tại thư mục: {results_dir.absolute()}")

if __name__ == "__main__":
    main()