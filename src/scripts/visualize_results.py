

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- Cấu hình hệ thống ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg

def main():
    """Hàm chính điều phối toàn bộ pipeline báo cáo."""
    
    print("Bắt đầu quá trình thống kê và trực quan hóa dữ liệu...")
    
    # --- 1. Thiết lập thư mục kết quả ---
    output_dir = Path(cfg.oanet.RESULTS_DIR) / "charts_data_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 2. Thống kê dữ liệu gốc (trước khi xử lý) ---
    ffhq_dir = Path(cfg.data.RAW_DATA_DIR) / 'ffhq'
    celeba_live_dir = Path(cfg.data.RAW_DATA_DIR) / 'celeba_live'
    celeba_spoof_dir = Path(cfg.data.RAW_DATA_DIR) / 'celeba_spoof' / 'raw_data'

    ffhq_count = len(list(ffhq_dir.glob('*.[jp][pn]g')))
    celeba_live_count = len(list(celeba_live_dir.glob('*.[jp][pn]g')))
    # Đếm tất cả các file trong 10 thư mục con
    celeba_spoof_count = sum(len(list(d.glob('*.[jp][pn]g'))) for d in celeba_spoof_dir.iterdir() if d.is_dir())

    total_live_raw = ffhq_count + celeba_live_count
    total_spoof_raw = celeba_spoof_count

    # --- 3. Thống kê dữ liệu cues (sau khi xử lý) ---
    live_cues_dir = Path(cfg.data.ANOMALOUS_CUES_DIR) / 'celeba_spoof_train'
    spoof_cues_dir = Path(cfg.data.ANOMALOUS_CUES_DIR) / 'celeba_spoof_spoof'
    
    live_cues_count = len(list(live_cues_dir.glob('*.pt')))
    spoof_cues_count = len(list(spoof_cues_dir.glob('*.pt')))

    # --- 4. Bắt đầu vẽ biểu đồ ---
    plt.style.use('seaborn-v0_8-talk')

    # Biểu đồ 1: Thống kê 3 bộ dữ liệu gốc
    source_counts = {
        'FFHQ (Live)': ffhq_count,
        'CelebA (Live)': celeba_live_count,
        'CelebA (Spoof)': celeba_spoof_count
    }
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    pd.Series(source_counts).plot(kind='bar', ax=ax1, color=['mediumseagreen', 'limegreen', 'indianred'])
    ax1.set_title('Raw Data Sources Breakdown', fontsize=18, weight='bold')
    ax1.set_ylabel('Number of Images', fontsize=14)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    for i, v in enumerate(pd.Series(source_counts)):
        ax1.text(i, v + 500, str(v), ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    save_path1 = output_dir / "1_raw_data_sources.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"-> Biểu đồ nguồn dữ liệu gốc đã lưu tại: {save_path1}")

    # Biểu đồ 2: Thống kê tổng live vs. spoof (gốc)
    raw_live_spoof_counts = {
        'Total Live (Raw)': total_live_raw,
        'Total Spoof (Raw)': total_spoof_raw
    }
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    pd.Series(raw_live_spoof_counts).plot(kind='bar', ax=ax2, color=['green', 'red'])
    ax2.set_title('Total Raw Live vs. Spoof Images', fontsize=18, weight='bold')
    ax2.set_ylabel('Number of Images', fontsize=14)
    ax2.tick_params(axis='x', rotation=0, labelsize=12)
    for i, v in enumerate(pd.Series(raw_live_spoof_counts)):
        ax2.text(i, v + 500, str(v), ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    save_path2 = output_dir / "2_raw_live_vs_spoof.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"-> Biểu đồ live vs. spoof (gốc) đã lưu tại: {save_path2}")

    # Biểu đồ 3: Thống kê số lượng cues (sau xử lý)
    processed_cues_counts = {
        'Live Cues': live_cues_count,
        'Spoof Cues': spoof_cues_count
    }
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    pd.Series(processed_cues_counts).plot(kind='bar', ax=ax3, color=['darkcyan', 'coral'])
    ax3.set_title('Processed Anomalous Cues Count', fontsize=18, weight='bold')
    ax3.set_ylabel('Number of Cues (.pt files)', fontsize=14)
    ax3.tick_params(axis='x', rotation=0, labelsize=12)
    for i, v in enumerate(pd.Series(processed_cues_counts)):
        ax3.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    save_path3 = output_dir / "3_processed_cues_count.png"
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"-> Biểu đồ số lượng cues đã lưu tại: {save_path3}")

    print(f"\n--- TRỰC QUAN HÓA HOÀN TẤT ---")
    print(f"Toàn bộ biểu đồ đã được lưu tại thư mục: {output_dir.absolute()}")

if __name__ == "__main__":

    main()