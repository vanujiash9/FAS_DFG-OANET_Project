import os
import sys
from pathlib import Path
import shutil
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg

# --- Cấu hình Paths ---
PROCESSED_ROOT = Path(cfg.data.PROCESSED_DATA_DIR)
DFG_TRAIN_DIR = PROCESSED_ROOT / "dfg_real_faces" / "train"
DFG_VAL_DIR = PROCESSED_ROOT / "dfg_real_faces" / "val"
OANET_BASE = PROCESSED_ROOT / "oanet_dataset"

def collect_image_paths(directory: Path):
    """Thu thập tất cả đường dẫn ảnh từ một thư mục."""
    if not directory.exists():
        print(f"warning: directory not found - {directory}")
        return []
    return sorted([p for p in directory.rglob("*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])

def main():
    print("--- starting preparation of live data for oa-net ---")

    # 1. Dọn dẹp các thư mục live cũ của OA-Net (nếu có)
    for split in ['train', 'val', 'test']:
        live_dir = OANET_BASE / split / "live"
        if live_dir.exists():
            print(f"[CLEANUP] removing old oa-net live directory: {live_dir}")
            shutil.rmtree(live_dir)
        live_dir.mkdir(parents=True, exist_ok=True)

    # 2. Thu thập tất cả các ảnh live đã được xử lý cho DFG
    print("collecting all processed live images from dfg directories...")
    all_live_paths = collect_image_paths(DFG_TRAIN_DIR) + collect_image_paths(DFG_VAL_DIR)
    
    total_live_available = len(all_live_paths)
    print(f"found {total_live_available} total processed live images.")

    if total_live_available == 0:
        print("error: no processed live images found. please run preprocess.py first.")
        return

    # 3. Chia ảnh live cho OA-Net dựa trên số lượng ảnh spoof đã có
    for split in ['train', 'val', 'test']:
        spoof_dir = OANET_BASE / split / "spoof"
        live_dir = OANET_BASE / split / "live"
        
        if not spoof_dir.exists():
            print(f"spoof directory for '{split}' split not found. skipping.")
            continue
            
        spoof_count = len(list(spoof_dir.glob("*")))
        if spoof_count == 0:
            print(f"no spoof images in '{split}' split. skipping.")
            continue

        print(f"\nbalancing '{split}' split with {spoof_count} live images...")
        
        if total_live_available >= spoof_count:
            selected_paths = random.sample(all_live_paths, spoof_count)
            # Xóa các ảnh đã chọn để không bị trùng lặp giữa các tập
            all_live_paths = [p for p in all_live_paths if p not in selected_paths]
            total_live_available = len(all_live_paths)
        else:
            print(f"warning: not enough live images ({total_live_available}) to balance {split} split. "
                  f"using all remaining live images.")
            selected_paths = all_live_paths
            all_live_paths = []
            total_live_available = 0
            
        # Copy các file đã chọn
        for src_path in tqdm(selected_paths, desc=f"copying to {split}/live"):
            dest_path = live_dir / src_path.name
            shutil.copy2(str(src_path), str(dest_path))

    final_train_live = len(list((OANET_BASE / "train" / "live").glob("*")))
    final_val_live = len(list((OANET_BASE / "val" / "live").glob("*")))
    final_test_live = len(list((OANET_BASE / "test" / "live").glob("*")))
    
    print(f"\n[DONE] live data preparation for oa-net complete.")
    print(f"total live images copied: train={final_train_live}, val={final_val_live}, test={final_test_live}")

if __name__ == "__main__":
    main()