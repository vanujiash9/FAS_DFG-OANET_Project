# src/scripts/copy_live_for_oanet.py
import shutil
from pathlib import Path

# Thư mục gốc DFG chứa ảnh live đã xử lý
DFG_ROOT = Path("FAS_project/data/processed/dfg")

# Thư mục đích OA-Net live
OANET_LIVE_ROOT = Path("FAS_project/data/processed/oanet_dataset/live")

# Xóa thư mục live OA-Net cũ nếu tồn tại
if OANET_LIVE_ROOT.exists():
    print(f"[INFO] Xóa thư mục live cũ: {OANET_LIVE_ROOT}")
    shutil.rmtree(OANET_LIVE_ROOT)

# Copy các thư mục con (ffhq, celeba, ...) từ DFG sang OA-Net
for folder in [d for d in DFG_ROOT.iterdir() if d.is_dir()]:
    src = folder
    dst = OANET_LIVE_ROOT / folder.name
    print(f"[INFO] Copy {src} -> {dst}")
    shutil.copytree(src, dst)

print("[DONE] Đã copy tất cả ảnh live sang OA-Net dataset")
