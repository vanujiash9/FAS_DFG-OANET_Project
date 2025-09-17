# src/scripts/visualize_split_pt_full.py
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random
import shutil
import numpy as np

# ================= Thư mục =================
SPLIT_DIR = Path("data/splits_pt")
VIS_DIR = Path("results/visualizations_split")
if VIS_DIR.exists():
    shutil.rmtree(VIS_DIR)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# ================= Hàm hỗ trợ =================
def load_pt_image(pt_file):
    """Load tensor từ .pt, hỗ trợ tensor trực tiếp hoặc dict['img']"""
    data = torch.load(pt_file)
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, dict) and "img" in data:
        tensor = data["img"]
    else:
        print(f"[WARN] Không tìm thấy key 'img' trong {pt_file.name}")
        return None
    # Chuẩn hóa về hình ảnh uint8
    if tensor.ndim == 3:  # C x H x W
        img_np = ((tensor + 1) * 127.5).permute(1,2,0).byte().cpu().numpy()
        return img_np
    return None

def save_sample_images(samples_dict, subfolder):
    out_dir = VIS_DIR / subfolder
    out_dir.mkdir(exist_ok=True)
    for key, img_list in samples_dict.items():
        key_dir = out_dir / key
        key_dir.mkdir(exist_ok=True)
        for i, img in enumerate(img_list):
            if isinstance(img, Image.Image):
                img.save(key_dir / f"{i+1}.jpg")
            elif isinstance(img, np.ndarray):
                Image.fromarray(img).save(key_dir / f"{i+1}.jpg")

# ================= Thống kê và ảnh mẫu =================
stats = {}
samples = {}

for split_dir in SPLIT_DIR.iterdir():  # dfg, oan
    if not split_dir.is_dir():
        continue
    split_name = split_dir.name
    stats[split_name] = 0
    samples[split_name] = {}
    for class_dir in split_dir.rglob("*"):  # train/val/test/live/spoof
        if class_dir.is_dir():
            imgs = []
            for f in class_dir.glob("*.pt"):
                img_np = load_pt_image(f)
                if img_np is not None:
                    imgs.append(img_np)
            if imgs:
                key = class_dir.relative_to(SPLIT_DIR).as_posix()
                samples[split_name][key] = random.sample(imgs, min(5, len(imgs)))
                stats[f"{split_name} {key}"] = len(imgs)
                stats[split_name] += len(imgs)

# ================= Lưu ảnh mẫu =================
save_sample_images(samples, "samples")

# ================= Vẽ biểu đồ =================
fig, ax = plt.subplots(figsize=(14,8))
labels = list(stats.keys())
counts = list(stats.values())

bars = ax.barh(labels, counts, color='skyblue')
ax.set_xlabel("Số lượng ảnh")
ax.set_title("Thống kê các splits PT dataset DFG + OA-Net")

for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(count), va='center')

plt.tight_layout()
plt.savefig(VIS_DIR / "dataset_overview.png", dpi=200)
plt.show()

print(f"[INFO] Thống kê chi tiết: {stats}")
print(f"[INFO] Ảnh mẫu lưu tại: {VIS_DIR / 'samples'}")
