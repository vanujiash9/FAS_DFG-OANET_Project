# src/scripts/visualize_dataset_pt_ffhq.py
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random
import shutil

# Thư mục dữ liệu
DFG_ROOT = Path("data/processed/dfg")
OANET_ROOT = Path("data/processed/oanet_dataset")

LIVE_FOLDERS = ["ffhq", "celeba"]
SPOOF_FOLDER = OANET_ROOT / "spoof"

# Thư mục lưu kết quả
VIS_DIR = Path("results/visualizations")
if VIS_DIR.exists():
    shutil.rmtree(VIS_DIR)
VIS_DIR.mkdir(parents=True, exist_ok=True)

stats = {}
samples = {}

# ================== DFG Live ==================
for folder in LIVE_FOLDERS:
    path = DFG_ROOT / folder
    if folder == "ffhq":
        # Chỉ load file .pt
        files = list(path.rglob("*.pt"))
        imgs = []
        for pt_file in files:
            tensor = torch.load(pt_file)
            img_np = ((tensor + 1) * 127.5).permute(1,2,0).byte().cpu().numpy()
            imgs.append(img_np)
        stats[f"DFG {folder}"] = len(imgs)
        samples[f"DFG {folder}"] = random.sample(imgs, min(5, len(imgs)))
    else:
        files = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))
        stats[f"DFG {folder}"] = len(files)
        sample_files = random.sample(files, min(5, len(files)))
        imgs = [Image.open(f).convert("RGB") for f in sample_files]
        samples[f"DFG {folder}"] = imgs

for folder in LIVE_FOLDERS:
    path = OANET_ROOT / "live" / folder
    if folder == "ffhq":

        files = list(path.rglob("*.pt"))
        imgs = []
        for pt_file in files:
            tensor = torch.load(pt_file)
            img_np = ((tensor + 1) * 127.5).permute(1,2,0).byte().cpu().numpy()
            imgs.append(img_np)
        stats[f"OA-Net live {folder}"] = len(imgs)
        samples[f"OA-Net live {folder}"] = random.sample(imgs, min(5, len(imgs)))
    else:

        files = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))
        stats[f"OA-Net live {folder}"] = len(files)
        sample_files = random.sample(files, min(5, len(files)))
        imgs = [Image.open(f).convert("RGB") for f in sample_files]
        samples[f"OA-Net live {folder}"] = imgs


# ================== OA-Net Spoof ==================
spoof_types = []
for spoof_type_dir in sorted(SPOOF_FOLDER.iterdir()):
    if spoof_type_dir.is_dir():
        files = list(spoof_type_dir.rglob("*.jpg")) + list(spoof_type_dir.rglob("*.png"))
        stats[f"Spoof {spoof_type_dir.name}"] = len(files)
        sample_files = random.sample(files, min(5, len(files)))
        imgs = [Image.open(f).convert("RGB") for f in sample_files]
        samples[f"Spoof {spoof_type_dir.name}"] = imgs
        spoof_types.append(spoof_type_dir.name)

# ================== Lưu ảnh mẫu ==================
def save_sample_images(samples_dict, subfolder):
    out_dir = VIS_DIR / subfolder
    out_dir.mkdir(exist_ok=True)
    for key, img_list in samples_dict.items():
        key_dir = out_dir / key
        key_dir.mkdir(exist_ok=True)
        for i, img in enumerate(img_list):
            if isinstance(img, Image.Image):
                img.save(key_dir / f"{i+1}.jpg")
            else:
                # np array từ pt
                Image.fromarray(img).save(key_dir / f"{i+1}.jpg")

save_sample_images(samples, "samples")

# ================== Biểu đồ thống kê ==================
fig, ax = plt.subplots(figsize=(14,8))
labels = list(stats.keys())
counts = list(stats.values())

bars = ax.barh(labels, counts, color='skyblue')
ax.set_xlabel("Số lượng ảnh")
ax.set_title("Thống kê DFG + OA-Net Live + OA-Net Spoof")

for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(count), va='center')

plt.tight_layout()
plt.savefig(VIS_DIR / "dataset_overview.png", dpi=200)
plt.show()

print(f"[INFO] Thống kê chi tiết: {stats}")
print(f"[INFO] Danh sách loại spoof: {spoof_types}")
print(f"[INFO] Ảnh mẫu lưu tại: {VIS_DIR / 'samples'}")
