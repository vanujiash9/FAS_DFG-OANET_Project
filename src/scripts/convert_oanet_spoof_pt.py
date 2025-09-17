# src/scripts/convert_oanet_spoof_pt.py
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor

SPOOF_DIR = Path("data/processed/oanet_dataset/spoof")
PT_DIR = Path("data/processed/oanet_dataset/spoof_pt")
PT_DIR.mkdir(parents=True, exist_ok=True)

spoof_types = [d for d in SPOOF_DIR.iterdir() if d.is_dir()]

for spoof_type in spoof_types:
    out_type_dir = PT_DIR / spoof_type.name
    out_type_dir.mkdir(exist_ok=True)
    img_files = list(spoof_type.rglob("*.jpg")) + list(spoof_type.rglob("*.png"))
    
    for img_file in img_files:
        img = Image.open(img_file).convert("RGB")
        tensor = to_tensor(img) * 2 - 1  # normalize [-1,1]
        torch.save(tensor, out_type_dir / (img_file.stem + ".pt"))

print(f"[DONE] Converted OA-Net spoof sang PT tại {PT_DIR}")
