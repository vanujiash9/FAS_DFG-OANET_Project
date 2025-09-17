import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import json
import random
import shutil
import cv2
from torchvision.transforms.functional import resize, to_tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg

# --- Cấu hình ---
RAW_ROOT = Path(cfg.data.RAW_DATA_DIR)
PROCESSED_ROOT = Path(cfg.data.PROCESSED_DATA_DIR)
OANET_BASE = PROCESSED_ROOT / "oanet_dataset"
DFG_BASE = PROCESSED_ROOT / "dfg_real_faces"

# Sử dụng GPU nếu có thể
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# MTCNN được cấu hình để chạy trên 1 GPU
mtcnn = MTCNN(
    keep_all=False, 
    device=device, 
    image_size=cfg.data.OANET_IMAGE_SIZE, # 224x224
    margin=20,
    select_largest=True,
    post_process=True
)

BATCH_SIZE = 512
CONF_THRESHOLD_LIVE = 0.95
NUM_WORKERS = cfg.data.NUM_DATALOADER_WORKERS

# Tỷ lệ split
SPLIT_RATIOS = {"train": 0.9, "val": 0.1}

def load_image(path):
    """Loads a single image. Returns None if there's an error."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def save_tensor_image(tensor, path):
    """Saves a torch tensor (C, H, W) as an image."""
    img_np = ((tensor + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_np).save(path)

def process_batch(img_paths):
    """Processes a batch of image paths: load, detect. Returns tensors."""
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(load_image, img_paths))
    
    valid_imgs = [img for img in results if img is not None]
    if not valid_imgs:
        return []
    
    with torch.no_grad():
        faces_tensors, probs = mtcnn(valid_imgs, return_prob=True)
    
    processed_faces = [f for f, p in zip(faces_tensors, probs) if f is not None and (p is None or p >= CONF_THRESHOLD_LIVE)]
    return processed_faces

def main():
    random.seed(42)
    
    print("--- starting live data preprocessing ---")

    # Clear DFG and OA-Net live processed dirs
    if DFG_BASE.exists():
        shutil.rmtree(DFG_BASE)
    if (OANET_BASE / "live").exists():
        shutil.rmtree(OANET_BASE / "live")

    # Process all live images
    print("\n[1/2] processing all live images...")
    
    ffhq_raw_paths = sorted(list((RAW_ROOT / "ffhq").rglob("*")))
    celeba_live_raw_paths = sorted(list((RAW_ROOT / "celeba_live").rglob("*")))
    all_live_raw_paths = ffhq_raw_paths + celeba_live_raw_paths
    
    processed_live_tensors = []
    for i in tqdm(range(0, len(all_live_raw_paths), BATCH_SIZE), desc="processing live images"):
        batch_paths = all_live_raw_paths[i:i+BATCH_SIZE]
        faces = process_batch(batch_paths)
        processed_live_tensors.extend(faces)
    
    print(f"\ntotal processed live images: {len(processed_live_tensors)}")
    
    print("\n[2/2] splitting data for DFG and OA-Net...")
    
    # DFG
    dfg_train_dir = DFG_BASE / "train"
    dfg_val_dir = DFG_BASE / "val"
    dfg_train_dir.mkdir(parents=True, exist_ok=True)
    dfg_val_dir.mkdir(parents=True, exist_ok=True)
    
    random.shuffle(processed_live_tensors)
    n_dfg_train = int(len(processed_live_tensors) * SPLIT_RATIOS["DFG"]["train"])
    dfg_train_tensors = processed_live_tensors[:n_dfg_train]
    dfg_val_tensors = processed_live_tensors[n_dfg_train:]
    
    for i, tensor in tqdm(enumerate(dfg_train_tensors), total=len(dfg_train_tensors), desc="saving dfg train files"):
        save_tensor_image(tensor, dfg_train_dir / f"live_{i:06d}.jpg")
    for i, tensor in tqdm(enumerate(dfg_val_tensors), total=len(dfg_val_tensors), desc="saving dfg val files"):
        save_tensor_image(tensor, dfg_val_dir / f"live_val_{i:06d}.jpg")
        
    print(f"dfg data: split into train ({len(dfg_train_tensors)}) and val ({len(dfg_val_tensors)}).")
    
    # OA-Net
    for split in ['train', 'val', 'test']:
        spoof_dir = OANET_BASE / split / "spoof"
        if spoof_dir.exists():
            num_spoof = len(list(spoof_dir.glob("*")))
            if num_spoof > 0:
                selected_live = random.sample(processed_live_tensors, num_spoof)
                live_dir = OANET_BASE / split / "live"
                live_dir.mkdir(parents=True, exist_ok=True)
                for i, tensor in tqdm(enumerate(selected_live), total=len(selected_live), desc=f"saving oanet {split} live files"):
                    save_tensor_image(tensor, live_dir / f"live_{i:06d}.jpg")

    print("\n[DONE] live data preprocessing and splitting complete.")

if __name__ == "__main__":
    main()