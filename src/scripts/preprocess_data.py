# src/scripts/preprocess_data.py
import os
import shutil
from pathlib import Path
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# ==== CONFIGURATION ====
RAW_LIVE_DIRS = [
    Path("data/raw/celeba_live"),
    Path("data/raw/ffhq")
]
PROCESSED_DFG_DIR = Path("data/processed/dfg")
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Clean up and create the destination directory
if PROCESSED_DFG_DIR.exists():
    print(f"[INFO] Removing old directory: {PROCESSED_DFG_DIR}")
    shutil.rmtree(PROCESSED_DFG_DIR)
PROCESSED_DFG_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Starting live faces data preprocessing for DFG...")

# ==== MTCNN INITIALIZATION ====
mtcnn = MTCNN(keep_all=False, device=DEVICE, margin=20, post_process=False)

def process_and_save_face(image_path, save_dir):
    """
    Detects, crops, and saves a face from an image file.
    Returns True if successful, False otherwise.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)

    # Check if a face was detected with a high probability
    if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.5:
        x1, y1, x2, y2 = [int(c) for c in boxes[0]]
        
        # Crop the face from the original BGR image
        face = img[y1:y2, x1:x2]
        
        # Ensure the cropped face is not empty before resizing
        if face.size == 0:
            return False

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        save_path = save_dir / image_path.name
        cv2.imwrite(str(save_path), face)
        return True
    
    # Return False if no valid face was found
    return False

total_processed = 0
total_saved = 0

# ==== PROCESS EACH SOURCE DIRECTORY ====
for raw_dir in RAW_LIVE_DIRS:
    if not raw_dir.is_dir():
        print(f"[WARNING] Directory not found: {raw_dir}. Skipping.")
        continue

    print(f"[INFO] Processing source directory: {raw_dir}")
    for entry in tqdm(os.scandir(raw_dir), desc=f"Processing {raw_dir.name}"):
        if entry.is_file():
            img_path = Path(entry.path)
            total_processed += 1
            if process_and_save_face(img_path, PROCESSED_DFG_DIR):
                total_saved += 1

print(f"\n[DONE] Total images processed: {total_processed}")
print(f"[DONE] Live faces saved: {total_saved}")
if total_processed > 0:
    print(f"[INFO] Retention rate: {total_saved/total_processed:.2%}")
else:
    print("[INFO] No images found to process.")

print(f"[DONE] All processed images saved to: {PROCESSED_DFG_DIR}")