import os
import sys
import shutil
from huggingface_hub import hf_hub_download

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg

HUGGING_FACE_TOKEN = "YOUR HUGGINGFACE TOKEN"

def download_arcface_from_hf():
    """Tải trọng số Arcface từ Hugging Face Hub."""
    print(f"\n--- downloading arcface weights from hugging face hub ---")
    
    # Một repository ArcFace PyTorch đáng tin cậy khác trên Hugging Face
    repo_id = "deepinsight/iresnet50"
    filename = "iresnet50.pth"
    dest_path = cfg.dfg.ARCFACE_PRETRAINED_PATH
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        print(f"arcface weights already exist at {dest_path}. skipping download.")
        return

    try:
        print(f"downloading {filename} from {repo_id}...")
        cached_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True,
            token=HUGGING_FACE_TOKEN
        )
        
        print(f"copying downloaded file to {dest_path}...")
        shutil.copy2(cached_file_path, dest_path)
        print(f"\n[SUCCESS] arcface weights downloaded and copied successfully to {dest_path}!")
        
        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"file size: {file_size_mb:.2f} MB (expected around 240 MB)")
    except Exception as e:
        print(f"\n[ERROR] could not download arcface weights: {e}")


if __name__ == "__main__":
    download_arcface_from_hf()