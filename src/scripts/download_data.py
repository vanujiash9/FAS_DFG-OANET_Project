import os
import sys
import time
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import shutil
import random
import io
from huggingface_hub import hf_hub_download # NEW: Import hf_hub_download

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg

HUGGING_FACE_TOKEN = "YOUR HUGGING FACE TOKEN"

def save_image_from_hf_dataset(image_obj, save_path):
    if image_obj.mode == 'L':
        image_obj = image_obj.convert('RGB')
    image_obj.save(save_path)

def process_and_save_images_from_cache(
    repo_id, split_name, image_key, num_images_to_save, target_dir, 
    filename_prefix, filter_value=None, filter_column=None, 
    progress_interval=5000, random_sample_count=None
):
    start_time = time.time()
    saved_count = 0
    print(f"\n--- starting processing and saving up to {num_images_to_save} images from {repo_id} ({split_name}) ---")
    print(f"saving to: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    try:
        existing_files_count = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if existing_files_count >= num_images_to_save:
            print(f"found {existing_files_count} existing images in target directory. skipping processing.")
            return existing_files_count

        print(f"current images in target directory: {existing_files_count}. processing remaining {num_images_to_save - existing_files_count}.")
        
        dataset = load_dataset(repo_id, split=split_name, streaming=False, token=HUGGING_FACE_TOKEN)
        
        # --- Apply filter if provided ---
        if filter_value is not None and filter_column:
            if filter_column not in dataset.column_names:
                raise KeyError(f"Filter column '{filter_column}' not found in dataset '{repo_id}'. Available columns: {dataset.column_names}")
            dataset = dataset.filter(lambda example: example[filter_column] == filter_value)
            print(f"DEBUG: Dataset filtered for '{filter_column}' == '{filter_value}'. New length: {len(dataset)}")


        # Adjust num_images_to_save based on actual dataset length if it's smaller
        actual_dataset_len = len(dataset)
        if num_images_to_save > actual_dataset_len:
            print(f"WARNING: Requested {num_images_to_save} images, but dataset only has {actual_dataset_len}. Saving all available.")
            num_images_to_save = actual_dataset_len

        # Handle random sampling if required
        if random_sample_count and actual_dataset_len > random_sample_count:
            indices = list(range(actual_dataset_len))
            random.seed(42)
            selected_indices = random.sample(indices, random_sample_count)
            dataset = dataset.select(selected_indices)
            num_images_to_save = len(dataset)
            print(f"randomly selected {len(dataset)} examples for processing. New target: {num_images_to_save} images.")

        
        tqdm_initial = existing_files_count
        tqdm_total = num_images_to_save
        
        pbar = tqdm(total=tqdm_total, initial=tqdm_initial, desc=f"saving {filename_prefix} to dir")

        for i, example in enumerate(dataset):
            if saved_count >= num_images_to_save:
                break
            
            image_filename = f"{filename_prefix}_{i:05d}.jpg"
            save_path = os.path.join(target_dir, image_filename)
            
            if os.path.exists(save_path):
                saved_count += 1
                pbar.update(1)
                continue

            # --- CUSTOM LOGIC FOR HANDLING IMAGE DATA BASED ON DETECTED KEY ---
            img_content = example[image_key]
            img_pil = None
            
            if isinstance(img_content, Image.Image): # This is the ideal case
                img_pil = img_content
            elif isinstance(img_content, dict) and 'bytes' in img_content: # e.g., for 'webp' column as dict with bytes
                img_pil = Image.open(io.BytesIO(img_content['bytes'])).convert('RGB')
            elif isinstance(img_content, bytes): # Raw bytes (could be from 'webp' directly)
                img_pil = Image.open(io.BytesIO(img_content)).convert('RGB')
            elif isinstance(img_content, str): # This is the case for Ar4ikov/celebA_spoof 'Filepath'
                # For `datasets` when image_key is a filepath, it usually auto-loads it to PIL.Image
                # This block should ideally not be reached if `datasets` is working as intended for image columns.
                # If it is reached, it means `datasets` didn't auto-convert. We'll try manual loading from cache path.
                try:
                    # `datasets` often converts filepaths to PIL Images automatically when accessed.
                    # If it's still a string, it means datasets didn't auto-convert.
                    # We need the base path of the dataset cache for 'Filepath'.
                    # This is quite complex to get reliably.
                    # A safer bet: if it's a string, try to open it directly if it's an absolute path.
                    # If it's a relative path, we need dataset's root.
                    if os.path.isabs(img_content): # If it's an absolute path already
                        img_pil = Image.open(img_content).convert('RGB')
                    else: # If it's a relative path, assume it's relative to the snapshot directory
                        # This requires knowing the snapshot dir path which is hidden by `load_dataset`
                        # The best way is to let `datasets` do its job.
                        # If `example[key]` is indeed a string filepath, we assume it's fully resolvable
                        # or that `datasets` would have converted it.
                        raise ValueError(f"Image data for key '{image_key}' is a string filepath ('{img_content}'), "
                                         "not a PIL Image or bytes. `datasets` auto-conversion failed or dataset structure changed."
                                         "Please verify the dataset structure on Hugging Face Viewer.")
                except Exception as file_read_e:
                    print(f"WARNING: Could not process image from string content. Error: {file_read_e}")
                    continue # Skip this example

            else: 
                raise ValueError(f"Unexpected image data type for key '{image_key}': {type(img_content)}. Skipping example {i}.")

            if img_pil:
                save_image_from_hf_dataset(img_pil, save_path)
            else:
                raise ValueError("Image data could not be prepared as PIL Image for saving.")
            # --- END CUSTOM LOGIC ---

            saved_count += 1
            pbar.update(1)

            if saved_count % progress_interval == 0:
                print(f"saved {saved_count}/{num_images_to_save} {filename_prefix} images.")
        pbar.close()

    except Exception as e:
        print(f"error processing {filename_prefix} images: {e}")
        print("please check dataset integrity in cache, or disk space.")
        print(f"specific error: {e}")

    end_time = time.time()
    final_count = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\nfinished {filename_prefix} processing. total images saved: {final_count}")
    print(f"time taken: {end_time - start_time:.2f} seconds")
    return final_count

def check_celeba_spoof_spoof_files(expected_num_images):
    print(f"\n--- kiểm tra ảnh giả mạo (spoof) từ celeba-spoof ---")
    print(f"thư mục đích cho ảnh giả mạo: {cfg.data.CELEBA_SPOOF_SPOOF_FRAMES_DIR}")
    os.makedirs(cfg.data.CELEBA_SPOOF_SPOOF_FRAMES_DIR, exist_ok=True)

    current_spoof_files_count = len([f for f in os.listdir(cfg.data.CELEBA_SPOOF_SPOOF_FRAMES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"số ảnh giả mạo hiện có trong thư mục: {current_spoof_files_count}")
    if current_spoof_files_count < expected_num_images:
        print(f"cần thêm {expected_num_images - current_spoof_files_count} ảnh giả mạo để đạt đủ {expected_num_images}.")
        print(f"bạn có thể tải chúng thủ công từ nguồn celeba-spoof hoặc từ bộ dữ liệu `ar4ikov/celebA_spoof` (lọc `Class`='spoof') và đặt vào thư mục.")
    else:
        print("đã tìm thấy đủ ảnh giả mạo.")
    return current_spoof_files_count

def download_arcface_from_hf():
    """Tải trọng số Arcface từ Hugging Face Hub."""
    print(f"\n--- downloading arcface weights from hugging face hub ---")
    
    # Một repository ArcFace PyTorch đáng tin cậy trên Hugging Face
    repo_id = "jayanta/arcface-torch"
    filename = "iresnet50.pth"
    dest_path = cfg.dfg.ARCFACE_PRETRAINED_PATH
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        print(f"arcface weights already exist at {dest_path}. skipping download.")
        return

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(dest_path),
            local_dir_use_symlinks=False, # Copy the file instead of symlinking
            resume_download=True
        )
        # Rename the downloaded file to match our config
        shutil.move(os.path.join(os.path.dirname(dest_path), filename), dest_path)
        print(f"\n[SUCCESS] arcface weights downloaded successfully to {dest_path}!")
    except Exception as e:
        print(f"\n[ERROR] could not download arcface weights: {e}")


if __name__ == "__main__":
    print("----- bắt đầu quá trình chuẩn bị dữ liệu (tự động từ hugging face + kiểm tra thủ công) -----")
    
    os.makedirs(cfg.data.FFHQ_RAW_DIR, exist_ok=True)
    os.makedirs(cfg.data.CELEBA_SPOOF_LIVE_FRAMES_DIR, exist_ok=True)
    os.makedirs(cfg.data.CELEBA_SPOOF_SPOOF_FRAMES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.dfg.ARCFACE_PRETRAINED_PATH), exist_ok=True)

    # 1. Tải trọng số Arcface trước tiên (vì nó nhanh và quan trọng)
    download_arcface_from_hf()
    
    # 2. Tải/Xử lý FFHQ (70,000 images)
    ffhq_downloaded = process_and_save_images_from_cache(
        repo_id="gaunernst/ffhq-1024-wds",
        split_name="train",
        image_key="webp", # Khóa ảnh cho FFHQ là 'webp'
        num_images_to_save=cfg.data.FFHQ_DOWNLOAD_COUNT,
        target_dir=cfg.data.FFHQ_RAW_DIR,
        filename_prefix="ffhq"
    )
    
    # 3. Tải/Xử lý CelebA-Spoof Live (70,000 images)
    celeba_live_downloaded = process_and_save_images_from_cache(
        repo_id="Ar4ikov/celebA_spoof",
        split_name="train",
        image_key="Filepath", # Khóa ảnh cho CelebA-Spoof là 'Filepath'
        num_images_to_save=cfg.data.CELEBA_LIVE_DOWNLOAD_COUNT,
        target_dir=cfg.data.CELEBA_SPOOF_LIVE_FRAMES_DIR,
        filename_prefix="celeba_live",
        filter_value="live",
        filter_column='Class', # Khóa nhãn cho CelebA-Spoof là 'Class'
        random_sample_count=None
    )
    
    # 4. Kiểm tra ảnh Spoof
    spoof_checked = check_celeba_spoof_spoof_files(expected_num_images=cfg.data.CELEBA_SPOOF_DOWNLOAD_COUNT)
    
    print("\n----- hoàn tất quá trình chuẩn bị dữ liệu -----")
    print(f"tổng số ảnh ffhq đã tải: {ffhq_downloaded}")
    print(f"tổng số ảnh live celeba-spoof đã tải: {celeba_live_downloaded}")
    print(f"số ảnh giả mạo celeba-spoof đã kiểm tra: {spoof_checked}")
    print("vui lòng kiểm tra các thư mục và đảm bảo tất cả dữ liệu đã đầy đủ trước khi chạy script tiền xử lý.")