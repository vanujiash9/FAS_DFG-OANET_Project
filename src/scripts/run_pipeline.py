import os
import sys
from pathlib import Path
import random
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import logging
from torch.cuda.amp import autocast, GradScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.data_management.datasets import get_oanet_dataloader
from src.models.oanet.network import OANet
from src.training_utils.losses import oanet_loss_fn
from src.training_utils.optimizers import get_oanet_optimizer
from src.training_utils.schedulers import get_oanet_lr_scheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    
    logger = logging.getLogger('TRAINING_PIPELINE')
    print("\n[INFO] Bỏ qua Giai đoạn 1: Tạo Cues.")
    print("[INFO] Bắt đầu từ Giai đoạn 2: Chia dữ liệu và Huấn luyện OA-Net.")

    logger.info("="*50)
    logger.info("   STAGE 2: SPLITTING CUES DATASET")
    logger.info("="*50)

    live_cues_dir = Path(cfg.data.ANOMALOUS_CUES_DIR) / 'celeba_spoof_train'
    spoof_cues_dir = Path(cfg.data.ANOMALOUS_CUES_DIR) / 'celeba_spoof_spoof'

    if not live_cues_dir.is_dir() or not spoof_cues_dir.is_dir():
        logger.error(f"Lỗi: Không tìm thấy thư mục cues.")
        sys.exit(1)

    live_cues = [(f"{live_cues_dir.name}/{p.name}", 0) for p in live_cues_dir.glob('*.pt')]
    spoof_cues = [(f"{spoof_cues_dir.name}/{p.name}", 1) for p in spoof_cues_dir.glob('*.pt')]
    
    all_samples = live_cues + spoof_cues
    random.shuffle(all_samples)
    
    train_count = int(len(all_samples) * 0.8)
    val_count = int(len(all_samples) * 0.1)
    
    splits = {
        'train': all_samples[:train_count],
        'val': all_samples[train_count : train_count + val_count],
        'test': all_samples[train_count + val_count :]
    }
    
    splits_file_path = Path(cfg.data.PROCESSED_DATA_DIR) / 'oanet_cues_splits.json'
    with open(splits_file_path, 'w') as f:
        json.dump(splits, f, indent=4)
    cfg.data.OANET_SPLITS_PATH = str(splits_file_path)

    logger.info(f"Đã chia dữ liệu và lưu vào {splits_file_path}")
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])} samples.")

    logger.info("="*50)
    logger.info("   STAGE 3: TRAINING OA-NET (Tối ưu hóa ĐA-GPU & AMP)")
    logger.info("="*50)
    
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    
    train_dataloader = get_oanet_dataloader('train', cfg.oanet.OANET_BATCH_SIZE)
    val_dataloader = get_oanet_dataloader('val', cfg.oanet.OANET_BATCH_SIZE, shuffle=False)

    model = OANet(device=device)

    model.to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Phát hiện {torch.cuda.device_count()} GPUs. Kích hoạt nn.DataParallel.")
        model = nn.DataParallel(model)
    
    optimizer = get_oanet_optimizer(model)
    criterion = oanet_loss_fn()
    lr_scheduler = get_oanet_lr_scheduler(optimizer)
    scaler = GradScaler()

    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = cfg.oanet.OANET_EARLY_STOPPING_PATIENCE
    num_epochs = cfg.oanet.OANET_NUM_EPOCHS
    EARLY_STOPPING_WARMUP_EPOCHS = 5

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        lr_scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            Path(cfg.oanet.OANET_CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
            model_save_path = os.path.join(cfg.oanet.OANET_CHECKPOINTS_DIR, "best_oanet_vit_model.pth")
            torch.save(model_to_save.state_dict(), model_save_path)
            logger.info(f"-> Val Loss cải thiện tới {best_val_loss:.4f}. Đã lưu model tốt nhất.")
            early_stopping_counter = 0
        else:
            if (epoch + 1) > EARLY_STOPPING_WARMUP_EPOCHS:
                early_stopping_counter += 1
                logger.info(f"-> Val Loss không cải thiện. Early Stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        if (epoch + 1) > EARLY_STOPPING_WARMUP_EPOCHS and early_stopping_counter >= early_stopping_patience:
            logger.info(f"!!! Early Stopping được kích hoạt sau {epoch + 1} epochs. Dừng huấn luyện.")
            break

    logger.info("OA-Net training complete.")
    print("\n--- PIPELINE FINISHED ---")