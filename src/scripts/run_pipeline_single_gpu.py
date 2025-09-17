import os
import sys
from pathlib import Path
import random
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.data_management.datasets import get_oanet_dataloader, get_oanet_eval_transforms, get_dfg_dataloader
from src.models.dfg.generator import DFGGenerator
from src.models.oanet.network import OANet
from src.training_utils.losses import dfg_loss_fn, oanet_loss_fn
from src.training_utils.optimizers import get_dfg_optimizer, get_oanet_optimizer
from src.training_utils.schedulers import get_dfg_lr_scheduler, get_oanet_lr_scheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ImageListDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path.name

def train_dfg():
    logger = logging.getLogger('DFG_TRAINING')
    logger.info("\n" + "="*50)
    logger.info("   STAGE 1: TRAINING DFG (DE-FAKE FACE GENERATOR)")
    logger.info("="*50)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    train_dataloader = get_dfg_dataloader('train', cfg.dfg.DFG_BATCH_SIZE)
    val_dataloader = get_dfg_dataloader('val', cfg.dfg.DFG_BATCH_SIZE, shuffle=False)
    model = DFGGenerator(device=device)
    optimizer = get_dfg_optimizer(model.unet.parameters())
    criterion = dfg_loss_fn()
    lr_scheduler = get_dfg_lr_scheduler(optimizer, num_training_steps=len(train_dataloader) * cfg.dfg.DFG_NUM_EPOCHS)
    model.to(device)
    best_val_loss = float('inf')
    logger.info("Bắt đầu huấn luyện DFG...")
    for epoch in range(cfg.dfg.DFG_NUM_EPOCHS):
        model.unet.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.dfg.DFG_NUM_EPOCHS} [DFG Train]")
        for pixel_values, _ in progress_bar:
            pixel_values = pixel_values.to(device)
            optimizer.zero_grad()
            noise_pred, noise = model(pixel_values)
            loss = criterion(noise_pred.float(), noise.float())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        avg_train_loss = total_train_loss / len(train_dataloader)
        model.unet.eval()
        total_val_loss = 0
        with torch.no_grad():
            for pixel_values, _ in val_dataloader:
                pixel_values = pixel_values.to(device)
                noise_pred, noise = model(pixel_values)
                loss = criterion(noise_pred.float(), noise.float())
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        logger.info(f"Epoch {epoch+1}: DFG Train Loss = {avg_train_loss:.4f}, DFG Val Loss = {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(cfg.dfg.DFG_CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
            model_save_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, "best_dfg_unet.pth")
            torch.save(model.unet.state_dict(), model_save_path)
            logger.info(f"Saved new best DFG model to {model_save_path} with val loss: {best_val_loss:.4f}")
    logger.info("Huấn luyện DFG hoàn tất.")
    return True

def generate_cues():
    logger = logging.getLogger('CUE_GENERATION')
    logger.info("\n" + "="*50)
    logger.info("   STAGE 2: GENERATING ANOMALOUS CUES")
    logger.info("="*50)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    dfg_unet_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, "best_dfg_unet.pth")
    if not os.path.exists(dfg_unet_path):
        logger.error(f"Lỗi: Không tìm thấy checkpoint DFG tại {dfg_unet_path}. Vui lòng chạy Giai đoạn 1 trước.")
        return False
    model = DFGGenerator(device=device)
    model.unet.load_state_dict(torch.load(dfg_unet_path, map_location='cpu'))
    model.eval().to(device)
    live_cues_dir = Path(cfg.data.ANOMALOUS_CUES_DIR) / 'celeba_spoof_train'
    live_cues_dir.mkdir(parents=True, exist_ok=True)
    num_live_cues = len(list(live_cues_dir.glob('*.pt')))
    if num_live_cues == 0:
        logger.warning(f"Thư mục live cues {live_cues_dir} đang trống. Pipeline sẽ tiếp tục nhưng bộ dữ liệu có thể không cân bằng.")

    logger.info("Bắt đầu sinh cues cho ảnh SPOOF...")
    spoof_images_root = Path(cfg.data.PROCESSED_DATA_DIR) / 'oanet_dataset' / 'spoof'
    ignore_dirs = {'train', 'val', 'test'}
    spoof_subfolders = [d for d in spoof_images_root.iterdir() if d.is_dir() and d.name not in ignore_dirs and not d.name.startswith('.')]
    all_spoof_images = []
    for folder in spoof_subfolders:
        all_spoof_images.extend(folder.glob('*.[jp][pn]g'))
    logger.info(f"Tìm thấy {len(all_spoof_images)} ảnh spoof để xử lý.")
    spoof_dataset = ImageListDataset(all_spoof_images, transform=get_oanet_eval_transforms())
    spoof_dataloader = DataLoader(spoof_dataset, batch_size=cfg.oanet.OANET_BATCH_SIZE, shuffle=False)
    spoof_cues_dir = Path(cfg.data.ANOMALOUS_CUES_DIR) / 'celeba_spoof_spoof'
    spoof_cues_dir.mkdir(parents=True, exist_ok=True)
    for images, filenames in tqdm(spoof_dataloader, desc="Generating Spoof Cues"):
        images = images.to(device)
        with torch.no_grad():
            _, cues = model.generate_reconstruction_and_cue(images)
        for i in range(cues.shape[0]):
            torch.save(cues[i].cpu(), spoof_cues_dir / f"cue_{Path(filenames[i]).stem}.pt")
    logger.info("Sinh cues hoàn tất.")
    return True

def split_and_train_oanet():
    logger = logging.getLogger('OA_NET_PIPELINE')
    logger.info("\n" + "="*50)
    logger.info("   STAGE 3: SPLITTING CUES DATASET")
    logger.info("="*50)
    live_cues_dir = Path(cfg.data.ANOMALOUS_CUES_DIR) / 'celeba_spoof_train'
    spoof_cues_dir = Path(cfg.data.ANOMALOUS_CUES_DIR) / 'celeba_spoof_spoof'
    if not live_cues_dir.is_dir() or not spoof_cues_dir.is_dir():
        logger.error("Lỗi: Không tìm thấy thư mục cues. Vui lòng chạy Giai đoạn 2 trước.")
        return
    live_cues = [(f"{live_cues_dir.name}/{p.name}", 0) for p in live_cues_dir.glob('*.pt')]
    spoof_cues = [(f"{spoof_cues_dir.name}/{p.name}", 1) for p in spoof_cues_dir.glob('*.pt')]
    if len(spoof_cues) > len(live_cues):
        spoof_cues = random.sample(spoof_cues, len(live_cues))
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
    logger.info("\n" + "="*50)
    logger.info("   STAGE 4: TRAINING OA-NET")
    logger.info("="*50)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    train_dataloader = get_oanet_dataloader('train', cfg.oanet.OANET_BATCH_SIZE)
    val_dataloader = get_oanet_dataloader('val', cfg.oanet.OANET_BATCH_SIZE, shuffle=False)
    dfg_unet_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, "best_dfg_unet.pth")
    model = OANet(dfg_unet_path, device=device).to(device)
    optimizer = get_oanet_optimizer(model)
    criterion = oanet_loss_fn()
    lr_scheduler = get_oanet_lr_scheduler(optimizer)
    best_val_loss = float('inf')
    for epoch in range(cfg.oanet.OANET_NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.oanet.OANET_NUM_EPOCHS} [OA-Net Train]")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        avg_train_loss = total_train_loss / len(train_dataloader)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [OA-Net Val]"):
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                logits = model(images)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        logger.info(f"Epoch {epoch+1}: OA-Net Train Loss = {avg_train_loss:.4f}, OA-Net Val Loss = {avg_val_loss:.4f}")
        lr_scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(cfg.oanet.OANET_CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
            model_save_path = os.path.join(cfg.oanet.OANET_CHECKPOINTS_DIR, "best_oanet_vit_model.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved new best OA-Net model to {model_save_path} with val loss: {best_val_loss:.4f}")
    logger.info("Huấn luyện OA-Net hoàn tất.")

if __name__ == "__main__":
    print("Bắt đầu chạy pipeline hoàn chỉnh...")
    try:
        train_dfg_success = True
        cues_generated_success = True
        if train_dfg_success:
            if cues_generated_success:
                split_and_train_oanet()
        print("\n--- PIPELINE HOÀN TẤT ---")
    except Exception as e:
        logging.error("Đã có lỗi nghiêm trọng xảy ra trong pipeline!", exc_info=True)
        print(f"\nLỖI: {e}")
