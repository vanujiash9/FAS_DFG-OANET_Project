import torch
import logging
import accelerate
from tqdm import tqdm
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.data_management.datasets import get_oanet_dataloader
from src.models.oanet.network import OANet
from src.training_utils.losses import oanet_loss_fn
from src.training_utils.optimizers import get_oanet_optimizer
from src.training_utils.schedulers import get_oanet_lr_scheduler
from src.visualization.viz import plot_oanet_train_val_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(cfg.oanet.OANET_LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def train_oanet():
    accelerator = accelerate.Accelerator(mixed_precision="fp16")
    device = accelerator.device
    logger.info(f"using device: {device}")

    train_dataloader = get_oanet_dataloader(split_type='train', batch_size=cfg.oanet.OANET_BATCH_SIZE)
    val_dataloader = get_oanet_dataloader(split_type='val', batch_size=cfg.oanet.OANET_BATCH_SIZE, shuffle=False)

    if not train_dataloader.dataset or not val_dataloader.dataset:
        logger.error("oanet train/val samples not found. run preprocessing script.")
        return

    dfg_unet_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, f"dfg_unet_epoch_{cfg.dfg.DFG_NUM_EPOCHS:03d}.pth")
    if not os.path.exists(dfg_unet_path):
        logger.error(f"dfg unet checkpoint not found at {dfg_unet_path}. please train dfg first.")
        return

    model = OANet(dfg_unet_path, device=device)
    optimizer = get_oanet_optimizer(model)
    criterion = oanet_loss_fn()

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    lr_scheduler = get_oanet_lr_scheduler(optimizer)

    logger.info("***** running oanet training *****")
    logger.info(f"  num epochs = {cfg.oanet.OANET_NUM_EPOCHS}")
    logger.info(f"  batch size per device = {cfg.oanet.OANET_BATCH_SIZE}")

    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(cfg.oanet.OANET_NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"epoch {epoch+1}/{cfg.oanet.OANET_NUM_EPOCHS} train", disable=not accelerator.is_local_main_process)
        
        for images, labels in train_progress_bar:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            with accelerator.accumulate(model):
                logits = model(images)
                loss = criterion(logits, labels)
                total_train_loss += loss.item()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            train_progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"epoch {epoch+1}/{cfg.oanet.OANET_NUM_EPOCHS} val", disable=not accelerator.is_local_main_process)

        with torch.no_grad():
            for images, labels in val_progress_bar:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                logits = model(images)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        logger.info(f"epoch {epoch+1}: train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}")
        lr_scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            model_save_path = os.path.join(cfg.oanet.OANET_CHECKPOINTS_DIR, "best_oanet.pth")
            torch.save(unwrapped_model.state_dict(), model_save_path)
            logger.info(f"saved best oanet model to {model_save_path} with val loss: {best_val_loss:.4f}")
        else:
            early_stopping_counter += 1
            logger.info(f"early stopping counter: {early_stopping_counter}/{cfg.oanet.OANET_EARLY_STOPPING_PATIENCE}")
            if early_stopping_counter >= cfg.oanet.OANET_EARLY_STOPPING_PATIENCE:
                logger.info("early stopping triggered.")
                break
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        epoch_save_path = os.path.join(cfg.oanet.OANET_CHECKPOINTS_DIR, f"oanet_epoch_{epoch+1:03d}.pth")
        torch.save(unwrapped_model.state_dict(), epoch_save_path)
        logger.info(f"saved oanet model to {epoch_save_path}")

    if accelerator.is_local_main_process:
        plot_oanet_train_val_loss(train_losses, val_losses)
    logger.info("oanet training complete.")

if __name__ == "__main__":
    train_oanet()