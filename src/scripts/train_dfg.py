import torch
import logging
import accelerate
from tqdm import tqdm
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.data_management.datasets import get_dfg_dataloader
from src.models.dfg.generator import DFGGenerator
from src.training_utils.losses import dfg_loss_fn
from src.training_utils.optimizers import get_dfg_optimizer
from src.training_utils.schedulers import get_dfg_lr_scheduler
from src.visualization.viz import plot_dfg_training_loss, save_dfg_cue_visualization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(cfg.dfg.DFG_LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def train_dfg():
    accelerator = accelerate.Accelerator(mixed_precision="fp16")
    device = accelerator.device
    logger.info(f"using device: {device}")

    train_dataloader = get_dfg_dataloader(cfg.data.DFG_REAL_FACES_TRAIN_DIR, cfg.dfg.DFG_BATCH_SIZE)
    if not train_dataloader.dataset:
        logger.error("no dfg live images found. run preprocessing script.")
        return

    model = DFGGenerator(device=device)
    optimizer = get_dfg_optimizer(model.unet.parameters())
    criterion = dfg_loss_fn()

    model.unet, optimizer, train_dataloader = accelerator.prepare(
        model.unet, optimizer, train_dataloader
    )
    model.vae.to(device)
    model.identity_encoder.to(device)

    num_training_steps = len(train_dataloader) * cfg.dfg.DFG_NUM_EPOCHS
    lr_scheduler = get_dfg_lr_scheduler(optimizer, num_training_steps)

    logger.info("***** running dfg training *****")
    logger.info(f"  num epochs = {cfg.dfg.DFG_NUM_EPOCHS}")
    logger.info(f"  batch size per device = {cfg.dfg.DFG_BATCH_SIZE}")

    losses = []

    for epoch in range(cfg.dfg.DFG_NUM_EPOCHS):
        model.unet.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.dfg.DFG_NUM_EPOCHS}", disable=not accelerator.is_local_main_process)

        for step, (pixel_values, _) in enumerate(progress_bar):
            pixel_values = pixel_values.to(device)

            with accelerator.accumulate(model.unet):
                noise_pred, noise = model(pixel_values)
                loss = criterion(noise_pred.float(), noise.float())
                total_loss += loss.item()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        losses.append(avg_loss)
        logger.info(f"epoch {epoch+1} average loss: {avg_loss:.4f}")

        accelerator.wait_for_everyone()
        unwrapped_unet = accelerator.unwrap_model(model.unet)
        unet_save_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, f"dfg_unet_epoch_{epoch+1:03d}.pth")
        torch.save(unwrapped_unet.state_dict(), unet_save_path)
        logger.info(f"saved dfg unet model to {unet_save_path}")

        if accelerator.is_local_main_process and (epoch + 1) % 5 == 0:
            logger.info("generating and visualizing dfg cues from training data...")
            model.eval()
            sample_batch = next(iter(train_dataloader))[0].to(device)
            original_images = (sample_batch / 2 + 0.5).clamp(0, 1)

            reconstructed_images, cues = model.generate_reconstruction_and_cue(original_images)
            save_dfg_cue_visualization(original_images, reconstructed_images, cues, f"dfg_train_epoch_{epoch+1:03d}", 0) # batch_idx=0
            model.train()
            logger.info("dfg cues visualization saved.")

    if accelerator.is_local_main_process:
        plot_dfg_training_loss(losses)
    logger.info("dfg training complete.")

if __name__ == "__main__":
    train_dfg()