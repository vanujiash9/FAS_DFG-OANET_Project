import torch
import os
import logging
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.data_management.datasets import OANetDataset, get_oanet_eval_transforms, DataLoader
from src.models.dfg.generator import DFGGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_cues_for_split(dfg_model, dataloader, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dfg_model.eval()
    
    for i, (images, labels) in enumerate(tqdm(dataloader, desc=f"generating cues for {output_dir.split('/')[-1]}")):
        images = images.to(dfg_model.device)
        
        with torch.no_grad():
            _, cues = dfg_model.generate_reconstruction_and_cue(images)
        
        for j in range(images.shape[0]):
            original_path = dataloader.dataset.samples[i * dataloader.batch_size + j][0]
            filename = os.path.basename(original_path)
            cue_filename = f"cue_{os.path.splitext(filename)[0]}.pt"
            cue_path = os.path.join(output_dir, cue_filename)
            torch.save(cues[j].cpu(), cue_path)

def generate_anomalous_cues():
    device = torch.device(cfg.DEVICE)
    logger.info(f"using device: {device}")

    dfg_unet_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, f"dfg_unet_epoch_{cfg.dfg.DFG_NUM_EPOCHS:03d}.pth")
    if not os.path.exists(dfg_unet_path):
        logger.error(f"dfg unet checkpoint not found at {dfg_unet_path}. please train dfg first.")
        return

    dfg_model = DFGGenerator(device=device)
    dfg_model.unet.load_state_dict(torch.load(dfg_unet_path, map_location='cpu'))
    dfg_model.eval()
    dfg_model.requires_grad_(False)
    logger.info(f"loaded dfg unet from {dfg_unet_path}")

    transform = get_oanet_eval_transforms()

    for split_type in ['train', 'val', 'test']:
        dataset = OANetDataset(split_type=split_type, transform=transform)
        dataloader = DataLoader(dataset, batch_size=cfg.oanet.OANET_BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2)
        
        output_dir = os.path.join(cfg.data.ANOMALOUS_CUES_DIR, f"celeba_spoof_{split_type}")
        generate_cues_for_split(dfg_model, dataloader, output_dir)
        logger.info(f"cues generated and saved for {split_type} split in {output_dir}")

    logger.info("anomalous cue generation complete.")

if __name__ == "__main__":
    generate_anomalous_cues()