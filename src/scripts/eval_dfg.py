import torch
import logging
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.data_management.datasets import OANetDataset, get_oanet_eval_transforms, DataLoader
from src.models.dfg.generator import DFGGenerator
from src.training_utils.metrics import calculate_psnr_ssim
from src.visualization.viz import save_dfg_cue_visualization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(cfg.dfg.DFG_EVAL_LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def evaluate_dfg():
    device = torch.device(cfg.DEVICE)
    logger.info(f"using device for dfg evaluation: {device}")

    dfg_unet_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, "best_dfg_unet.pth")
    if not os.path.exists(dfg_unet_path):
        logger.error(f"best dfg unet checkpoint not found at {dfg_unet_path}. please train dfg first.")
        return

    dfg_model = DFGGenerator(device=device)
    dfg_model.unet.load_state_dict(torch.load(dfg_unet_path, map_location='cpu'))
    dfg_model.eval()
    dfg_model.requires_grad_(False)
    logger.info(f"loaded dfg unet from {dfg_unet_path}")

    eval_dataloader = get_oanet_dataloader('test', cfg.dfg.DFG_BATCH_SIZE, shuffle=False)

    total_psnr = 0.0
    total_ssim = 0.0
    real_count_for_metrics = 0

    viz_real_count = 0
    viz_spoof_count = 0

    logger.info("***** running dfg evaluation *****")
    report_path = cfg.dfg.DFG_EVAL_REPORT_FILE
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f_report:
        f_report.write("--- dfg evaluation report ---\n")
        f_report.write(f"dfg model path: {dfg_unet_path}\n")
        f_report.write(f"number of test samples: {len(eval_dataloader.dataset)}\n\n")
        f_report.write("quantitative metrics (psnr/ssim calculated on real faces only):\n")

        for batch_idx, (images, labels) in enumerate(tqdm(eval_dataloader, desc="evaluating dfg")):
            images = images.to(device)

            reconstructed_images, cues = dfg_model.generate_reconstruction_and_cue(images)

            for i in range(images.shape[0]):
                if labels[i].item() == 0:
                    original_np = images[i].cpu().permute(1, 2, 0).numpy()
                    reconstructed_np = reconstructed_images[i].cpu().permute(1, 2, 0).numpy()
                    
                    psnr, ssim = calculate_psnr_ssim(original_np, reconstructed_np)
                    total_psnr += psnr
                    total_ssim += ssim
                    real_count_for_metrics += 1
            
            if (viz_real_count < cfg.dfg.DFG_EVAL_SAMPLES_TO_VIZ or viz_spoof_count < cfg.dfg.DFG_EVAL_SAMPLES_TO_VIZ) and \
               cfg.dfg.DFG_EVAL_SAMPLES_TO_VIZ > 0:
                for i in range(images.shape[0]):
                    if labels[i].item() == 0 and viz_real_count < cfg.dfg.DFG_EVAL_SAMPLES_TO_VIZ:
                        save_dfg_cue_visualization(
                            images[i:i+1], reconstructed_images[i:i+1], cues[i:i+1], 
                            f"dfg_eval_real_sample_{viz_real_count+1:03d}", 0, num_samples=1
                        )
                        viz_real_count += 1
                    elif labels[i].item() == 1 and viz_spoof_count < cfg.dfg.DFG_EVAL_SAMPLES_TO_VIZ:
                        save_dfg_cue_visualization(
                            images[i:i+1], reconstructed_images[i:i+1], cues[i:i+1], 
                            f"dfg_eval_spoof_sample_{viz_spoof_count+1:03d}", 0, num_samples=1
                        )
                        viz_spoof_count += 1
            
            if viz_real_count >= cfg.dfg.DFG_EVAL_SAMPLES_TO_VIZ and viz_spoof_count >= cfg.dfg.DFG_EVAL_SAMPLES_TO_VIZ:
                break

        avg_psnr = total_psnr / real_count_for_metrics if real_count_for_metrics > 0 else 0.0
        avg_ssim = total_ssim / real_count_for_metrics if real_count_for_metrics > 0 else 0.0

        f_report.write(f"  average psnr on real faces: {avg_psnr:.4f}\n")
        f_report.write(f"  average ssim on real faces: {avg_ssim:.4f}\n")
        
        f_report.write("\nqualitative evaluation (visualizations):\n")
        f_report.write(f"  saved {viz_real_count} real face visualizations to {cfg.dfg.DFG_CUES_VIZ_DIR}\n")
        f_report.write(f"  saved {viz_spoof_count} spoof face visualizations to {cfg.dfg.DFG_CUES_VIZ_DIR}\n")
        f_report.write("  - for real faces, the anomalous cue (heatmap) should appear dark.\n")
        f_report.write("  - for spoof faces, the anomalous cue should highlight spoofing artifacts (appear bright).\n")

        logger.info(f"\n--- dfg evaluation results ---")
        logger.info(f"average psnr on real faces: {avg_psnr:.4f}")
        logger.info(f"average ssim on real faces: {avg_ssim:.4f}")
        logger.info(f"dfg evaluation visualizations saved to {cfg.dfg.DFG_CUES_VIZ_DIR}")
        logger.info(f"dfg evaluation report saved to {report_path}")

    logger.info("dfg evaluation complete.")

if __name__ == "__main__":
    evaluate_dfg()