import torch
import logging
from tqdm import tqdm
import os
import numpy as np
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.data_management.datasets import get_oanet_dataloader
from src.models.oanet.network import OANet
from src.training_utils.metrics import compute_classification_metrics, compute_fixed_threshold_anti_spoofing_metrics, compute_eer_and_related_metrics
from src.visualization.viz import plot_confusion_matrix, plot_roc_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(cfg.oanet.EVAL_RESULTS_FILE), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def evaluate_oanet():
    device = torch.device(cfg.DEVICE)
    logger.info(f"using device for oanet evaluation: {device}")

    test_dataloader = get_oanet_dataloader('test', cfg.oanet.OANET_BATCH_SIZE, shuffle=False)
    if not test_dataloader.dataset:
        logger.error("oanet test samples not found. run splitting script first.")
        return

    dfg_unet_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, "best_dfg_unet.pth")
    if not os.path.exists(dfg_unet_path):
        logger.error(f"best dfg unet checkpoint not found at {dfg_unet_path}. please train dfg first.")
        return

    oa_net_model_path = os.path.join(cfg.oanet.OANET_CHECKPOINTS_DIR, "best_oanet.pth")
    if not os.path.exists(oa_net_model_path):
        logger.error(f"best oanet model not found at {oa_net_model_path}. please train oanet first.")
        return

    model = OANet(dfg_unet_path, device=device)
    model.load_state_dict(torch.load(oa_net_model_path, map_location=device))
    model.eval()
    model.to(device)
    logger.info(f"loaded oanet model from {oa_net_model_path}")

    all_preds_binary = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(test_dataloader, desc="evaluating oanet"):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds_binary = probs.round().squeeze(1).long()

        all_preds_binary.extend(preds_binary.cpu().numpy())
        all_labels.extend(labels.squeeze(1).cpu().numpy())
        all_probs.extend(probs.squeeze(1).cpu().numpy())

    all_labels_np = np.array(all_labels)
    all_preds_binary_np = np.array(all_preds_binary)
    all_probs_np = np.array(all_probs)

    report_path = cfg.oanet.EVAL_RESULTS_FILE
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f_report:
        f_report.write("--- oanet evaluation report ---\n")
        f_report.write(f"oa net model path: {oa_net_model_path}\n")
        f_report.write(f"number of test samples: {len(test_dataloader.dataset)}\n\n")

        logger.info(f"\n--- metrics at default threshold (0.5) ---")
        f_report.write(f"--- metrics at default threshold (0.5) ---\n")
        classification_metrics = compute_classification_metrics(all_labels_np, all_preds_binary_np, all_probs_np)
        anti_spoofing_metrics_default_thresh = compute_fixed_threshold_anti_spoofing_metrics(all_labels_np, all_probs_np, threshold=cfg.oanet.OANET_EVAL_THRESHOLD)

        for metric, value in classification_metrics.items():
            logger.info(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")
            f_report.write(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}\n")
        for metric, value in anti_spoofing_metrics_default_thresh.items():
            logger.info(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")
            f_report.write(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}\n")

        logger.info(f"\n--- metrics at equal error rate (eer) threshold ---")
        f_report.write(f"\n--- metrics at equal error rate (eer) threshold ---\n")
        eer_metrics = compute_eer_and_related_metrics(all_labels_np, all_probs_np)
        for metric, value in eer_metrics.items():
            logger.info(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")
            f_report.write(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}\n")

        plot_confusion_matrix(all_labels_np, all_preds_binary_np, save_path=os.path.join(cfg.oanet.VISUALIZATIONS_DIR, "oanet_confusion_matrix.png"))
        plot_roc_curve(all_labels_np, all_probs_np, save_path=os.path.join(cfg.oanet.VISUALIZATIONS_DIR, "oanet_roc_curve.png"))
        logger.info("evaluation visualizations saved.")
        f_report.write(f"\nevaluation visualizations saved to {cfg.oanet.VISUALIZATIONS_DIR}\n")
        logger.info("oanet evaluation complete.")
        f_report.write("oa net evaluation complete.\n")

if __name__ == "__main__":
    evaluate_oanet()