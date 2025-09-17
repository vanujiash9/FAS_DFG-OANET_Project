# File: src/training_utils/schedulers.py
# PHIÊN BẢN ĐÃ SỬA LỖI

# SỬA LỖI: Thêm dòng import quan trọng này
import torch.optim as optim
from src.configs.config_loader import cfg

def get_oanet_lr_scheduler(optimizer):
    """Scheduler để giảm learning rate khi Val Loss không cải thiện."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.oanet.OANET_LR_SCHEDULER_FACTOR,
        patience=cfg.oanet.OANET_LR_SCHEDULER_PATIENCE,
        verbose=True
    )

def get_dfg_lr_scheduler(optimizer, num_training_steps):
    """Scheduler cho DFG (ví dụ: cosine)."""
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps,
    )