import torch.optim as optim
import torch.nn as nn
from src.configs.config_loader import cfg

def get_dfg_optimizer(params):
    return optim.Adam(
        params,
        lr=cfg.dfg.DFG_LEARNING_RATE,
        betas=(cfg.dfg.DFG_BETA1, cfg.dfg.DFG_BETA2)
    )

def get_oanet_optimizer(model):
    original_model = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = [
        {
            'params': original_model.vit_backbone.parameters(),
            'lr': cfg.oanet.OANET_LEARNING_RATE * cfg.oanet.OANET_BACKBONE_LR_FACTOR
        },
        {
            'params': original_model.cue_encoder.parameters()
        },
        {
            'params': original_model.cross_attention_modules.parameters()
        },
        {
            'params': original_model.classifier.parameters()
        }
    ]
    return optim.Adam(
        param_groups,
        lr=cfg.oanet.OANET_LEARNING_RATE,
        betas=(cfg.oanet.OANET_BETAS[0], cfg.oanet.OANET_BETAS[1])
    )
