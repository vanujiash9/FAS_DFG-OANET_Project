import torch
import torch.nn as nn
from collections import OrderedDict
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.configs.config_loader import cfg

class ArcfaceEncoder(nn.Module):
    def __init__(self, model_path=cfg.dfg.ARCFACE_PRETRAINED_PATH, device=cfg.DEVICE):
        super().__init__()
        self.device = device
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512),
            nn.AdaptiveAvgPool2d((7, 7))
        ).to(device)

        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[9:] if k.startswith('backbone.') else (k[7:] if k.startswith('module.') else k)
                    new_state_dict[name] = v
                self.backbone.load_state_dict(new_state_dict, strict=False)
                print(f"arcface weights loaded from {model_path}. (strict=false)")
            else:
                print(f"arcface weights not found at {model_path}. using random initialization.")
        except Exception as e:
            print(f"error loading arcface model: {e}. using random initialization.")
            
        self.backbone.eval()
        self.projection_layer = nn.Linear(512, 768).to(device)

    def forward(self, x):
        features_map = self.backbone(x)
        batch_size = features_map.shape[0]
        features_reshaped = features_map.permute(0, 2, 3, 1).reshape(batch_size, 49, 512)
        identity_embedding = self.projection_layer(features_reshaped)
        return identity_embedding