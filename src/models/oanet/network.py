import torch
import torch.nn as nn
from torchvision import transforms
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.configs.config_loader import cfg
from src.models.oanet.vit import ViTBackbone
from src.models.oanet.resnet_cue import ResNetCueEncoder
from src.models.oanet.cross_attn import OffRealAttention
from src.models.oanet.classifier import OANetClassifier

class OANet(nn.Module):
    def __init__(self, device=cfg.DEVICE):
        super().__init__()
        self.device = device
        self.vit_backbone = ViTBackbone().to(device)
        self.cue_encoder = ResNetCueEncoder(
            target_embed_dim=self.vit_backbone.model.config.hidden_size
        ).to(device)
        self.cross_attention_modules = nn.ModuleList([
            OffRealAttention(embed_dim=self.vit_backbone.model.config.hidden_size, 
                             num_heads=self.vit_backbone.model.config.num_attention_heads)
            for _ in range(self.vit_backbone.model.config.num_hidden_layers)
        ]).to(device)
        self.classifier = OANetClassifier(self.vit_backbone.model.config.hidden_size).to(device)
        self.vit_normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def forward(self, cue_input):
        cue_norm = self.vit_normalize(cue_input)
        vit_outputs = self.vit_backbone(cue_norm, output_hidden_states=True)
        cue_features = self.cue_encoder(cue_norm)
        
        B, C, H, W = cue_features.shape
        cue_tokens_reshaped = cue_features.view(B, C, H * W)
        cue_tokens = cue_tokens_reshaped.permute(0, 2, 1)

        hidden_states = vit_outputs.hidden_states[0]
        for i, layer_module in enumerate(self.vit_backbone.model.encoder.layer):
            layer_output = layer_module(hidden_states)[0]
            cross_attn_output = self.cross_attention_modules[i](
                query_features=layer_output, 
                key_value_features=cue_tokens
            )
            hidden_states = layer_output + cross_attn_output 

        cls_token_output = hidden_states[:, 0, :]
        logits = self.classifier(cls_token_output)
        return logits