import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetCueEncoder(nn.Module):
    def __init__(self, target_embed_dim=768):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)
        
        original_embed_dim = 512
        self.projection = nn.Conv2d(original_embed_dim, target_embed_dim, kernel_size=1)

    def forward(self, x):
        features = self.feature_extractor(x)
        projected_features = self.projection(features)
        return projected_features