from transformers import ViTModel
import torch.nn as nn

class ViTBackbone(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)
    
    def forward(self, pixel_values, output_hidden_states=False):
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=output_hidden_states)
        return outputs