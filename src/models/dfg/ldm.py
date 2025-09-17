import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.configs.config_loader import cfg

class VAE(AutoencoderKL):
    def __init__(self, device=cfg.DEVICE):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
        self.model.requires_grad_(False)
        self.scaling_factor = self.model.config.scaling_factor

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

class UNetDenoiser(UNet2DConditionModel):
    def __init__(self, device=cfg.DEVICE):
        super().__init__()
        self.model = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)

    def forward(self, latents, timesteps, encoder_hidden_states):
        return self.model(latents, timesteps, encoder_hidden_states=encoder_hidden_states)

class NoiseScheduler(DDPMScheduler):
    def __init__(self):
        super().__init__(
            num_train_timesteps=cfg.dfg.DFG_MAX_DIFFUSION_STEPS,
            beta_schedule="linear",
            prediction_type="epsilon"
        )