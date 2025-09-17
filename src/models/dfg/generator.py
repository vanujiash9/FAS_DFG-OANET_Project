import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.configs.config_loader import cfg
from src.models.dfg.ldm import VAE, UNetDenoiser, NoiseScheduler
from src.models.dfg.arcface import ArcfaceEncoder

class DFGGenerator(nn.Module):
    def __init__(self, device=cfg.DEVICE):
        super().__init__()
        self.device = device

        self.vae = VAE(device=device)
        self.unet = UNetDenoiser(device=device)
        self.identity_encoder = ArcfaceEncoder(device=device)

        self.noise_scheduler = NoiseScheduler()

    def forward(self, pixel_values):
        latent_dist = self.vae.encode(pixel_values).latent_dist
        z_0 = latent_dist.sample() * self.vae.scaling_factor

        with torch.no_grad():
            identity_embedding = self.identity_encoder(pixel_values) 

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (z_0.shape[0],), device=self.device).long()
        noise = torch.randn_like(z_0)
        z_t = self.noise_scheduler.add_noise(z_0, noise, timesteps)
        
        noise_pred = self.unet(z_t, timesteps, encoder_hidden_states=identity_embedding).sample
        
        return noise_pred, noise

    @torch.no_grad()
    def generate_reconstruction_and_cue(self, x_input):
        self.eval()
        
        x_input_norm = x_input * 2.0 - 1.0 
        x_input_norm = x_input_norm.to(self.device)

        latent_dist = self.vae.encode(x_input_norm).latent_dist
        z_0 = latent_dist.sample() * self.vae.scaling_factor

        identity_embedding = self.identity_encoder(x_input_norm)

        initial_timesteps = torch.tensor([cfg.dfg.DFG_T_HAT_CUE_GENERATION - 1], device=self.device).repeat(x_input.shape[0])
        z_t_hat = self.noise_scheduler.add_noise(z_0, torch.randn_like(z_0), initial_timesteps)
        
        self.noise_scheduler.set_timesteps(cfg.dfg.DFG_T_HAT_CUE_GENERATION, device=self.device)
        
        current_latent = z_t_hat
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(current_latent, t, encoder_hidden_states=identity_embedding).sample
            current_latent = self.noise_scheduler.step(noise_pred, t, current_latent).prev_sample

        z_prime_0 = current_latent

        x_prime_0 = self.vae.decode(z_prime_0 / self.vae.scaling_factor).sample
        
        reconstructed_images = (x_prime_0 / 2 + 0.5).clamp(0, 1)

        anomalous_cues = torch.abs(x_input - reconstructed_images)
        
        return reconstructed_images, anomalous_cues