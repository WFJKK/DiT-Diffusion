"""
Defines the Diffuser module for applying forward diffusion (noise addition) in DDPM-style models.
"""

import torch
import torch.nn as nn 

class Diffuser(nn.Module):
    """
    Diffusion noise scheduler for adding noise to inputs over time.
    """
    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        self.register_buffer('betas', betas)
        alphas = 1. - betas
        self.register_buffer('alphas', alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.], dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], dim=0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def forward(self, x0, t):
        """
        Adds noise to clean input x0 at timestep t.

        Args:
            x0 (Tensor): Clean images of shape (B, C, H, W)
            t (Tensor): Timesteps of shape (B,) with values in [0, timesteps-1]

        Returns:
            x_t (Tensor): Noisy images at timestep t
            noise (Tensor): Noise added to x0
        """
        assert isinstance(x0, torch.Tensor), f"x0 must be a torch.Tensor, got {type(x0)}"
        assert isinstance(t, torch.Tensor), f"t must be a torch.Tensor, got {type(t)}"
        assert x0.dim() == 4, f"x0 must be 4D (B, C, H, W), got shape {x0.shape}"
        assert t.dim() == 1, f"t must be 1D tensor of shape (B,), got shape {t.shape}"
        assert t.shape[0] == x0.shape[0], f"Batch size mismatch: x0 batch {x0.shape[0]}, t batch {t.shape[0]}"
        assert t.max() < self.timesteps and t.min() >= 0, \
            f"t must be in [0, {self.timesteps - 1}], got range [{t.min()}, {t.max()}]"

        device = self.sqrt_alphas_cumprod.device
        x0 = x0.to(device)
        t = t.to(device)
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise


