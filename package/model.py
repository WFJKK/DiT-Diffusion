"""
Transformer-based diffusion model components with adaptive layer normalization and patch-based image processing.
"""


import torch
import torch.nn as nn
from .patchification import Patchify, Unpatchify


class AdaLNZeroLayer(nn.Module):
    """
    Adaptive layer normalization with zero-initialized linear projection.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.lin = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)

    def forward(self, conditioning):
        """
        Args:
            conditioning (Tensor): Input tensor of shape (B, embed_dim)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: alpha, mu, sigma of shape (B, embed_dim)
        """
        assert isinstance(conditioning, torch.Tensor), f"Expected torch.Tensor, got {type(conditioning)}"
        assert conditioning.dim() == 2, f"Expected shape (B, D), got {conditioning.shape}"
        conditioning = conditioning.to(self.lin.weight.device)
        return self.lin(self.silu(conditioning)).chunk(3, dim=1)


class TimeModulator(nn.Module):
    """
    Modulates features with time-conditioned AdaLN output.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.ada = AdaLNZeroLayer(embed_dim)

    def forward(self, x, conditioning):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, embed_dim)
            conditioning (Tensor): Time embedding of shape (B, embed_dim)

        Returns:
            Tensor: Modulated tensor of shape (B, N, embed_dim)
        """
        assert isinstance(x, torch.Tensor), f"x must be a torch.Tensor, got {type(x)}"
        assert isinstance(conditioning, torch.Tensor), f"conditioning must be a torch.Tensor, got {type(conditioning)}"
        assert x.dim() == 3, f"x must be 3D (B, N, D), got {x.shape}"
        assert conditioning.shape[0] == x.shape[0], "Batch size mismatch"
        conditioning = conditioning.to(x.device)
        a, mu, s = self.ada(conditioning)
        out = x * (1 + s[:, None, :]) + mu[:, None, :]
        return x + a[:, None, :] * out


class TransformerBlock(nn.Module):
    """
    Transformer block with time-modulated attention and MLP.
    """
    def __init__(self, embed_dim, hidden_dim, num_heads, mha_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=mha_dropout)
        self.time_modulator1 = TimeModulator(embed_dim)
        self.time_modulator2 = TimeModulator(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x, t):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, embed_dim)
            t (Tensor): Time embedding of shape (B, embed_dim)

        Returns:
            Tensor: Output tensor of shape (B, N, embed_dim)
        """
        assert x.dim() == 3, f"Expected x to be (B, N, D), got {x.shape}"
        assert t.shape == (x.shape[0], x.shape[2]), f"Expected t to be (B, D), got {t.shape}"

        x_norm1 = self.norm1(x)
        att, _ = self.mha(x_norm1, x_norm1, x_norm1, is_causal=False)
        att_mod = self.time_modulator1(att, t)
        x = x + att_mod

        y = self.mlp(self.norm2(x))
        y_mod = self.time_modulator2(y, t)
        return x + y_mod


class DiffusionTransformer(nn.Module):
    """
    Transformer-based architecture for diffusion models with patchified image input.
    """
    def __init__(self, n_diffusion_steps, input_dim, img_H, img_W,
                 embed_dim, hidden_dim, patch_size_H, patch_size_W,
                 num_heads, mha_dropout, num_layers, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size_H = patch_size_H
        self.patch_size_W = patch_size_W
        self.n_diffusion_steps = n_diffusion_steps

        self.time_embed = nn.Embedding(n_diffusion_steps, embed_dim)
        self.n_patches = (img_H // patch_size_H) * (img_W // patch_size_W)

        self.patchify = Patchify(patch_size_H, patch_size_W, input_dim, embed_dim)
        self.unpatchify = Unpatchify(patch_size_H, patch_size_W, input_dim, embed_dim, img_H, img_W)

        self.pos_embed = nn.Embedding(self.n_patches, embed_dim)
        self.pos_dropout = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, hidden_dim, num_heads, mha_dropout)
            for _ in range(num_layers)
        ])
        self.last_ada = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.last_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x, t):
        """
        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)
            t (Tensor): Diffusion timestep tensor of shape (B,)

        Returns:
            Tensor: Predicted noise tensor of shape (B, C, H, W)
        """
        assert isinstance(x, torch.Tensor) and x.dim() == 4, f"x must be (B, C, H, W), got {x.shape}"
        assert isinstance(t, torch.Tensor) and t.dim() == 1, f"t must be (B,), got {t.shape}"
        assert t.shape[0] == x.shape[0], f"Batch size mismatch: x {x.shape[0]}, t {t.shape[0]}"
        assert t.max() < self.time_embed.num_embeddings and t.min() >= 0, \
            f"t values must be in [0, {self.time_embed.num_embeddings - 1}]"

        device = self.time_embed.weight.device
        x = x.to(device)
        t = t.to(device)

        time_embed = self.time_embed(t)
        x = self.patchify(x)
        B, N, _ = x.shape

        pos = self.pos_embed(torch.arange(N, device=device))
        pos = pos.unsqueeze(0).expand(B, -1, -1)
        x = x + self.pos_dropout(pos)

        for block in self.blocks:
            x = block(x, time_embed)

        x_norm = self.last_norm(x)
        mu_last, sigma_last = self.last_ada(time_embed).chunk(2, dim=-1)
        x = x_norm * (1 + sigma_last[:, None, :]) + mu_last[:, None, :]
        noise = self.unpatchify(x)
        return noise


