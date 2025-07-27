"""
Modules to convert images to/from patch embeddings for Transformer models via patchification.
"""

import torch
import torch.nn as nn 

class Patchify(nn.Module):
    """
    Converts an image tensor (B, C, H, W) into a sequence of patch embeddings (B, N_patches, embed_dim).
    """
    def __init__(self, patch_size_H, patch_size_W, input_dim, embed_dim): 
        super().__init__()
        self.patch_size_H = patch_size_H
        self.patch_size_W = patch_size_W
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.proj = nn.Linear(input_dim * patch_size_W * patch_size_H, embed_dim)

    def forward(self, x):
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        assert x.dim() == 4, f"Expected input to be 4D (B, C, H, W), got {x.dim()}D"
        x = x.to(self.proj.weight.device)
        B, C, H, W = x.shape
        assert C == self.input_dim, f"Expected input channels {self.input_dim}, got {C}"
        assert H % self.patch_size_H == 0 and W % self.patch_size_W == 0, \
            f"Image size ({H}, {W}) must be divisible by patch size ({self.patch_size_H}, {self.patch_size_W})"
        n_H = H // self.patch_size_H
        n_W = W // self.patch_size_W
        x = x.reshape(B, C, self.patch_size_H, n_H, self.patch_size_W, n_W)
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(B, n_H * n_W, C * self.patch_size_H * self.patch_size_W)
        out = self.proj(x)
        return out  


class Unpatchify(nn.Module): 
    """
    Reconstructs an image tensor (B, C, H, W) from patch embeddings (B, N_patches, embed_dim).
    """
    def __init__(self, patch_size_H, patch_size_W, input_dim, embed_dim, img_H, img_W):
        super().__init__()
        self.patch_size_H = patch_size_H
        self.patch_size_W = patch_size_W
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.img_H = img_H
        self.img_W = img_W
        self.n_H = img_H // patch_size_H
        self.n_W = img_W // patch_size_W
        self.n_patches = self.n_H * self.n_W
        self.proj = nn.Linear(embed_dim, input_dim * patch_size_H * patch_size_W)

    def forward(self, x):
        """
        x: (B, N_patches, embed_dim) -> returns: (B, C, H, W)
        """
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        assert x.dim() == 3, f"Expected input to be 3D (B, N_patches, embed_dim), got {x.dim()}D"
        assert x.shape[2] == self.embed_dim, \
            f"Expected embedding dim {self.embed_dim}, got {x.shape[2]}"
        x = x.to(self.proj.weight.device)
        B, N_patches, _ = x.shape
        assert N_patches == self.n_patches, \
            f"Expected {self.n_patches} patches, got {N_patches}"
        x = self.proj(x) 
        x = x.view(B, self.n_H, self.n_W, self.input_dim, self.patch_size_H, self.patch_size_W)
        x = x.permute(0, 3, 1, 4, 2, 5)  
        x = x.reshape(B, self.input_dim, self.img_H, self.img_W)
        return x









