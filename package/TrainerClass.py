"""Trainer class managing datasets, dataloaders, and loss calculation for diffusion Transformer models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DiffusionTransformerTrainer(nn.Module):
    """
    Trainer module for diffusion models using a DiffusionTransformer architecture.
    """
    def __init__(self, model, diffuser, train_dataset, val_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.diffuser = diffuser
        self.model = model 
        self.criterion = nn.MSELoss(reduction='mean')

    def get_dataloaders(self, batch_size):
        """
        Returns dataloaders for train, validation, and test datasets.

        Args:
            batch_size (int): Batch size for loading data.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, val, test dataloaders.
        """
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, val_dataloader

    def calc_loss(self, batch_data):
        """
        Computes MSE loss between predicted and true noise.

        Args:
            batch_data (Tensor): Input image batch of shape (B, C, H, W)

        Returns:
            Tensor: Scalar loss value.
        """
        B = batch_data.shape[0]
        diff_t = torch.randint(0, self.model.n_diffusion_steps, (B,))
        noised_image, noise = self.diffuser(batch_data, diff_t)
        noise_pred = self.model(noised_image, diff_t)
        loss = self.criterion(noise_pred, noise)
        return loss


        





