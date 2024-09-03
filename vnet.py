# src/vnet.py

import torch
import torch.nn as nn

class VNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=14):
        super(VNet, self).__init__()
        # Define your VNet architecture here
        # Example:
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        # Add more layers as needed
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2
