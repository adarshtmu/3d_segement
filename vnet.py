import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownSampling(nn.Module):
    def __init__(self, in_channels):
        super(DownSampling, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)



class VNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VNet, self).__init__()

        # Downsampling path
        self.down_1 = nn.Sequential(ConvBlock(in_channels, 16), ConvBlock(16, 16))
        self.down_2 = nn.Sequential(DownSampling(16), ConvBlock(16, 32), ConvBlock(32, 32))
        self.down_3 = nn.Sequential(DownSampling(32), ConvBlock(32, 64), ConvBlock(64, 64), ConvBlock(64, 64))
        self.down_4 = nn.Sequential(DownSampling(64), ConvBlock(64, 128), ConvBlock(128, 128), ConvBlock(128, 128))

        # Bridge
        self.bridge = nn.Sequential(DownSampling(128), ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256))

        # Upsampling path
        self.up_4 = nn.Sequential(UpSampling(256, 128), ConvBlock(128, 128), ConvBlock(128, 128))
        self.up_3 = nn.Sequential(UpSampling(128, 64), ConvBlock(64, 64), ConvBlock(64, 64))
        self.up_2 = nn.Sequential(UpSampling(64, 32), ConvBlock(32, 32), ConvBlock(32, 32))
        self.up_1 = nn.Sequential(UpSampling(32, 16), ConvBlock(16, 16), ConvBlock(16, 16))

        # Final output
        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        # Downsample
        down_1 = self.down_1(x)
        print(f"Down 1 shape: {down_1.shape}")
        down_2 = self.down_2(down_1)
        print(f"Down 2 shape: {down_2.shape}")
        down_3 = self.down_3(down_2)
        print(f"Down 3 shape: {down_3.shape}")
        down_4 = self.down_4(down_3)
        print(f"Down 4 shape: {down_4.shape}")

        # Bridge
        bridge = self.bridge(down_4)
        print(f"Bridge shape: {bridge.shape}")

        # Upsample with skip connections
        up_4 = self.up_4(bridge)
        print(f"Up 4 shape before addition: {up_4.shape}")
        up_4 = self.crop_and_add(up_4, down_4)
        print(f"Up 4 shape after addition: {up_4.shape}")

        up_3 = self.up_3(up_4)
        print(f"Up 3 shape before addition: {up_3.shape}")
        up_3 = self.crop_and_add(up_3, down_3)
        print(f"Up 3 shape after addition: {up_3.shape}")

        up_2 = self.up_2(up_3)
        print(f"Up 2 shape before addition: {up_2.shape}")
        up_2 = self.crop_and_add(up_2, down_2)
        print(f"Up 2 shape after addition: {up_2.shape}")

        up_1 = self.up_1(up_2)
        print(f"Up 1 shape before addition: {up_1.shape}")
        up_1 = self.crop_and_add(up_1, down_1)
        print(f"Up 1 shape after addition: {up_1.shape}")

        # Final output
        output = self.final(up_1)
        print(f"Final output shape: {output.shape}")
        return output

    def crop_and_add(self, upsampled, bypass):
        if upsampled.shape[2:] != bypass.shape[2:]:
            upsampled = self.center_crop(upsampled, bypass.shape[2:])
        return upsampled + bypass

    def center_crop(self, tensor, target_shape):
        _, _, d, h, w = tensor.shape
        target_d, target_h, target_w = target_shape

        d1 = (d - target_d) // 2
        h1 = (h - target_h) // 2
        w1 = (w - target_w) // 2

        return tensor[:, :, d1:d1 + target_d, h1:h1 + target_h, w1:w1 + target_w]
