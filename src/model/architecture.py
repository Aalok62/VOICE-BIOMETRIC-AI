"""
src/model/architecture.py
=========================
Custom CNN architecture for audio classification from Mel-spectrograms.

Input : (batch, 1, n_mels, n_frames) Mel-spectrogram tensor
Output: (batch, n_classes) log-softmax probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Reusable Conv2D -> BatchNorm -> ReLU -> MaxPool block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class BioSoundCNN(nn.Module):
    """
    Lightweight CNN for living-being audio classification.

    Architecture
    ------------
    Input (1 x 128 x 128)
    ConvBlock: 1 -> 32,  pool -> (32 x 64 x 64)
    ConvBlock: 32 -> 64, pool -> (64 x 32 x 32)
    ConvBlock: 64 -> 128,pool -> (128 x 16 x 16)
    ConvBlock: 128-> 256,pool -> (256 x 8 x 8)
    GlobalAveragePool    -> (256,)
    Dropout(0.5)
    Linear 256 -> 128 -> n_classes
    """

    def __init__(self, n_classes: int, n_mels: int = 128, n_frames: int = 128):
        super().__init__()
        self.n_classes = n_classes

        self.conv_blocks = nn.Sequential(
            ConvBlock(1,   32),   # 1x128x128 -> 32x64x64
            ConvBlock(32,  64),   # 32x64x64  -> 64x32x32
            ConvBlock(64,  128),  # 64x32x32  -> 128x16x16
            ConvBlock(128, 256),  # 128x16x16 -> 256x8x8
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> 256x1x1
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (batch, 1, n_mels, n_frames)

        Returns
        -------
        Tensor (batch, n_classes) – log-softmax scores
        """
        x = self.conv_blocks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)       # flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def predict_proba(self, x):
        """Return softmax probabilities (not log)."""
        with torch.no_grad():
            log_probs = self.forward(x)
            return torch.exp(log_probs)


def build_model(n_classes: int, device: str = "cpu") -> BioSoundCNN:
    """Instantiate and return a BioSoundCNN on the specified device."""
    model = BioSoundCNN(n_classes=n_classes)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
