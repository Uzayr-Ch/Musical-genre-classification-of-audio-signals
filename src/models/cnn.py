"""CNN feature extractor for local spectro-temporal patterns."""

from __future__ import annotations

import torch
from torch import nn


class CNNExtractor(nn.Module):
    """Extract local timbral and rhythmic features from mel-spectrograms."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return convolutional feature maps.

        Expected input shape: [batch, channels, mel_bins, time_frames]
        Output shape: [batch, feature_dim, reduced_mel_bins, reduced_time_frames]
        """
        return self.backbone(x)
