"""CNN feature extractor for local spectro-temporal patterns."""

from __future__ import annotations

import torch
from torch import nn


class CNN_Extractor(nn.Module):
    """CNN extractor that converts a mel-spectrogram to a sequence of tokens.

    Input: Tensor [B, 1, n_mels, T]
    Output: Tensor [B, T', feature_dim]
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Shape [B, 1, n_mels, T]

        Returns
        -------
        torch.Tensor
            Shape [B, T_reduced, feature_dim]
        """
        # x: [B, C, M, T]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x now: [B, feature_dim, M', T']
        # collapse the frequency axis (mel dimension) by averaging
        x = x.mean(dim=2)  # -> [B, feature_dim, T']

        # transpose to [B, T', feature_dim]
        x = x.permute(0, 2, 1).contiguous()

        return x


__all__ = ["CNN_Extractor"]
