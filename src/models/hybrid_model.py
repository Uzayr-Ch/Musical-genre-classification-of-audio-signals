"""Hybrid CNN + Transformer model for audio genre classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn import CNN_Extractor
from src.models.transformer import Transformer_Encoder


class HybridAudioClassifier(nn.Module):
    """Combine CNN local feature extraction with Transformer context modeling."""

    def __init__(self, num_genres: int, cnn_feature_dim: int = 256, transformer_heads: int = 8, transformer_layers: int = 4) -> None:
        super().__init__()
        self.cnn = CNN_Extractor(in_channels=1, feature_dim=cnn_feature_dim)

        # transformer expects input embed dim equal to cnn_feature_dim
        self.transformer = Transformer_Encoder(
            embed_dim=cnn_feature_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(cnn_feature_dim),
            nn.Linear(cnn_feature_dim, num_genres),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass:

        Spectrogram -> CNN -> Transformer -> MeanPool -> Linear logits

        Input: [B, 1, n_mels, T]
        Output: [B, num_genres]
        """
        tokens = self.cnn(x)  # [B, T', feature_dim]
        encoded = self.transformer(tokens)  # [B, T', feature_dim]
        pooled = encoded.mean(dim=1)  # mean over time -> [B, feature_dim]
        logits = self.classifier(pooled)
        return logits


__all__ = ["HybridAudioClassifier"]
