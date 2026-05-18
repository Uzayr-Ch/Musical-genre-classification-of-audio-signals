"""Hybrid CNN + Transformer model for audio genre classification."""

from __future__ import annotations

import torch
from torch import nn

from .cnn import CNNExtractor
from .transformer import TransformerEncoder


class HybridAudioClassifier(nn.Module):
    """Combine CNN local feature extraction with Transformer context modeling."""

    def __init__(self, num_classes: int, cnn_feature_dim: int = 256) -> None:
        super().__init__()
        self.cnn = CNNExtractor(in_channels=1, feature_dim=cnn_feature_dim)
        self.projection = nn.LazyLinear(cnn_feature_dim)
        self.transformer = TransformerEncoder(embed_dim=cnn_feature_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cnn_feature_dim),
            nn.Linear(cnn_feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the hybrid model end to end.

        Expected input shape: [batch, channels, mel_bins, time_frames]
        """
        features = self.cnn(x)
        batch_size, channels, mel_bins, time_frames = features.shape
        tokens = features.view(batch_size, channels, mel_bins * time_frames).transpose(1, 2)
        tokens = self.projection(tokens)
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
