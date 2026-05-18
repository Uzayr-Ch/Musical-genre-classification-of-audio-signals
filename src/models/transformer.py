"""Transformer encoder for long-range audio context modeling."""

from __future__ import annotations

import torch
from torch import nn


class TransformerEncoder(nn.Module):
    """Model global dependencies with multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of feature tokens.

        Parameters
        ----------
        x:
            Tensor of shape [batch, sequence_length, embed_dim].
        """
        sequence_length = x.size(1)
        x = x + self.positional_embedding[:, :sequence_length, :]
        x = self.dropout(x)
        return self.encoder(x)
