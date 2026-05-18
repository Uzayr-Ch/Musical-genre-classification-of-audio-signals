"""Transformer encoder for long-range audio context modeling."""

from __future__ import annotations

import torch
from torch import nn


class Transformer_Encoder(nn.Module):
    """Transformer encoder that models temporal dependencies.

    Expects input shape [B, T, embed_dim] and returns same shape.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_length: int = 5000,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Shape [B, T, embed_dim]

        Returns
        -------
        torch.Tensor
            Shape [B, T, embed_dim]
        """
        b, t, _ = x.shape
        if t > self.max_length:
            raise ValueError(f"Sequence length {t} exceeds max_length {self.max_length}")

        x = x + self.pos_embed[:, :t, :]
        x = self.dropout(x)
        x = self.encoder(x)
        return x


__all__ = ["Transformer_Encoder"]
