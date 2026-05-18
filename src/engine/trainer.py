"""Training orchestration for the hybrid audio genre classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TrainerConfig:
    """Configuration container for training behavior."""

    device: str = "cpu"
    epochs: int = 1


class Trainer:
    """Stub trainer that will later implement train and validation steps."""

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, config: TrainerConfig) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch and return the average loss."""
        raise NotImplementedError("Trainer.train_one_epoch is not implemented yet.")

    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model and return the validation loss or metric."""
        raise NotImplementedError("Trainer.validate is not implemented yet.")
