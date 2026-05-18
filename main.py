"""Entry point for the hybrid audio genre classification project."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.data.dataset import AudioGenreDataset
from src.engine.trainer import Trainer, TrainerConfig
from src.models.hybrid_model import HybridAudioClassifier
from src.utils.helpers import load_config, set_seed


def main() -> None:
    """Load configuration, initialize core components, and prepare training objects."""
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "configs" / "default.yaml")
    set_seed(42)

    dataset = AudioGenreDataset(project_root / "data" / "processed")
    model = HybridAudioClassifier(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=TrainerConfig(device="cpu", epochs=config["epochs"]),
    )

    _ = dataset, trainer


if __name__ == "__main__":
    main()
