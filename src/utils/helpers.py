"""General helper utilities used across the project."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def save_checkpoint(state: dict[str, Any], checkpoint_path: str | Path) -> None:
    """Save a model checkpoint to disk."""
    torch.save(state, checkpoint_path)
