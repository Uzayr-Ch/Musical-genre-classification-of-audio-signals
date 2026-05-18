"""PyTorch dataset definitions for audio genre classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class AudioGenreDataset(Dataset):
    """Dataset stub for loading audio clips and genre labels.

    This class is intended to read metadata, load preprocessed
    mel-spectrogram tensors, and return samples in a format suitable
    for model training.
    """

    def __init__(self, data_dir: str | Path, transform: Any | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sample as a dictionary of tensors.

        Parameters
        ----------
        index:
            Sample index.
        """
        raise NotImplementedError("AudioGenreDataset.__getitem__ is not implemented yet.")
