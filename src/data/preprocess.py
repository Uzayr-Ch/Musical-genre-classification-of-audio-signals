"""Audio preprocessing utilities for mel-spectrogram conversion.

Provides an AudioPreprocessor class that loads audio, pads/truncates
to a fixed duration, computes a mel-spectrogram, converts to dB
scale, and returns a PyTorch tensor shaped [1, n_mels, time_frames].
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch


class AudioPreprocessor:
    """Audio preprocessor for producing log-mel spectrogram tensors.

    Parameters
    ----------
    sr: int
        Target sampling rate.
    duration: float
        Target clip duration in seconds (used for pad/truncate).
    n_mels: int
        Number of mel bands.
    n_fft: int
        FFT window size.
    hop_length: int
        Hop length between frames.
    """

    def __init__(
        self,
        sr: int = 22050,
        duration: float = 3.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> None:
        self.sr = int(sr)
        self.duration = float(duration)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.target_length = int(self.sr * self.duration)

    def process_file(self, path: str | Path) -> torch.Tensor:
        """Load an audio file, compute log-mel spectrogram, return tensor.

        Returns
        -------
        torch.Tensor
            Float tensor with shape [1, n_mels, time_frames].
        """
        path = Path(path)
        waveform, _ = librosa.load(path.as_posix(), sr=self.sr, mono=True)

        # Pad or truncate to the target length (in samples)
        if waveform.shape[0] < self.target_length:
            pad_width = self.target_length - waveform.shape[0]
            waveform = np.pad(waveform, (0, pad_width), mode="constant")
        else:
            waveform = waveform[: self.target_length]

        # Compute Mel-spectrogram (power)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

        # Convert to log scale (dB). Use top reference to avoid -inf.
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Convert to float32 and to torch tensor
        tensor = torch.from_numpy(log_mel.astype(np.float32))

        # Ensure shape [1, n_mels, time_frames]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        return tensor


__all__ = ["AudioPreprocessor"]
