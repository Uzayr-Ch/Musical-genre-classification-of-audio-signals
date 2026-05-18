"""Audio preprocessing utilities for mel-spectrogram conversion."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(audio_path: str | Path, sr: int) -> tuple[np.ndarray, int]:
    """Load an audio file with librosa at a fixed sample rate."""
    waveform, sample_rate = librosa.load(audio_path, sr=sr, mono=True)
    return waveform, sample_rate


def audio_to_mel_spectrogram(
    audio_path: str | Path,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int = 128,
) -> np.ndarray:
    """Convert raw audio into a log-mel spectrogram.

    Parameters
    ----------
    audio_path:
        Path to the input audio file.
    sr:
        Target sample rate.
    n_fft:
        FFT window size.
    hop_length:
        Frame hop size.
    n_mels:
        Number of mel bands.
    """
    waveform, _ = load_audio(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec.astype(np.float32)
