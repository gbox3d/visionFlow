from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def load_audio(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 and resample to target_sr."""
    waveform, sr = sf.read(path)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != target_sr:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return waveform.astype(np.float32), sr


def sliding_windows(
    waveform: np.ndarray,
    sample_rate: int,
    window_sec: float = 5.0,
    overlap_sec: float = 2.0,
    min_chunk_sec: float = 1.0,
):
    window_size = int(window_sec * sample_rate)
    stride = int((window_sec - overlap_sec) * sample_rate)
    min_chunk_size = int(min_chunk_sec * sample_rate)

    for start in range(0, len(waveform), stride):
        end = start + window_size
        chunk = waveform[start:end]
        if len(chunk) < min_chunk_size:
            continue
        yield chunk, start / sample_rate

