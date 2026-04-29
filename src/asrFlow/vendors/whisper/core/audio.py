from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def _resample_linear(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Lightweight linear resampler to avoid pulling heavy optional DSP dependencies.
    """
    if orig_sr == target_sr:
        return waveform.astype(np.float32, copy=False)

    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)

    target_len = max(1, int(round((waveform.shape[0] * target_sr) / float(orig_sr))))
    src_x = np.arange(waveform.shape[0], dtype=np.float64)
    dst_x = np.linspace(0.0, float(waveform.shape[0] - 1), num=target_len, dtype=np.float64)
    return np.interp(dst_x, src_x, waveform).astype(np.float32)


def load_audio(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 and resample to target_sr."""
    waveform, sr = sf.read(path)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != target_sr:
        waveform = _resample_linear(np.asarray(waveform, dtype=np.float32), orig_sr=sr, target_sr=target_sr)
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
