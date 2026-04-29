from __future__ import annotations

import io
from typing import Any

import numpy as np
import soundfile as sf


def _to_mono_f32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.shape[1] == 1:
        return audio[:, 0].astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)


def encode_wav_bytes(audio_i16: np.ndarray, samplerate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_i16, samplerate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def decode_audio_bytes(
    audio_bytes: bytes,
    *,
    audio_format: str,
    samplerate: int | None = None,
    channels: int = 1,
    meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    fmt = (audio_format or "wav").strip().lower()
    meta = meta or {}

    if fmt in {"pcm_s16le", "pcm16", "raw_s16le"}:
        if not samplerate:
            raise ValueError("samplerate is required for pcm_s16le")
        pcm16 = np.frombuffer(audio_bytes, dtype="<i2")
        if channels > 1:
            if len(pcm16) % channels != 0:
                raise ValueError("pcm data length is not aligned to channels")
            pcm16 = pcm16.reshape(-1, channels).mean(axis=1)
        audio = pcm16.astype(np.float32) / 32768.0
        return audio, int(samplerate)

    if fmt in {"wav", "flac", "ogg"}:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        return _to_mono_f32(np.asarray(audio)), int(sr)

    try:
        import torchaudio  # type: ignore
    except Exception as e:
        raise ValueError(f"unsupported audio_format without torchaudio: {fmt}") from e

    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).cpu().numpy().astype(np.float32), int(sr)
