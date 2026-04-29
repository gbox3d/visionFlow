from __future__ import annotations

import io
import math
import wave

import numpy as np

from ttsFlow.engines.base import TtsEngine, TtsEngineResult


class StubTtsEngine(TtsEngine):
    name = "stub"

    def __init__(self, *, samplerate: int = 22050) -> None:
        self.samplerate = int(samplerate)

    def synthesize(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        speed: float = 1.0,
        audio_format: str = "wav",
    ) -> TtsEngineResult:
        if audio_format != "wav":
            raise ValueError(f"stub engine only supports wav, got: {audio_format}")

        duration_s = min(1.6, max(0.25, len(text) * 0.035))
        sample_count = max(1, int(self.samplerate * duration_s))
        t = np.arange(sample_count, dtype=np.float32) / float(self.samplerate)
        wave_hz = 440.0 + (len(text) % 7) * 35.0
        fade = np.minimum(1.0, np.minimum(t * 18.0, (duration_s - t) * 18.0))
        audio = (0.18 * np.sin(2.0 * math.pi * wave_hz * t) * fade).astype(np.float32)
        pcm = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16)

        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wav_file:
            wav_file.setframerate(self.samplerate)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(pcm.tobytes())

        return TtsEngineResult(
            audio_bytes=wav_bytes.getvalue(),
            samplerate=self.samplerate,
            channels=1,
            duration_ms=int(duration_s * 1000),
            meta={"mode": "smoke_test_tone"},
        )
