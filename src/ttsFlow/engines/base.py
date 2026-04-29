from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TtsEngineResult:
    audio_bytes: bytes
    samplerate: int
    channels: int = 1
    audio_format: str = "wav"
    duration_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class TtsEngine:
    name = "base"

    def warmup(self) -> None:
        return None

    def service_meta(self) -> dict[str, Any]:
        return {"engine": self.name}

    def synthesize(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        speed: float = 1.0,
        audio_format: str = "wav",
    ) -> TtsEngineResult:
        raise NotImplementedError
