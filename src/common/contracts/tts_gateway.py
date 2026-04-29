from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TtsGatewayError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message


@dataclass(slots=True)
class TtsSynthesizeRequest:
    request_id: int
    sequence_no: int
    timeout_ms: int
    text: str
    audio_format: str = "wav"
    language: str = "ko"
    speaker_id: int | None = None
    speed: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TtsSynthesizeResponse:
    audio_bytes: bytes
    audio_format: str = "wav"
    samplerate: int = 22050
    channels: int = 1
    duration_ms: int = 0
    text: str = ""
    language: str = "ko"
    processor_meta: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class TtsRequestHandler:
    def service_meta(self) -> dict[str, Any]:
        return {}

    async def synthesize(self, request: TtsSynthesizeRequest) -> TtsSynthesizeResponse:
        raise NotImplementedError
