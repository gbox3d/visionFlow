from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True, frozen=True)
class AsrTranscribeRequest:
    request_id: int
    sequence_no: int
    timeout_ms: int
    audio_bytes: bytes
    audio_format: str
    samplerate: int | None
    channels: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AsrTranscribeResponse:
    text: str
    language: str
    segments: list[dict[str, Any]]
    samplerate: int
    audio_format: str
    processor_meta: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class AsrGatewayError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message


class AsrRequestHandler(Protocol):
    def service_meta(self) -> dict[str, Any]:
        ...

    async def transcribe(self, request: AsrTranscribeRequest) -> AsrTranscribeResponse:
        ...


__all__ = [
    "AsrGatewayError",
    "AsrRequestHandler",
    "AsrTranscribeRequest",
    "AsrTranscribeResponse",
]
