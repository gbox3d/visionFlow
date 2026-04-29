from __future__ import annotations

import asyncio
from typing import Any

from common.contracts.tts_gateway import (
    TtsGatewayError,
    TtsRequestHandler,
    TtsSynthesizeRequest,
    TtsSynthesizeResponse,
)
from common.protocols.nfcp import StatusCode
from ttsFlow.engines.base import TtsEngine


class EngineTtsRequestHandler(TtsRequestHandler):
    def __init__(self, engine: TtsEngine, *, max_chars: int = 500) -> None:
        self.engine = engine
        self.max_chars = int(max_chars)

    def service_meta(self) -> dict[str, Any]:
        meta = {
            "downstream_service": "ttsFlow",
            "max_chars": self.max_chars,
            "supported_audio_formats": ["wav"],
        }
        meta.update(self.engine.service_meta())
        return meta

    async def synthesize(self, request: TtsSynthesizeRequest) -> TtsSynthesizeResponse:
        text = request.text.strip()
        if not text:
            raise TtsGatewayError(int(StatusCode.BAD_REQUEST), "text is required")
        if len(text) > self.max_chars:
            raise TtsGatewayError(
                int(StatusCode.TOO_LARGE),
                f"text too long: {len(text)} chars > {self.max_chars}",
            )
        if request.audio_format != "wav":
            raise TtsGatewayError(
                int(StatusCode.UNSUPPORTED_MEDIA),
                f"unsupported audio_format: {request.audio_format}",
            )

        try:
            result = await asyncio.to_thread(
                self.engine.synthesize,
                text,
                speaker_id=request.speaker_id,
                speed=request.speed,
                audio_format=request.audio_format,
            )
        except TtsGatewayError:
            raise
        except Exception as exc:
            raise TtsGatewayError(int(StatusCode.INTERNAL_ERROR), f"tts processing failed: {exc}") from exc

        return TtsSynthesizeResponse(
            audio_bytes=result.audio_bytes,
            audio_format=result.audio_format,
            samplerate=result.samplerate,
            channels=result.channels,
            duration_ms=result.duration_ms,
            text=text,
            language=request.language,
            processor_meta=result.meta,
        )
