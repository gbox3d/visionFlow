from __future__ import annotations

import asyncio
from typing import Any

from asrFlow.processors.miso_stt_asr import MisoSttAsrProcessor
from common.contracts.asr_gateway import (
    AsrGatewayError,
    AsrTranscribeRequest,
    AsrTranscribeResponse,
)
from common.protocols.nfcp import StatusCode
from common.runtime.audio_codec import decode_audio_bytes


class ProcessorAsrRequestHandler:
    def __init__(self, processor: MisoSttAsrProcessor) -> None:
        self.processor = processor

    def service_meta(self) -> dict[str, Any]:
        return {
            "downstream_service": "asrFlow",
            "model_loaded": getattr(self.processor, "_transcriber", None) is not None,
        }

    async def transcribe(self, request: AsrTranscribeRequest) -> AsrTranscribeResponse:
        try:
            audio_f32, samplerate = decode_audio_bytes(
                request.audio_bytes,
                audio_format=request.audio_format,
                samplerate=request.samplerate,
                channels=request.channels,
                meta=request.meta,
            )
        except Exception as exc:
            raise AsrGatewayError(int(StatusCode.BAD_PAYLOAD), f"audio decode failed: {exc}") from exc

        try:
            result = await asyncio.to_thread(
                self.processor.process,
                audio_f32,
                samplerate,
                request.sequence_no,
                f"req-{request.request_id}",
            )
        except Exception as exc:
            raise AsrGatewayError(int(StatusCode.INTERNAL_ERROR), f"asr processing failed: {exc}") from exc

        if result is None:
            return AsrTranscribeResponse(
                text="",
                language=str(request.meta.get("language", "auto")),
                segments=[],
                samplerate=samplerate,
                audio_format=request.audio_format,
            )

        return AsrTranscribeResponse(
            text=result.text,
            language=result.language,
            segments=result.segments,
            samplerate=samplerate,
            audio_format=request.audio_format,
            processor_meta=result.meta,
        )
