from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from common.contracts.tts_gateway import (
    TtsGatewayError,
    TtsRequestHandler,
    TtsSynthesizeRequest,
)
from common.protocols.nfcp import StatusCode
from neuroflow import __version__ as NEUROFLOW_VERSION


class TtsRestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    audio_format: str = "wav"
    language: str = "ko"
    speaker_id: int | None = None
    speed: float = Field(1.0, ge=0.25, le=4.0)


def _http_status(status_code: int) -> int:
    mapping = {
        int(StatusCode.BAD_REQUEST): 400,
        int(StatusCode.BAD_PAYLOAD): 400,
        int(StatusCode.UNSUPPORTED_MEDIA): 415,
        int(StatusCode.NOT_READY): 503,
        int(StatusCode.BUSY): 429,
        int(StatusCode.TIMEOUT): 504,
        int(StatusCode.TOO_LARGE): 413,
    }
    return mapping.get(int(status_code), 500)


def create_tts_rest_app(handler: TtsRequestHandler) -> FastAPI:
    app = FastAPI(
        title="NeuroFlow TTS REST API",
        version=NEUROFLOW_VERSION,
        docs_url="/docs",
        redoc_url=None,
    )
    started_at_ms = int(time.time() * 1000)

    def service_meta() -> dict[str, Any]:
        meta = {
            "service": "ttsFlow",
            "role": "tts_rest_server",
            "transport": "rest",
            "ready": True,
            "version": NEUROFLOW_VERSION,
            "endpoints": ["/health", "/describe", "/tts"],
        }
        meta.update(handler.service_meta() or {})
        return meta

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return service_meta()

    @app.get("/describe")
    async def describe() -> dict[str, Any]:
        meta = service_meta()
        meta.update(
            {
                "protocol": "HTTP/JSON",
                "uptime_ms": max(0, int(time.time() * 1000) - started_at_ms),
                "defaults": {
                    "audio_format": "wav",
                    "language": "ko",
                    "speed": 1.0,
                },
            }
        )
        return meta

    @app.post("/tts", response_class=Response)
    async def synthesize(payload: TtsRestRequest) -> Response:
        request = TtsSynthesizeRequest(
            request_id=int(time.time() * 1000) & 0xFFFFFFFF,
            sequence_no=1,
            timeout_ms=30_000,
            text=payload.text,
            audio_format=payload.audio_format.lower(),
            language=payload.language,
            speaker_id=payload.speaker_id,
            speed=payload.speed,
            meta=payload.model_dump(),
        )
        try:
            result = await handler.synthesize(request)
        except TtsGatewayError as exc:
            raise HTTPException(status_code=_http_status(exc.status_code), detail=exc.message) from exc

        processor_meta = result.processor_meta or {}
        headers = {
            "X-NeuroFlow-TTS-Format": result.audio_format,
            "X-NeuroFlow-TTS-Samplerate": str(result.samplerate),
            "X-NeuroFlow-TTS-Channels": str(result.channels),
            "X-NeuroFlow-TTS-Duration-Ms": str(result.duration_ms),
        }
        if processor_meta.get("provider") is not None:
            headers["X-NeuroFlow-TTS-Provider"] = str(processor_meta["provider"])
        if processor_meta.get("inference_ms") is not None:
            headers["X-NeuroFlow-TTS-Inference-Ms"] = str(processor_meta["inference_ms"])
        if processor_meta.get("rtf") is not None:
            headers["X-NeuroFlow-TTS-RTF"] = f"{float(processor_meta['rtf']):.6f}"

        return Response(
            content=result.audio_bytes,
            media_type="audio/wav",
            headers=headers,
        )

    return app


__all__ = ["TtsRestRequest", "create_tts_rest_app"]
