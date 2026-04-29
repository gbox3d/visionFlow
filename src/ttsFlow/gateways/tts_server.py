from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from common.contracts.tts_gateway import (
    TtsGatewayError,
    TtsRequestHandler,
    TtsSynthesizeRequest,
    TtsSynthesizeResponse,
)
from common.protocols.nfcp import (
    ACK_REQUIRED,
    CommandCode,
    Frame,
    MessageType,
    ServiceType,
    StatusCode,
    build_ack_frame,
    build_error_frame,
    build_result_frame,
    read_frame,
    write_frame,
)
from neuroflow import __version__ as NEUROFLOW_VERSION


def _coerce_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TtsGatewayError(int(StatusCode.BAD_REQUEST), f"invalid {field_name}: {value!r}") from exc


def _coerce_float(value: Any, *, field_name: str, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TtsGatewayError(int(StatusCode.BAD_REQUEST), f"invalid {field_name}: {value!r}") from exc


class TtsServer:
    def __init__(
        self,
        *,
        handler: TtsRequestHandler,
        host: str = "0.0.0.0",
        port: int = 26120,
    ) -> None:
        self.handler = handler
        self.host = host
        self.port = int(port)
        self._server: asyncio.AbstractServer | None = None
        self._started_at_ms = int(time.time() * 1000)

    def _service_meta(self) -> dict[str, Any]:
        meta = {
            "service": "ttsFlow",
            "role": "tts_server",
            "service_type": int(ServiceType.TTS),
            "ready": True,
            "version": NEUROFLOW_VERSION,
            "commands": [
                int(CommandCode.PING),
                int(CommandCode.DESCRIBE),
                int(CommandCode.SERVER_INFO),
                int(CommandCode.TTS_SYNTHESIZE),
            ],
        }
        meta.update(self.handler.service_meta() or {})
        meta.setdefault("service", "ttsFlow")
        meta.setdefault("role", "tts_server")
        meta.setdefault("service_type", int(ServiceType.TTS))
        meta.setdefault("ready", True)
        return meta

    def _server_info_meta(self) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        meta = self._service_meta()
        meta.update(
            {
                "protocol": "NFCP/1.0",
                "pid": os.getpid(),
                "host": self.host,
                "port": self.port,
                "uptime_ms": max(0, now_ms - self._started_at_ms),
            }
        )
        return meta

    async def _send_ping(self, writer, frame: Frame) -> None:
        await write_frame(
            writer,
            build_result_frame(
                frame.header,
                meta=self._service_meta(),
                status_code=StatusCode.COMPLETED,
                sequence_no=0,
            ),
        )

    async def _send_describe(self, writer, frame: Frame) -> None:
        meta = self._server_info_meta()
        meta.update(
            {
                "defaults": {
                    "audio_format": "wav",
                    "language": "ko",
                    "speed": 1.0,
                }
            }
        )
        await write_frame(
            writer,
            build_result_frame(
                frame.header,
                meta=meta,
                status_code=StatusCode.COMPLETED,
                sequence_no=0,
            ),
        )

    async def _send_server_info(self, writer, frame: Frame) -> None:
        await write_frame(
            writer,
            build_result_frame(
                frame.header,
                meta=self._server_info_meta(),
                status_code=StatusCode.COMPLETED,
                sequence_no=0,
            ),
        )

    def _build_synthesize_request(self, frame: Frame) -> TtsSynthesizeRequest:
        meta = dict(frame.meta or {})
        data_text = frame.data.decode("utf-8").strip() if frame.data else ""
        text = str(meta.get("text") or data_text).strip()
        return TtsSynthesizeRequest(
            request_id=int(frame.header.request_id),
            sequence_no=int(frame.header.sequence_no),
            timeout_ms=int(frame.header.timeout_ms),
            text=text,
            audio_format=str(meta.get("audio_format", "wav")).lower(),
            language=str(meta.get("language", "ko")),
            speaker_id=_coerce_optional_int(meta.get("speaker_id"), field_name="speaker_id"),
            speed=_coerce_float(meta.get("speed"), field_name="speed", default=1.0),
            meta=meta,
        )

    def _build_result_meta(self, response: TtsSynthesizeResponse) -> dict[str, Any]:
        meta = {
            "text": response.text,
            "language": response.language,
            "audio_format": response.audio_format,
            "samplerate": response.samplerate,
            "channels": response.channels,
            "duration_ms": response.duration_ms,
        }
        if response.processor_meta is not None:
            meta["processor_meta"] = response.processor_meta
        meta.update(response.meta or {})
        return meta

    async def _handle_synthesize(self, writer, frame: Frame) -> None:
        if frame.header.flags & ACK_REQUIRED:
            await write_frame(
                writer,
                build_ack_frame(
                    frame.header,
                    meta={"service": "ttsFlow", "state": "accepted"},
                    sequence_no=0,
                ),
            )

        try:
            request = self._build_synthesize_request(frame)
            response = await self.handler.synthesize(request)
        except TtsGatewayError as exc:
            await write_frame(
                writer,
                build_error_frame(
                    frame.header,
                    status_code=exc.status_code,
                    message=exc.message,
                ),
            )
            return
        except Exception as exc:
            await write_frame(
                writer,
                build_error_frame(
                    frame.header,
                    status_code=StatusCode.INTERNAL_ERROR,
                    message=f"handler failure: {exc}",
                ),
            )
            return

        await write_frame(
            writer,
            build_result_frame(
                frame.header,
                meta=self._build_result_meta(response),
                data=response.audio_bytes,
                status_code=StatusCode.COMPLETED,
                sequence_no=max(1, int(frame.header.sequence_no)),
            ),
        )

    async def _dispatch(self, reader, writer) -> None:
        while True:
            try:
                frame = await read_frame(reader)
            except asyncio.IncompleteReadError:
                return
            except Exception:
                return

            if frame.header.message_type != int(MessageType.REQUEST):
                await write_frame(
                    writer,
                    build_error_frame(
                        frame.header,
                        status_code=StatusCode.BAD_REQUEST,
                        message="message_type must be REQUEST",
                    ),
                )
                continue

            if frame.header.service_type not in (int(ServiceType.COMMON), int(ServiceType.TTS)):
                await write_frame(
                    writer,
                    build_error_frame(
                        frame.header,
                        status_code=StatusCode.BAD_REQUEST,
                        message="invalid service_type for TTS server",
                    ),
                )
                continue

            command = frame.header.command
            if command == int(CommandCode.PING):
                await self._send_ping(writer, frame)
            elif command == int(CommandCode.DESCRIBE):
                await self._send_describe(writer, frame)
            elif command == int(CommandCode.SERVER_INFO):
                await self._send_server_info(writer, frame)
            elif command == int(CommandCode.TTS_SYNTHESIZE):
                await self._handle_synthesize(writer, frame)
            else:
                await write_frame(
                    writer,
                    build_error_frame(
                        frame.header,
                        status_code=StatusCode.UNSUPPORTED_COMMAND,
                        message=f"unsupported command: {command}",
                    ),
                )

    async def _handle_client(self, reader, writer) -> None:
        addr = writer.get_extra_info("peername")
        print(f"[TtsServer] client connected: {addr}", flush=True)
        try:
            await self._dispatch(reader, writer)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            print(f"[TtsServer] client closed: {addr}", flush=True)

    async def run(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        print(f"[TtsServer] listening on {self.host}:{self.port}", flush=True)
        async with self._server:
            await self._server.serve_forever()


__all__ = ["TtsServer"]
