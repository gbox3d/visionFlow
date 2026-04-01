from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from common.contracts.asr_gateway import (
    AsrGatewayError,
    AsrRequestHandler,
    AsrTranscribeRequest,
    AsrTranscribeResponse,
)
from common.protocols.nfcp import (
    ACK_REQUIRED,
    END_OF_STREAM,
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
        raise AsrGatewayError(int(StatusCode.BAD_REQUEST), f"invalid {field_name}: {value!r}") from exc


def _coerce_required_int(value: Any, *, field_name: str, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AsrGatewayError(int(StatusCode.BAD_REQUEST), f"invalid {field_name}: {value!r}") from exc


class AsrIngressServer:
    def __init__(
        self,
        *,
        handler: AsrRequestHandler,
        streaming_processor: Any | None = None,
        host: str = "0.0.0.0",
        port: int = 26100,
        max_data_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.handler = handler
        self.streaming_processor = streaming_processor
        self.host = host
        self.port = int(port)
        self.max_data_bytes = int(max_data_bytes)
        self._server: asyncio.AbstractServer | None = None
        self._stream_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Service metadata
    # ------------------------------------------------------------------

    def _service_meta(self) -> dict[str, Any]:
        commands = [int(CommandCode.PING), int(CommandCode.DESCRIBE), int(CommandCode.ASR_TRANSCRIBE)]
        if self.streaming_processor is not None:
            commands.append(int(CommandCode.ASR_TRANSCRIBE_STREAM))
        meta = {
            "service": "voiceFlow",
            "role": "asr_ingress_server",
            "service_type": int(ServiceType.ASR),
            "ready": True,
            "version": NEUROFLOW_VERSION,
            "commands": commands,
        }
        meta.update(self.handler.service_meta() or {})
        meta.setdefault("service", "voiceFlow")
        meta.setdefault("role", "asr_ingress_server")
        meta.setdefault("service_type", int(ServiceType.ASR))
        meta.setdefault("ready", True)
        return meta

    # ------------------------------------------------------------------
    # PING / DESCRIBE
    # ------------------------------------------------------------------

    async def _send_ping(self, writer, frame: Frame) -> None:
        reply = build_result_frame(
            frame.header,
            meta=self._service_meta(),
            status_code=StatusCode.COMPLETED,
            sequence_no=0,
        )
        await write_frame(writer, reply)

    async def _send_describe(self, writer, frame: Frame) -> None:
        meta = self._service_meta()
        meta.update(
            {
                "protocol": "NFCP/1.0",
                "supported_audio_formats": ["wav", "pcm_s16le", "mp3", "webm", "mp4"],
                "streaming": self.streaming_processor is not None,
                "defaults": {
                    "samplerate": 16000,
                    "channels": 1,
                    "task": "transcribe",
                },
            }
        )
        reply = build_result_frame(
            frame.header,
            meta=meta,
            status_code=StatusCode.COMPLETED,
            sequence_no=0,
        )
        await write_frame(writer, reply)

    # ------------------------------------------------------------------
    # Batch ASR_TRANSCRIBE (existing)
    # ------------------------------------------------------------------

    def _build_transcribe_request(self, frame: Frame) -> AsrTranscribeRequest:
        meta = dict(frame.meta or {})
        if frame.header.data_length > self.max_data_bytes:
            raise AsrGatewayError(
                int(StatusCode.TOO_LARGE),
                f"audio payload too large: {frame.header.data_length}",
            )

        return AsrTranscribeRequest(
            request_id=int(frame.header.request_id),
            sequence_no=int(frame.header.sequence_no),
            timeout_ms=int(frame.header.timeout_ms),
            audio_bytes=frame.data,
            audio_format=str(meta.get("audio_format", "wav")),
            samplerate=_coerce_optional_int(meta.get("samplerate"), field_name="samplerate"),
            channels=_coerce_required_int(meta.get("channels"), field_name="channels", default=1),
            meta=meta,
        )

    def _build_result_meta(self, response: AsrTranscribeResponse) -> dict[str, Any]:
        payload_meta = {
            "text": response.text,
            "language": response.language,
            "segments": response.segments,
            "samplerate": response.samplerate,
            "audio_format": response.audio_format,
        }
        if response.processor_meta is not None:
            payload_meta["processor_meta"] = response.processor_meta
        payload_meta.update(response.meta or {})
        return payload_meta

    async def _handle_transcribe(self, writer, frame: Frame) -> None:
        if frame.header.flags & ACK_REQUIRED:
            await write_frame(
                writer,
                build_ack_frame(
                    frame.header,
                    meta={"service": "voiceFlow", "state": "accepted"},
                    sequence_no=0,
                ),
            )

        try:
            request = self._build_transcribe_request(frame)
            response = await self.handler.transcribe(request)
        except AsrGatewayError as exc:
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
                status_code=StatusCode.COMPLETED,
                sequence_no=max(1, int(frame.header.sequence_no)),
            ),
        )

    # ------------------------------------------------------------------
    # Streaming ASR_TRANSCRIBE_STREAM
    # ------------------------------------------------------------------

    async def _handle_stream(self, writer, frame: Frame, stream_state: dict) -> dict:
        """Handle one frame of a streaming session. Returns updated stream_state."""
        proc = self.streaming_processor
        meta = dict(frame.meta or {})
        action = meta.get("action", "")

        # --- STREAM START ---
        if action == "start":
            if stream_state.get("active"):
                await write_frame(writer, build_error_frame(
                    frame.header,
                    status_code=StatusCode.BUSY,
                    message="stream session already active",
                ))
                return stream_state

            if proc is None:
                await write_frame(writer, build_error_frame(
                    frame.header,
                    status_code=StatusCode.UNSUPPORTED_COMMAND,
                    message="streaming processor not configured",
                ))
                return stream_state

            # Reset processor state for new session
            await asyncio.to_thread(proc.reset)
            await asyncio.to_thread(proc.warmup)

            stream_state = {
                "active": True,
                "samplerate": int(meta.get("samplerate", 16000)),
                "channels": int(meta.get("channels", 1)),
                "seq": 0,
            }

            await write_frame(writer, build_ack_frame(
                frame.header,
                meta={"service": "voiceFlow", "state": "stream_started"},
                sequence_no=0,
            ))
            print(f"[AsrIngressServer] stream started: sr={stream_state['samplerate']}")
            return stream_state

        # --- STREAM END ---
        if action == "end" or (frame.header.flags & END_OF_STREAM):
            if not stream_state.get("active"):
                await write_frame(writer, build_error_frame(
                    frame.header,
                    status_code=StatusCode.BAD_REQUEST,
                    message="no active stream session",
                ))
                return stream_state

            seq = stream_state["seq"] + 1
            try:
                result = await asyncio.to_thread(proc.flush, seq=seq, source_id="stream")
            except Exception as exc:
                await write_frame(writer, build_error_frame(
                    frame.header,
                    status_code=StatusCode.INTERNAL_ERROR,
                    message=f"flush failed: {exc}",
                ))
                stream_state["active"] = False
                return stream_state

            text = result.text if result else ""
            language = result.language if result else ""
            full_text = (result.meta.get("full_text", text) if result and result.meta else text)

            await write_frame(writer, build_result_frame(
                frame.header,
                meta={
                    "text": text,
                    "full_text": full_text,
                    "language": language,
                    "event": "final",
                    "sequence_no": seq,
                },
                status_code=StatusCode.COMPLETED,
                sequence_no=seq,
            ))
            print(f"[AsrIngressServer] stream ended (seq={seq})")
            stream_state["active"] = False
            return stream_state

        # --- STREAM CHUNK (default) ---
        if not stream_state.get("active"):
            await write_frame(writer, build_error_frame(
                frame.header,
                status_code=StatusCode.BAD_REQUEST,
                message="no active stream session – send action:start first",
            ))
            return stream_state

        pcm_bytes = frame.data
        if not pcm_bytes:
            await write_frame(writer, build_result_frame(
                frame.header,
                meta={"text": "", "event": "ack", "sequence_no": stream_state["seq"]},
                status_code=StatusCode.PARTIAL,
                sequence_no=stream_state["seq"],
            ))
            return stream_state

        # PCM S16LE bytes → float32 numpy
        audio_f32 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        stream_state["seq"] += 1
        seq = stream_state["seq"]

        try:
            result = await asyncio.to_thread(
                proc.feed_audio, audio_f32, samplerate=stream_state["samplerate"], seq=seq, source_id="stream",
            )
        except Exception as exc:
            await write_frame(writer, build_error_frame(
                frame.header,
                status_code=StatusCode.INTERNAL_ERROR,
                message=f"feed_audio failed: {exc}",
            ))
            return stream_state

        text = result.text if result else ""
        language = result.language if result else ""
        full_text = (result.meta.get("full_text", text) if result and result.meta else text)

        await write_frame(writer, build_result_frame(
            frame.header,
            meta={
                "text": text,
                "full_text": full_text,
                "language": language,
                "event": "partial",
                "sequence_no": seq,
            },
            status_code=StatusCode.PARTIAL,
            sequence_no=seq,
        ))
        return stream_state

    # ------------------------------------------------------------------
    # Dispatch loop
    # ------------------------------------------------------------------

    async def _dispatch(self, reader, writer) -> None:
        stream_state: dict = {"active": False}

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

            if frame.header.service_type not in (int(ServiceType.COMMON), int(ServiceType.ASR)):
                await write_frame(
                    writer,
                    build_error_frame(
                        frame.header,
                        status_code=StatusCode.BAD_REQUEST,
                        message="invalid service_type for ASR ingress server",
                    ),
                )
                continue

            command = frame.header.command
            if command == int(CommandCode.PING):
                await self._send_ping(writer, frame)
            elif command == int(CommandCode.DESCRIBE):
                await self._send_describe(writer, frame)
            elif command == int(CommandCode.ASR_TRANSCRIBE):
                await self._handle_transcribe(writer, frame)
            elif command == int(CommandCode.ASR_TRANSCRIBE_STREAM):
                stream_state = await self._handle_stream(writer, frame, stream_state)
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
        print(f"[AsrIngressServer] client connected: {addr}")
        try:
            await self._dispatch(reader, writer)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            print(f"[AsrIngressServer] client closed: {addr}")

    async def run(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        caps = ["batch"]
        if self.streaming_processor is not None:
            caps.append("stream")
        print(f"[AsrIngressServer] listening on {self.host}:{self.port} ({', '.join(caps)})")
        async with self._server:
            await self._server.serve_forever()


__all__ = ["AsrIngressServer"]
