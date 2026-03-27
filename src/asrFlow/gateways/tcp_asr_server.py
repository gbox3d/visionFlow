from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from asrFlow.processors.miso_stt_asr import MisoSttAsrProcessor
from asrFlow.utils.audio import decode_audio_bytes
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class AsrTcpServer:
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 26100,
        processor: MisoSttAsrProcessor | None = None,
        max_data_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.processor = processor or MisoSttAsrProcessor()
        self.max_data_bytes = int(max_data_bytes)
        self._server: asyncio.AbstractServer | None = None

    @property
    def model_loaded(self) -> bool:
        return getattr(self.processor, "_transcriber", None) is not None

    def _service_meta(self) -> dict[str, Any]:
        return {
            "service": "asrFlow",
            "service_type": int(ServiceType.ASR),
            "ready": True,
            "model_loaded": self.model_loaded,
            "commands": [int(CommandCode.PING), int(CommandCode.ASR_TRANSCRIBE)],
        }

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

    async def _handle_transcribe(self, writer, frame: Frame) -> None:
        meta = dict(frame.meta or {})
        if frame.header.data_length > self.max_data_bytes:
            await write_frame(
                writer,
                build_error_frame(
                    frame.header,
                    status_code=StatusCode.TOO_LARGE,
                    message=f"audio payload too large: {frame.header.data_length}",
                ),
            )
            return

        if frame.header.flags & ACK_REQUIRED:
            await write_frame(
                writer,
                build_ack_frame(
                    frame.header,
                    meta={"service": "asrFlow", "state": "accepted"},
                    sequence_no=0,
                ),
            )

        try:
            audio_format = str(meta.get("audio_format", "wav"))
            samplerate = meta.get("samplerate")
            channels = int(meta.get("channels", 1))
            audio_f32, sr = decode_audio_bytes(
                frame.data,
                audio_format=audio_format,
                samplerate=int(samplerate) if samplerate is not None else None,
                channels=channels,
                meta=meta,
            )
        except Exception as e:
            await write_frame(
                writer,
                build_error_frame(
                    frame.header,
                    status_code=StatusCode.BAD_PAYLOAD,
                    message=f"audio decode failed: {e}",
                ),
            )
            return

        try:
            result = self.processor.process(
                audio_chunk=audio_f32,
                samplerate=sr,
                seq=frame.header.sequence_no,
                source_id=f"req-{frame.header.request_id}",
            )
        except Exception as e:
            await write_frame(
                writer,
                build_error_frame(
                    frame.header,
                    status_code=StatusCode.INTERNAL_ERROR,
                    message=f"asr processing failed: {e}",
                ),
            )
            return

        if result is None:
            payload_meta = {
                "text": "",
                "language": str(meta.get("language", "auto")),
                "segments": [],
                "samplerate": sr,
                "audio_format": audio_format,
            }
        else:
            payload_meta = {
                "text": result.text,
                "language": result.language,
                "segments": result.segments,
                "samplerate": sr,
                "audio_format": audio_format,
                "processor_meta": result.meta,
            }

        await write_frame(
            writer,
            build_result_frame(
                frame.header,
                meta=payload_meta,
                status_code=StatusCode.COMPLETED,
                sequence_no=1,
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

            if frame.header.service_type not in (int(ServiceType.COMMON), int(ServiceType.ASR)):
                await write_frame(
                    writer,
                    build_error_frame(
                        frame.header,
                        status_code=StatusCode.BAD_REQUEST,
                        message="invalid service_type for asr server",
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
        print(f"[AsrTcpServer] client connected: {addr}")
        try:
            await self._dispatch(reader, writer)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            print(f"[AsrTcpServer] client closed: {addr}")

    async def run(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        print(f"[AsrTcpServer] listening on {self.host}:{self.port}")
        async with self._server:
            await self._server.serve_forever()


def build_processor_from_env() -> MisoSttAsrProcessor:
    return MisoSttAsrProcessor(
        backend=os.getenv("ASRFLOW_STT_BACKEND", "ct2"),
        model_name=os.getenv("ASRFLOW_STT_MODEL", "large-v3"),
        model_path=os.getenv("ASRFLOW_STT_MODEL_PATH") or None,
        device=os.getenv("ASRFLOW_STT_DEVICE", "auto"),
        fp16=_env_bool("ASRFLOW_STT_FP16", True),
        language=(os.getenv("ASRFLOW_STT_LANGUAGE") or "").strip() or None,
        task=os.getenv("ASRFLOW_STT_TASK", "transcribe"),
        beam_size=int(os.getenv("ASRFLOW_STT_BEAM_SIZE", "5")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NFCP asrFlow TCP server")
    parser.add_argument("--env", default=".env", help="optional .env file path")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.env and Path(args.env).exists():
        load_dotenv(args.env, override=False)
        print(f"[AsrTcpServer] loaded env: {args.env}")

    host = args.host or os.getenv("ASRFLOW_HOST", "0.0.0.0")
    port = args.port or int(os.getenv("ASRFLOW_PORT", "26100"))

    server = AsrTcpServer(
        host=host,
        port=port,
        processor=build_processor_from_env(),
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
