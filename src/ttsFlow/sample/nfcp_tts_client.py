from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from common.protocols.nfcp import (
    CommandCode,
    MessageType,
    ServiceType,
    StatusCode,
    build_request_frame,
    read_frame,
    write_frame,
)


async def _request_tts(host: str, port: int, text: str, output: Path, speed: float) -> None:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        frame = build_request_frame(
            ServiceType.TTS,
            CommandCode.TTS_SYNTHESIZE,
            request_id=int(time.time() * 1000) & 0xFFFFFFFF,
            meta={"text": text, "audio_format": "wav", "language": "ko", "speed": speed},
            timeout_ms=30_000,
        )
        await write_frame(writer, frame)
        response = await read_frame(reader)
        if response.header.message_type == int(MessageType.ERROR):
            message = (response.meta or {}).get("message", "tts request failed")
            raise RuntimeError(f"{response.header.status_code}: {message}")
        if response.header.status_code != int(StatusCode.COMPLETED):
            raise RuntimeError(f"unexpected status: {response.header.status_code}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(response.data)
        meta = response.meta or {}
        print(
            f"[nf-tts-client] wrote {output} "
            f"({len(response.data)} bytes, {meta.get('duration_ms', 0)} ms)"
        )
    finally:
        writer.close()
        await writer.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a TTS_SYNTHESIZE request to nf-tts-server")
    parser.add_argument("text")
    parser.add_argument("-o", "--output", type=Path, default=Path("tts_output.wav"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=26120)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    asyncio.run(_request_tts(args.host, args.port, args.text, args.output, args.speed))


__all__ = ["main"]
