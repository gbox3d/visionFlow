from __future__ import annotations

import argparse
import asyncio
import random
from typing import Any

import numpy as np
import sounddevice as sd

from asrFlow.utils.audio import encode_wav_bytes
from common.protocols.nfcp import (
    ACK_REQUIRED,
    CommandCode,
    MessageType,
    ServiceType,
    build_request_frame,
    read_frame,
    write_frame,
)


def _record_audio(duration_s: float, samplerate: int, channels: int, device: int | None) -> np.ndarray:
    frames = int(duration_s * samplerate)
    audio = sd.rec(
        frames,
        samplerate=samplerate,
        channels=channels,
        dtype="int16",
        device=device,
        blocking=True,
    )
    return np.asarray(audio)


async def _send_ping(host: str, port: int, session_id: int) -> None:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        request_id = random.getrandbits(63)
        frame = build_request_frame(
            ServiceType.COMMON,
            CommandCode.PING,
            session_id=session_id,
            request_id=request_id,
            meta={"want_details": True},
        )
        await write_frame(writer, frame)
        response = await read_frame(reader)
        print(f"[mic-client] ping response: {response.meta}")
    finally:
        writer.close()
        await writer.wait_closed()


async def _transcribe(
    host: str,
    port: int,
    audio_bytes: bytes,
    *,
    samplerate: int,
    channels: int,
    session_id: int,
) -> None:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        request_id = random.getrandbits(63)
        request = build_request_frame(
            ServiceType.ASR,
            CommandCode.ASR_TRANSCRIBE,
            session_id=session_id,
            request_id=request_id,
            timeout_ms=30_000,
            flags=ACK_REQUIRED,
            meta={
                "audio_format": "wav",
                "samplerate": samplerate,
                "channels": channels,
                "language": "ko",
                "task": "transcribe",
            },
            data=audio_bytes,
        )
        await write_frame(writer, request)
        while True:
            response = await read_frame(reader)
            msg_type = MessageType(response.header.message_type)
            status = response.header.status_code
            if msg_type == MessageType.ACK:
                print(f"[mic-client] ack: status={status} meta={response.meta}")
                continue
            if msg_type == MessageType.EVENT:
                print(f"[mic-client] event: status={status} meta={response.meta}")
                continue
            if msg_type == MessageType.RESULT:
                print(f"[mic-client] result: {response.meta}")
                return
            print(f"[mic-client] error: status={status} meta={response.meta}")
            return
    finally:
        writer.close()
        await writer.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record microphone audio and send it to the asrFlow server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=26100)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--skip-ping", action="store_true")
    args = parser.parse_args()

    session_id = random.getrandbits(63)
    if not args.skip_ping:
        asyncio.run(_send_ping(args.host, args.port, session_id))

    print(
        f"[mic-client] recording {args.duration:.1f}s from microphone "
        f"(sr={args.samplerate}, ch={args.channels}, device={args.device})"
    )
    audio_i16 = _record_audio(args.duration, args.samplerate, args.channels, args.device)
    wav_bytes = encode_wav_bytes(audio_i16, args.samplerate)
    asyncio.run(
        _transcribe(
            args.host,
            args.port,
            wav_bytes,
            samplerate=args.samplerate,
            channels=args.channels,
            session_id=session_id,
        )
    )


if __name__ == "__main__":
    main()
