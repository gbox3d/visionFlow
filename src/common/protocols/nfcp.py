from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum, IntFlag
import json
import struct
import time
from typing import Any
import zlib


MAGIC = b"NFCP"
VERSION_MAJOR = 1
VERSION_MINOR = 0
HEADER_STRUCT = struct.Struct("<4sBBHBBHHHQQIIIIQII")
HEADER_SIZE = HEADER_STRUCT.size


class ProtocolError(Exception):
    pass


class MessageType(IntEnum):
    REQUEST = 1
    ACK = 2
    EVENT = 3
    RESULT = 4
    ERROR = 5
    CANCEL = 6


class ServiceType(IntEnum):
    COMMON = 0
    ASR = 1
    LLM = 2
    TTS = 3
    PIPELINE = 4
    VISION = 5


class CommandCode(IntEnum):
    PING = 99
    HEALTH = 100
    DESCRIBE = 101

    ASR_TRANSCRIBE = 1001
    ASR_TRANSCRIBE_STREAM = 1002

    LLM_GENERATE = 2001
    LLM_SUMMARIZE = 2002

    TTS_SYNTHESIZE = 3001

    VISION_PROCESS_FRAME = 5001
    VISION_GET_STATE = 5002
    VISION_UPLOAD_FRAME = 5003
    VISION_GET_FRAME = 5004
    VISION_LIST_CAMERAS = 5005

    PIPELINE_RUN = 9001


class StatusCode(IntEnum):
    NONE = 0

    ACCEPTED = 100
    RUNNING = 101
    PARTIAL = 102
    COMPLETED = 103
    CANCELLED = 140
    REJECTED = 141

    BAD_HEADER = 200
    BAD_REQUEST = 201
    BAD_PAYLOAD = 202
    UNSUPPORTED_COMMAND = 203
    UNSUPPORTED_MEDIA = 204
    NOT_READY = 205
    BUSY = 206
    TIMEOUT = 207
    INTERNAL_ERROR = 208
    UNAUTHORIZED = 209
    CHECKSUM_MISMATCH = 210
    TOO_LARGE = 211


class HeaderFlags(IntFlag):
    NONE = 0
    ACK_REQUIRED = 0x0001
    MORE_FRAMES = 0x0002
    END_OF_STREAM = 0x0004
    BODY_CRC32_PRESENT = 0x0008
    META_COMPRESSED = 0x0010
    DATA_COMPRESSED = 0x0020


ACK_REQUIRED = int(HeaderFlags.ACK_REQUIRED)
MORE_FRAMES = int(HeaderFlags.MORE_FRAMES)
END_OF_STREAM = int(HeaderFlags.END_OF_STREAM)
BODY_CRC32_PRESENT = int(HeaderFlags.BODY_CRC32_PRESENT)


def now_ms() -> int:
    return int(time.time() * 1000)


def _encode_meta(meta: dict[str, Any] | None) -> bytes:
    if not meta:
        return b""
    return json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _decode_meta(meta_bytes: bytes) -> dict[str, Any]:
    if not meta_bytes:
        return {}
    try:
        data = json.loads(meta_bytes.decode("utf-8"))
    except Exception as e:
        raise ProtocolError(f"invalid meta json: {e}") from e
    if not isinstance(data, dict):
        raise ProtocolError("meta json must be an object")
    return data


def _compute_crc(meta_bytes: bytes, data: bytes) -> int:
    return zlib.crc32(meta_bytes + data) & 0xFFFFFFFF


@dataclass(slots=True, frozen=True)
class Header:
    magic: bytes = MAGIC
    version_major: int = VERSION_MAJOR
    version_minor: int = VERSION_MINOR
    header_size: int = HEADER_SIZE
    message_type: int = int(MessageType.REQUEST)
    service_type: int = int(ServiceType.COMMON)
    command: int = 0
    status_code: int = int(StatusCode.NONE)
    flags: int = 0
    session_id: int = 0
    request_id: int = 0
    sequence_no: int = 0
    meta_length: int = 0
    data_length: int = 0
    timeout_ms: int = 30_000
    timestamp_ms: int = 0
    body_crc32: int = 0
    reserved: int = 0

    def to_bytes(self) -> bytes:
        return HEADER_STRUCT.pack(
            self.magic,
            self.version_major,
            self.version_minor,
            self.header_size,
            self.message_type,
            self.service_type,
            self.command,
            self.status_code,
            self.flags,
            self.session_id,
            self.request_id,
            self.sequence_no,
            self.meta_length,
            self.data_length,
            self.timeout_ms,
            self.timestamp_ms,
            self.body_crc32,
            self.reserved,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "Header":
        if len(data) != HEADER_SIZE:
            raise ProtocolError(f"invalid header size: {len(data)}")
        header = cls(*HEADER_STRUCT.unpack(data))
        if header.magic != MAGIC:
            raise ProtocolError(f"invalid magic: {header.magic!r}")
        if header.header_size != HEADER_SIZE:
            raise ProtocolError(f"invalid header_size: {header.header_size}")
        if header.version_major != VERSION_MAJOR:
            raise ProtocolError(
                f"unsupported protocol version: {header.version_major}.{header.version_minor}"
            )
        return header

    def with_updates(self, **kwargs: Any) -> "Header":
        return replace(self, **kwargs)


@dataclass(slots=True)
class Frame:
    header: Header
    meta: dict[str, Any] | None = None
    data: bytes = b""

    def to_bytes(self) -> bytes:
        meta_bytes = _encode_meta(self.meta)
        data = self.data or b""
        body_crc32 = self.header.body_crc32
        if self.header.flags & BODY_CRC32_PRESENT:
            body_crc32 = _compute_crc(meta_bytes, data)
        header = self.header.with_updates(
            meta_length=len(meta_bytes),
            data_length=len(data),
            timestamp_ms=self.header.timestamp_ms or now_ms(),
            body_crc32=body_crc32,
        )
        return header.to_bytes() + meta_bytes + data


def _validate_lengths(header: Header) -> None:
    if header.meta_length < 0 or header.data_length < 0:
        raise ProtocolError("negative body length")
    if header.meta_length > 64 * 1024:
        raise ProtocolError("meta too large")
    if header.data_length > 64 * 1024 * 1024:
        raise ProtocolError("data too large")


async def read_frame(reader) -> Frame:
    header = Header.from_bytes(await reader.readexactly(HEADER_SIZE))
    _validate_lengths(header)
    meta_bytes = await reader.readexactly(header.meta_length) if header.meta_length else b""
    data = await reader.readexactly(header.data_length) if header.data_length else b""
    if header.flags & BODY_CRC32_PRESENT:
        if header.body_crc32 != _compute_crc(meta_bytes, data):
            raise ProtocolError("body crc mismatch")
    meta = _decode_meta(meta_bytes)
    return Frame(header=header, meta=meta, data=data)


async def write_frame(writer, frame: Frame) -> None:
    writer.write(frame.to_bytes())
    await writer.drain()


def build_request_frame(
    service_type: ServiceType,
    command: CommandCode,
    *,
    session_id: int = 0,
    request_id: int = 0,
    meta: dict[str, Any] | None = None,
    data: bytes = b"",
    timeout_ms: int = 30_000,
    flags: int = 0,
) -> Frame:
    header = Header(
        message_type=int(MessageType.REQUEST),
        service_type=int(service_type),
        command=int(command),
        status_code=int(StatusCode.NONE),
        flags=flags,
        session_id=session_id,
        request_id=request_id,
        timeout_ms=timeout_ms,
        timestamp_ms=now_ms(),
    )
    return Frame(header=header, meta=meta or {}, data=data)


def build_ack_frame(
    request: Header,
    *,
    meta: dict[str, Any] | None = None,
    sequence_no: int = 0,
) -> Frame:
    header = Header(
        message_type=int(MessageType.ACK),
        service_type=request.service_type,
        command=request.command,
        status_code=int(StatusCode.ACCEPTED),
        flags=0,
        session_id=request.session_id,
        request_id=request.request_id,
        sequence_no=sequence_no,
        timeout_ms=request.timeout_ms,
        timestamp_ms=now_ms(),
    )
    return Frame(header=header, meta=meta or {}, data=b"")


def build_result_frame(
    request: Header,
    *,
    meta: dict[str, Any] | None = None,
    data: bytes = b"",
    status_code: StatusCode = StatusCode.COMPLETED,
    message_type: MessageType = MessageType.RESULT,
    sequence_no: int = 1,
) -> Frame:
    header = Header(
        message_type=int(message_type),
        service_type=request.service_type,
        command=request.command,
        status_code=int(status_code),
        flags=0,
        session_id=request.session_id,
        request_id=request.request_id,
        sequence_no=sequence_no,
        timeout_ms=request.timeout_ms,
        timestamp_ms=now_ms(),
    )
    return Frame(header=header, meta=meta or {}, data=data)


def build_error_frame(
    request: Header,
    *,
    status_code: StatusCode,
    message: str,
    meta: dict[str, Any] | None = None,
    sequence_no: int = 1,
) -> Frame:
    payload = {"error": message}
    if meta:
        payload.update(meta)
    header = Header(
        message_type=int(MessageType.ERROR),
        service_type=request.service_type,
        command=request.command,
        status_code=int(status_code),
        flags=0,
        session_id=request.session_id,
        request_id=request.request_id,
        sequence_no=sequence_no,
        timeout_ms=request.timeout_ms,
        timestamp_ms=now_ms(),
    )
    return Frame(header=header, meta=payload, data=b"")
