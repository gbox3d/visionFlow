"""
nf-vision-camhub NFCP TCP server

Camera clients connect and send VISION_UPLOAD_FRAME with JPEG data.
AI clients connect and send VISION_GET_FRAME to fetch the latest frame.

Commands:
  PING                 -> service meta
  DESCRIBE             -> detailed capabilities
  VISION_UPLOAD_FRAME  -> camera client uploads JPEG (meta: name, width, height)
  VISION_GET_FRAME     -> AI client fetches latest JPEG (meta: name)
  VISION_LIST_CAMERAS  -> list all registered cameras
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from common.protocols.nfcp import (
    CommandCode,
    Frame,
    MessageType,
    ServiceType,
    StatusCode,
    build_error_frame,
    build_result_frame,
    read_frame,
    write_frame,
)
from neuroflow import __version__ as NEUROFLOW_VERSION

from visionflow.camhub.hub import FrameHub


class CamHubServer:
    """NFCP TCP server that relays camera frames between camera clients and AI clients."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 26200,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.hub = FrameHub()
        self._server: asyncio.AbstractServer | None = None

    # ------------------------------------------------------------------
    # Service metadata
    # ------------------------------------------------------------------

    def _service_meta(self) -> dict[str, Any]:
        return {
            "service": "visionFlow",
            "role": "camhub_server",
            "service_type": int(ServiceType.VISION),
            "ready": True,
            "version": NEUROFLOW_VERSION,
            "commands": [
                int(CommandCode.PING),
                int(CommandCode.DESCRIBE),
                int(CommandCode.VISION_UPLOAD_FRAME),
                int(CommandCode.VISION_GET_FRAME),
                int(CommandCode.VISION_LIST_CAMERAS),
            ],
        }

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
        cameras = self.hub.list_cameras()
        meta.update(
            {
                "protocol": "NFCP/1.0",
                "description": "Camera image hub - receives frames from camera clients, serves to AI clients",
                "cameras": cameras,
                "camera_count": len(cameras),
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
    # VISION_UPLOAD_FRAME
    # ------------------------------------------------------------------

    async def _handle_upload_frame(self, writer, frame: Frame) -> None:
        meta = dict(frame.meta or {})
        name = meta.get("name")
        if not name:
            await write_frame(
                writer,
                build_error_frame(
                    frame.header,
                    status_code=StatusCode.BAD_REQUEST,
                    message="meta.name is required",
                ),
            )
            return

        jpeg_data = frame.data
        if not jpeg_data:
            await write_frame(
                writer,
                build_error_frame(
                    frame.header,
                    status_code=StatusCode.BAD_PAYLOAD,
                    message="empty frame data",
                ),
            )
            return

        width = int(meta.get("width", 0))
        height = int(meta.get("height", 0))
        extra_meta = {k: v for k, v in meta.items() if k not in ("name", "width", "height")}

        stored = self.hub.put(
            name=name,
            jpeg=jpeg_data,
            width=width,
            height=height,
            meta=extra_meta,
        )

        await write_frame(
            writer,
            build_result_frame(
                frame.header,
                meta={
                    "camera": name,
                    "seq": stored.seq,
                    "size": len(jpeg_data),
                },
                status_code=StatusCode.COMPLETED,
                sequence_no=stored.seq,
            ),
        )

    # ------------------------------------------------------------------
    # VISION_GET_FRAME
    # ------------------------------------------------------------------

    async def _handle_get_frame(self, writer, frame: Frame) -> None:
        meta = dict(frame.meta or {})
        name = meta.get("name")
        if not name:
            await write_frame(
                writer,
                build_error_frame(
                    frame.header,
                    status_code=StatusCode.BAD_REQUEST,
                    message="meta.name is required",
                ),
            )
            return

        stored = self.hub.get(name)
        if stored is None:
            await write_frame(
                writer,
                build_error_frame(
                    frame.header,
                    status_code=StatusCode.BAD_REQUEST,
                    message=f"camera '{name}' not found",
                ),
            )
            return

        await write_frame(
            writer,
            build_result_frame(
                frame.header,
                meta={
                    "camera": stored.name,
                    "seq": stored.seq,
                    "width": stored.width,
                    "height": stored.height,
                    "timestamp": stored.timestamp,
                },
                data=stored.jpeg,
                status_code=StatusCode.COMPLETED,
                sequence_no=stored.seq,
            ),
        )

    # ------------------------------------------------------------------
    # VISION_LIST_CAMERAS
    # ------------------------------------------------------------------

    async def _handle_list_cameras(self, writer, frame: Frame) -> None:
        cameras = self.hub.list_cameras()
        await write_frame(
            writer,
            build_result_frame(
                frame.header,
                meta={
                    "cameras": cameras,
                    "count": len(cameras),
                },
                status_code=StatusCode.COMPLETED,
                sequence_no=0,
            ),
        )

    # ------------------------------------------------------------------
    # Dispatch loop
    # ------------------------------------------------------------------

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

            if frame.header.service_type not in (int(ServiceType.COMMON), int(ServiceType.VISION)):
                await write_frame(
                    writer,
                    build_error_frame(
                        frame.header,
                        status_code=StatusCode.BAD_REQUEST,
                        message="invalid service_type for camhub server",
                    ),
                )
                continue

            command = frame.header.command
            if command == int(CommandCode.PING):
                await self._send_ping(writer, frame)
            elif command == int(CommandCode.DESCRIBE):
                await self._send_describe(writer, frame)
            elif command == int(CommandCode.VISION_UPLOAD_FRAME):
                await self._handle_upload_frame(writer, frame)
            elif command == int(CommandCode.VISION_GET_FRAME):
                await self._handle_get_frame(writer, frame)
            elif command == int(CommandCode.VISION_LIST_CAMERAS):
                await self._handle_list_cameras(writer, frame)
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
        print(f"[CamHubServer] client connected: {addr}")
        try:
            await self._dispatch(reader, writer)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            print(f"[CamHubServer] client closed: {addr}")

    async def run(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port,
        )
        print(f"[CamHubServer] listening on {self.host}:{self.port}")
        print(f"[CamHubServer] VISION_UPLOAD_FRAME={int(CommandCode.VISION_UPLOAD_FRAME)}"
              f"  VISION_GET_FRAME={int(CommandCode.VISION_GET_FRAME)}"
              f"  VISION_LIST_CAMERAS={int(CommandCode.VISION_LIST_CAMERAS)}")
        async with self._server:
            await self._server.serve_forever()


__all__ = ["CamHubServer"]
