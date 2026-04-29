"""
NFCP camera client for nf-vision-camhub

Captures frames from a local camera and sends VISION_UPLOAD_FRAME over TCP.

Usage:
    uv run nf-vision-camhub-client                              # default cam0
    uv run nf-vision-camhub-client --name front-cam              # custom camera name
    uv run nf-vision-camhub-client --camera-id 1 --fps 15        # camera index 1, 15fps
    uv run nf-vision-camhub-client --host 192.168.0.10           # remote hub
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

import cv2
from dotenv import load_dotenv

from common.protocols.nfcp import (
    CommandCode,
    Frame,
    Header,
    MessageType,
    ServiceType,
    StatusCode,
    build_request_frame,
    read_frame,
    write_frame,
    now_ms,
)


async def _run_client(args) -> None:
    print(f"[camhub-client] Connecting to {args.host}:{args.port} ...")
    reader, writer = await asyncio.open_connection(args.host, args.port)
    print(f"[camhub-client] Connected")

    # --- PING to verify ---
    ping_frame = build_request_frame(
        ServiceType.VISION,
        CommandCode.PING,
        request_id=1,
    )
    await write_frame(writer, ping_frame)
    ping_reply = await read_frame(reader)
    print(f"[camhub-client] PING ok: {ping_reply.meta}")

    # --- open camera ---
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[camhub-client] ERROR: Cannot open camera {args.camera_id}")
        writer.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = 1.0 / args.fps

    print(f"[camhub-client] Camera: {args.name} (id={args.camera_id})")
    print(f"[camhub-client] Resolution: {actual_w}x{actual_h}, Target FPS: {args.fps}")
    print(f"[camhub-client] JPEG quality: {args.quality}")
    print(f"[camhub-client] Press Ctrl+C to stop\n")

    seq = 0
    fps_t0 = time.time()
    fps_n = 0
    request_id = 10

    try:
        while True:
            t0 = time.time()

            ret, frame_img = cap.read()
            if not ret or frame_img is None:
                await asyncio.sleep(0.5)
                continue

            ok, buf = cv2.imencode(".jpg", frame_img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                continue

            jpeg_bytes = buf.tobytes()
            seq += 1
            request_id += 1

            upload_frame = build_request_frame(
                ServiceType.VISION,
                CommandCode.VISION_UPLOAD_FRAME,
                request_id=request_id,
                meta={
                    "name": args.name,
                    "width": actual_w,
                    "height": actual_h,
                },
                data=jpeg_bytes,
            )
            await write_frame(writer, upload_frame)

            reply = await read_frame(reader)
            if reply.header.status_code != int(StatusCode.COMPLETED):
                print(f"[camhub-client] Upload error: {reply.meta}")

            # FPS counter
            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 3.0:
                fps = fps_n / (now - fps_t0)
                print(f"[camhub-client] {args.name}: seq={seq}, {fps:.1f} fps, {len(jpeg_bytes)} bytes")
                fps_n = 0
                fps_t0 = now

            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\n[camhub-client] Stopped. Total frames sent: {seq}")
    except (ConnectionResetError, BrokenPipeError):
        print(f"[camhub-client] Connection lost. Total frames sent: {seq}")
    finally:
        cap.release()
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def main() -> None:
    load_dotenv()

    default_host = os.getenv("NF_CAMHUB_HOST", "127.0.0.1")
    default_port = int(os.getenv("NF_CAMHUB_PORT", "26200"))
    default_name = os.getenv("NF_CAMHUB_CLIENT_NAME", "cam0")
    default_camera_id = int(os.getenv("CAMERA_ID", "0"))
    default_fps = float(os.getenv("NF_CAMHUB_CLIENT_FPS", "10"))
    default_quality = int(os.getenv("NF_CAMHUB_CLIENT_QUALITY", "80"))
    res = os.getenv("CAMERA_RESOLUTION", "640x480")
    default_width, default_height = (int(x) for x in res.split("x"))

    parser = argparse.ArgumentParser(description="NFCP camera client for nf-vision-camhub")
    parser.add_argument("--host", default=default_host, help=f"Hub host (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Hub port (default: {default_port})")
    parser.add_argument("--name", default=default_name, help=f"Camera name identifier (default: {default_name})")
    parser.add_argument("--camera-id", type=int, default=default_camera_id, help=f"OpenCV camera index (default: {default_camera_id})")
    parser.add_argument("--fps", type=float, default=default_fps, help=f"Target FPS (default: {default_fps})")
    parser.add_argument("--quality", type=int, default=default_quality, help=f"JPEG quality 1-100 (default: {default_quality})")
    parser.add_argument("--width", type=int, default=default_width, help=f"Capture width (default: {default_width})")
    parser.add_argument("--height", type=int, default=default_height, help=f"Capture height (default: {default_height})")
    args = parser.parse_args()

    asyncio.run(_run_client(args))


if __name__ == "__main__":
    main()
