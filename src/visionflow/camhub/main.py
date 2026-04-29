"""
nf-vision-camhub entry point

NFCP TCP server that relays camera frames.

Usage:
    uv run nf-vision-camhub                    # default 0.0.0.0:26200
    uv run nf-vision-camhub --host 127.0.0.1   # bind to localhost only
    uv run nf-vision-camhub --port 8080         # custom port
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    default_host = os.getenv("NF_CAMHUB_HOST", "0.0.0.0")
    default_port = int(os.getenv("NF_CAMHUB_PORT", "26200"))

    parser = argparse.ArgumentParser(description="nf-vision-camhub: Camera Image Hub Server (NFCP TCP)")
    parser.add_argument("--host", default=default_host, help=f"Bind address (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port (default: {default_port})")
    args = parser.parse_args()

    from visionflow.camhub.server import CamHubServer

    server = CamHubServer(host=args.host, port=args.port)
    print(f"[nf-vision-camhub] NFCP TCP server starting on {args.host}:{args.port}")

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[nf-vision-camhub] Stopped")


if __name__ == "__main__":
    main()
