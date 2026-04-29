from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from asrFlow.utils.env import env_int_any, env_value
from neuroflow.app.tts_server import _print_env_load_result, build_handler
from ttsFlow.gateways.rest_tts_server import create_tts_rest_app


def resolve_rest_host(host: str | None = None) -> str:
    if host:
        return host
    return env_value("NF_TTS_REST_HOST") or "0.0.0.0"


def resolve_rest_port(port: int | None = None) -> int:
    if port is not None:
        return int(port)
    return env_int_any(("NF_TTS_REST_PORT",), 26121)


def build_app(
    env_path: str | Path | None = None,
    *,
    engine_name: str | None = None,
    model_path: str | None = None,
    config_path: str | None = None,
    model_id: str | None = None,
    warmup: bool = True,
) -> tuple[object, object]:
    handler, env_result = build_handler(
        env_path,
        engine_name=engine_name,
        model_path=model_path,
        config_path=config_path,
        model_id=model_id,
        warmup=warmup,
    )
    return create_tts_rest_app(handler), env_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch NeuroFlow TTS REST API server")
    parser.add_argument("--env", default=None, help="Optional env override file path")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--engine", default=None, help="TTS engine: speecht5-ko | piper-ko | stub")
    parser.add_argument("--model", default=None, help="Optional Piper ONNX model path")
    parser.add_argument("--model-id", default=None, help="Optional Hugging Face model id for speecht5-ko")
    parser.add_argument("--config", default=None, help="Optional Piper ONNX config path")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    app, env_result = build_app(
        args.env,
        engine_name=args.engine,
        model_path=args.model,
        config_path=args.config,
        model_id=args.model_id,
        warmup=not args.no_warmup,
    )
    _print_env_load_result(env_result)
    uvicorn.run(app, host=resolve_rest_host(args.host), port=resolve_rest_port(args.port))


__all__ = ["build_app", "main", "resolve_rest_host", "resolve_rest_port"]
