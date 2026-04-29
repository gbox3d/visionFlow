from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from asrFlow.bootstrap import load_runtime_env
from asrFlow.utils.env import env_float_any, env_int_any, env_str_any, env_value
from ttsFlow.engines.factory import build_tts_engine
from ttsFlow.gateways.tts_server import TtsServer
from ttsFlow.services.nfcp_tts_handler import EngineTtsRequestHandler


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_server_host(host: str | None = None) -> str:
    if host:
        return host
    return env_value("NF_TTS_SERVER_HOST") or "0.0.0.0"


def resolve_server_port(port: int | None = None) -> int:
    if port is not None:
        return int(port)
    return env_int_any(("NF_TTS_SERVER_PORT",), 26120)


def resolve_model_path(model_path: str | None = None) -> Path:
    if model_path:
        return Path(model_path).expanduser()
    value = env_value("NF_TTS_MODEL_PATH")
    if value:
        return Path(value).expanduser()
    return _repo_root() / "models" / "piper-kss-korean.onnx"


def resolve_config_path(config_path: str | None = None, *, model_path: str | Path | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser()
    value = env_value("NF_TTS_CONFIG_PATH")
    if value:
        return Path(value).expanduser()
    base_model_path = Path(model_path).expanduser() if model_path is not None else resolve_model_path()
    return Path(f"{base_model_path}.json")


def build_server(
    env_path: str | Path | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    engine_name: str | None = None,
    model_path: str | None = None,
    config_path: str | None = None,
    model_id: str | None = None,
    warmup: bool = True,
) -> tuple[TtsServer, object]:
    handler, env_result = build_handler(
        env_path,
        engine_name=engine_name,
        model_path=model_path,
        config_path=config_path,
        model_id=model_id,
        warmup=warmup,
    )
    server = TtsServer(
        handler=handler,
        host=resolve_server_host(host),
        port=resolve_server_port(port),
    )
    return server, env_result


def build_handler(
    env_path: str | Path | None = None,
    *,
    engine_name: str | None = None,
    model_path: str | None = None,
    config_path: str | None = None,
    model_id: str | None = None,
    warmup: bool = True,
) -> tuple[EngineTtsRequestHandler, object]:
    env_result = load_runtime_env(env_path)
    resolved_model_path = resolve_model_path(model_path)
    engine = build_tts_engine(
        engine=engine_name or env_str_any(("NF_TTS_ENGINE",), "speecht5-ko"),
        model_path=resolved_model_path,
        config_path=resolve_config_path(config_path, model_path=resolved_model_path),
        model_id=model_id or env_str_any(("NF_TTS_MODEL_ID", "NF_TTS_MODEL"), "ahnhs2k/speecht5-korean"),
        device=env_str_any(("NF_TTS_DEVICE",), "cpu"),
        speed=env_float_any(("NF_TTS_SPEED",), 1.0),
    )
    if warmup:
        engine.warmup()

    handler = EngineTtsRequestHandler(
        engine,
        max_chars=env_int_any(("NF_TTS_MAX_CHARS",), 500),
    )
    return handler, env_result


def _print_env_load_result(env_result) -> None:
    if env_result.requested_path is not None and env_result.requested_path in env_result.missing_paths:
        print(f"[NeuroFlow TTS] warning: env file not found: {env_result.requested_path}", flush=True)
    for path in env_result.loaded_paths:
        print(f"[NeuroFlow TTS] loaded env: {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch NeuroFlow NFCP TTS server")
    parser.add_argument("--env", default=None, help="Optional env override file path")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--engine", default=None, help="TTS engine: speecht5-ko | piper-ko | stub")
    parser.add_argument("--model", default=None, help="Optional Piper ONNX model path")
    parser.add_argument("--model-id", default=None, help="Optional Hugging Face model id for speecht5-ko")
    parser.add_argument("--config", default=None, help="Optional Piper ONNX config path")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    server, env_result = build_server(
        args.env,
        host=args.host,
        port=args.port,
        engine_name=args.engine,
        model_path=args.model,
        config_path=args.config,
        model_id=args.model_id,
        warmup=not args.no_warmup,
    )
    _print_env_load_result(env_result)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("[NeuroFlow TTS] shutdown requested", flush=True)


__all__ = ["build_handler", "build_server", "main", "resolve_server_host", "resolve_server_port"]
