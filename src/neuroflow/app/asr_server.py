from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from asrFlow.bootstrap import build_processor_from_env, load_runtime_env
from asrFlow.services.nfcp_asr_handler import ProcessorAsrRequestHandler
from asrFlow.utils.env import env_bool_any, env_float_any, env_int_any, env_lang_any, env_str_any, env_value
from voiceFlow.gateways.asr_ingress_server import AsrIngressServer


def resolve_server_host(host: str | None = None) -> str:
    if host:
        return host
    return env_value("NF_ASR_SERVER_HOST") or "0.0.0.0"


def resolve_server_port(port: int | None = None) -> int:
    if port is not None:
        return int(port)
    return env_int_any(("NF_ASR_SERVER_PORT",), 26100)


def _build_streaming_processor():
    """Build streaming processor from env if enabled."""
    if not env_bool_any(("NF_ASR_STREAM_ENABLED",), True):
        return None

    from asrFlow.processors.qwen_streaming_asr import QwenStreamingAsrProcessor

    proc = QwenStreamingAsrProcessor(
        model_name=env_str_any(("ASRFLOW_STREAM_MODEL", "ASRFLOW_STT_MODEL"), "Qwen/Qwen3-ASR-0.6B"),
        model_path=env_value("ASRFLOW_STREAM_MODEL_PATH") or env_value("ASRFLOW_STT_MODEL_PATH") or None,
        language=env_lang_any(("ASRFLOW_STREAM_LANGUAGE", "ASRFLOW_STT_LANGUAGE"), "auto"),
        max_new_tokens=env_int_any(("ASRFLOW_STREAM_MAX_NEW_TOKENS",), 512),
        chunk_size_sec=env_float_any(("ASRFLOW_STREAM_CHUNK_SEC",), 2.0),
        unfixed_chunk_num=env_int_any(("ASRFLOW_STREAM_UNFIXED_CHUNK_NUM",), 2),
        unfixed_token_num=env_int_any(("ASRFLOW_STREAM_UNFIXED_TOKEN_NUM",), 5),
        max_accum_samples=env_int_any(("ASRFLOW_STREAM_MAX_ACCUM_SAMPLES",), 2000),
    )
    proc.warmup()
    return proc


def build_server(
    env_path: str | Path | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
) -> tuple[AsrIngressServer, object]:
    env_result = load_runtime_env(env_path)

    # Batch processor
    processor = build_processor_from_env(config_base_dir=env_result.base_dir)
    processor.warmup()
    handler = ProcessorAsrRequestHandler(processor)

    # Streaming processor
    streaming_processor = _build_streaming_processor()

    server = AsrIngressServer(
        handler=handler,
        streaming_processor=streaming_processor,
        host=resolve_server_host(host),
        port=resolve_server_port(port),
    )
    return server, env_result


def _print_env_load_result(env_result) -> None:
    if env_result.requested_path is not None and env_result.requested_path in env_result.missing_paths:
        print(f"[NeuroFlow ASR] warning: env file not found: {env_result.requested_path}")
    for path in env_result.loaded_paths:
        print(f"[NeuroFlow ASR] loaded env: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch NeuroFlow canonical NFCP ASR server")
    parser.add_argument("--env", default=None, help="Optional env override file path")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    server, env_result = build_server(args.env, host=args.host, port=args.port)
    _print_env_load_result(env_result)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        # PM2 reload/restart during serve_forever can raise KeyboardInterrupt on shutdown.
        print("[NeuroFlow ASR] shutdown requested")


__all__ = ["build_server", "main", "resolve_server_host", "resolve_server_port"]
