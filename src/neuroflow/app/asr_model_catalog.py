from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from asrFlow.processors.miso_stt_asr import BackendName
from asrFlow.vendors.whisper.core.config import CT2_MODELS, WHISPER_MODELS


NativeStreamBackendName = Literal["qwen_transformers"]

WHISPER_BACKENDS: tuple[BackendName, ...] = ("ct2", "hf_generate", "hf_pipeline")
QWEN_BACKENDS: tuple[BackendName, ...] = ("qwen_transformers",)
ALL_BACKENDS: tuple[BackendName, ...] = WHISPER_BACKENDS + QWEN_BACKENDS
QWEN_NATIVE_STREAM_BACKENDS: tuple[NativeStreamBackendName, ...] = ("qwen_transformers",)


@dataclass(frozen=True)
class ModelPreset:
    model_name: str
    label: str
    supported_backends: tuple[BackendName, ...]
    default_backend: BackendName
    note: str


@dataclass(frozen=True)
class NativeStreamModelPreset:
    model_name: str
    label: str
    supported_backends: tuple[NativeStreamBackendName, ...]
    default_backend: NativeStreamBackendName
    note: str


MODEL_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        model_name="tiny",
        label="Whisper tiny",
        supported_backends=WHISPER_BACKENDS,
        default_backend="ct2",
        note="가벼운 Whisper baseline",
    ),
    ModelPreset(
        model_name="base",
        label="Whisper base",
        supported_backends=WHISPER_BACKENDS,
        default_backend="ct2",
        note="작은 Whisper baseline",
    ),
    ModelPreset(
        model_name="small",
        label="Whisper small",
        supported_backends=WHISPER_BACKENDS,
        default_backend="ct2",
        note="실험용 기본 Whisper",
    ),
    ModelPreset(
        model_name="medium",
        label="Whisper medium",
        supported_backends=WHISPER_BACKENDS,
        default_backend="ct2",
        note="정확도 우선 Whisper",
    ),
    ModelPreset(
        model_name="large-v3",
        label="Whisper large-v3",
        supported_backends=WHISPER_BACKENDS,
        default_backend="ct2",
        note="대표적인 고품질 Whisper",
    ),
    ModelPreset(
        model_name="openai/whisper-large-v3-turbo",
        label="Whisper large-v3-turbo",
        supported_backends=("hf_generate", "hf_pipeline"),
        default_backend="hf_generate",
        note="HF backend 전용 turbo 계열",
    ),
    ModelPreset(
        model_name="Qwen/Qwen3-ASR-0.6B",
        label="Qwen3-ASR-0.6B",
        supported_backends=QWEN_BACKENDS,
        default_backend="qwen_transformers",
        note="Qwen ASR 계열. chunk 예제에서는 transformers backend로 고정 청크 추론",
    ),
)

NATIVE_STREAM_MODEL_PRESETS: tuple[NativeStreamModelPreset, ...] = (
    NativeStreamModelPreset(
        model_name="Qwen/Qwen3-ASR-0.6B",
        label="Qwen3-ASR-0.6B",
        supported_backends=QWEN_NATIVE_STREAM_BACKENDS,
        default_backend="qwen_transformers",
        note="Qwen native streaming (transformers backend, Windows/Linux 모두 지원)",
    ),
)


_PRESET_BY_MODEL = {preset.model_name.strip().lower(): preset for preset in MODEL_PRESETS}
_NATIVE_STREAM_PRESET_BY_MODEL = {
    preset.model_name.strip().lower(): preset for preset in NATIVE_STREAM_MODEL_PRESETS
}
_WHISPER_NAMES = {
    *(name.strip().lower() for name in WHISPER_MODELS),
    *(name.strip().lower() for name in WHISPER_MODELS.values()),
    *(name.strip().lower() for name in CT2_MODELS),
    *(name.strip().lower() for name in CT2_MODELS.values()),
}


def preset_model_names() -> list[str]:
    return [preset.model_name for preset in MODEL_PRESETS]


def resolve_model_preset(model_name: str) -> ModelPreset | None:
    return _PRESET_BY_MODEL.get(model_name.strip().lower())


def native_stream_model_names() -> list[str]:
    return [preset.model_name for preset in NATIVE_STREAM_MODEL_PRESETS]


def resolve_native_stream_model_preset(model_name: str) -> NativeStreamModelPreset | None:
    return _NATIVE_STREAM_PRESET_BY_MODEL.get(model_name.strip().lower())


def infer_supported_backends(model_name: str, *, model_path: str | None = None) -> tuple[BackendName, ...]:
    preset = resolve_model_preset(model_name)
    if preset is not None:
        return preset.supported_backends

    raw_name = model_name.strip().lower()
    raw_path = (model_path or "").strip().lower()
    probe = raw_path or raw_name

    if "qwen" in probe and "asr" in probe:
        return QWEN_BACKENDS
    if raw_name in _WHISPER_NAMES or "whisper" in probe or "faster-whisper" in probe:
        return WHISPER_BACKENDS

    return ALL_BACKENDS


def infer_native_stream_backends(
    model_name: str,
    *,
    model_path: str | None = None,
) -> tuple[NativeStreamBackendName, ...]:
    preset = resolve_native_stream_model_preset(model_name)
    if preset is not None:
        return preset.supported_backends

    raw_name = model_name.strip().lower()
    raw_path = (model_path or "").strip().lower()
    probe = raw_path or raw_name

    if "qwen" in probe and "asr" in probe:
        return QWEN_NATIVE_STREAM_BACKENDS

    return ()


def default_backend_for_model(
    model_name: str,
    *,
    preferred_backend: str | None = None,
    model_path: str | None = None,
) -> BackendName:
    supported = infer_supported_backends(model_name, model_path=model_path)
    if preferred_backend in supported:
        return preferred_backend

    preset = resolve_model_preset(model_name)
    if preset is not None and preset.default_backend in supported:
        return preset.default_backend

    return supported[0]


def default_native_stream_backend_for_model(
    model_name: str,
    *,
    preferred_backend: str | None = None,
    model_path: str | None = None,
) -> NativeStreamBackendName | None:
    supported = infer_native_stream_backends(model_name, model_path=model_path)
    if not supported:
        return None
    if preferred_backend in supported:
        return preferred_backend

    preset = resolve_native_stream_model_preset(model_name)
    if preset is not None and preset.default_backend in supported:
        return preset.default_backend

    return supported[0]


def describe_model_experiment(model_name: str, *, model_path: str | None = None) -> str:
    preset = resolve_model_preset(model_name)
    if preset is not None:
        backends = ", ".join(preset.supported_backends)
        return f"{preset.label} | backends: {backends} | {preset.note}"

    supported = ", ".join(infer_supported_backends(model_name, model_path=model_path))
    raw_name = model_name.strip()
    raw_path = (model_path or "").strip()
    probe = (raw_path or raw_name).lower()

    if "qwen" in probe and "asr" in probe:
        return (
            "Qwen ASR custom model | backends: qwen_transformers | "
            "chunk 예제에서는 transformers backend로 동작하며 native streaming은 별도 stream 예제를 사용"
        )
    if raw_name.strip().lower() in _WHISPER_NAMES or "whisper" in probe or "faster-whisper" in probe:
        return f"Whisper custom model | backends: {supported} | model id/path를 직접 실험 가능"

    display_name = raw_path or raw_name or "<empty>"
    if raw_path:
        display_name = str(Path(raw_path).name)
    return f"Custom model ({display_name}) | backends: {supported} | 모델 포맷에 맞는 backend를 선택하세요"


def describe_native_stream_experiment(model_name: str, *, model_path: str | None = None) -> str:
    preset = resolve_native_stream_model_preset(model_name)
    if preset is not None:
        backends = ", ".join(preset.supported_backends)
        return f"{preset.label} | backends: {backends} | {preset.note}"

    supported = infer_native_stream_backends(model_name, model_path=model_path)
    if not supported:
        return "이 모델은 현재 native streaming 예제에서 지원하지 않습니다"

    supported_text = ", ".join(supported)
    return (
        f"Native stream model | backends: {supported_text} | "
        "Qwen ASR transformers backend (Windows/Linux 지원)"
    )


__all__ = [
    "ALL_BACKENDS",
    "MODEL_PRESETS",
    "NATIVE_STREAM_MODEL_PRESETS",
    "QWEN_BACKENDS",
    "QWEN_NATIVE_STREAM_BACKENDS",
    "WHISPER_BACKENDS",
    "default_backend_for_model",
    "default_native_stream_backend_for_model",
    "describe_model_experiment",
    "describe_native_stream_experiment",
    "infer_native_stream_backends",
    "infer_supported_backends",
    "native_stream_model_names",
    "preset_model_names",
    "resolve_model_preset",
    "resolve_native_stream_model_preset",
]
