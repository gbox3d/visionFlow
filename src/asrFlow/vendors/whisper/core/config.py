from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path

# NeuroFlow/src/asrFlow/vendors/whisper/core/config.py -> NeuroFlow root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR = Path(os.environ.get("WHISPER_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HF_CACHE_DIR = MODEL_DIR / "huggingface"
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "hub")

CT2_CACHE_DIR = MODEL_DIR / "faster_whisper"
CT2_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DeviceSpec:
    type: str
    index: int | None = None


def _load_torch():
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def _ct2_cuda_available() -> bool:
    try:
        ct2 = importlib.import_module("ctranslate2")
    except Exception:
        return False

    get_cuda_count = getattr(ct2, "get_cuda_device_count", None)
    if not callable(get_cuda_count):
        return False

    try:
        return int(get_cuda_count()) > 0
    except Exception:
        return False


def _device_type(device: DeviceSpec | object) -> str:
    device_type = getattr(device, "type", None)
    if device_type is None:
        return str(device).strip().lower()
    return str(device_type).strip().lower()


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    return None if language.lower() in ("none", "auto") else language


def configure_hf_quiet_logging() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import logging

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    try:
        hf_utils = importlib.import_module("transformers.utils")
        hf_logging = getattr(hf_utils, "logging", None)
        if hf_logging is not None:
            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
    except Exception:
        pass


def get_device(preferred: str = "auto") -> DeviceSpec:
    requested = (preferred or "auto").strip().lower()
    if requested in ("", "auto"):
        return DeviceSpec("cuda", 0) if _ct2_cuda_available() else DeviceSpec("cpu")

    if requested.startswith("cuda"):
        if ":" in requested:
            _, _, idx_raw = requested.partition(":")
            try:
                return DeviceSpec("cuda", int(idx_raw))
            except ValueError:
                return DeviceSpec("cuda")
        return DeviceSpec("cuda")

    if requested == "cpu":
        return DeviceSpec("cpu")

    return DeviceSpec(requested)


def get_dtype(device: DeviceSpec | object, prefer_fp16: bool = True):
    torch = _load_torch()
    if torch is None:
        raise RuntimeError("HF backend requires torch, but torch is unavailable in this build.")

    device_type = _device_type(device)
    if device_type == "cpu":
        return torch.float32
    if not prefer_fp16:
        return torch.float32
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_compute_type(device: DeviceSpec | object, fp16: bool = True) -> str:
    device_type = _device_type(device)
    if device_type != "cuda":
        return "int8"

    if fp16:
        try:
            ct2 = importlib.import_module("ctranslate2")
            get_supported = getattr(ct2, "get_supported_compute_types", None)
            if callable(get_supported):
                supported = {str(x).lower() for x in get_supported("cuda")}
                if "float16" in supported:
                    return "float16"
        except Exception:
            pass
        return "float32"
    return "float32"


def get_device_info() -> dict:
    torch = _load_torch()
    cuda_available = _ct2_cuda_available()
    info = {
        "device": "cpu",
        "dtype": "float32",
        "cuda_available": cuda_available,
        "model_dir": str(MODEL_DIR),
        "hf_cache": str(HF_CACHE_DIR),
        "ct2_cache": str(CT2_CACHE_DIR),
    }
    if torch is not None and torch.cuda.is_available():
        try:
            info["device"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
            info["bf16_support"] = torch.cuda.is_bf16_supported()
        except Exception:
            pass
    elif cuda_available:
        info["device"] = "cuda"
    return info


WHISPER_MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}

CT2_MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large-v3",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def resolve_model_id(name_or_id: str) -> str:
    return WHISPER_MODELS.get(name_or_id, name_or_id)


def resolve_ct2_model_id(name_or_id: str) -> str:
    return CT2_MODELS.get(name_or_id, name_or_id)
