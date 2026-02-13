from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers.utils import logging as hf_logging

# NeuroFlow/src/voiceFlow/vendors/miso_stt/core/config.py -> NeuroFlow root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
load_dotenv(PROJECT_ROOT / ".env", override=False)

MODEL_DIR = Path(os.environ.get("WHISPER_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HF_CACHE_DIR = MODEL_DIR / "huggingface"
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "hub")

CT2_CACHE_DIR = MODEL_DIR / "faster_whisper"
CT2_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    return None if language.lower() in ("none", "auto") else language


def configure_hf_quiet_logging() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import logging

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preferred)


def get_dtype(device: torch.device, prefer_fp16: bool = True) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if not prefer_fp16:
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_compute_type(device: torch.device, fp16: bool = True) -> str:
    if device.type == "cpu":
        return "int8"
    if fp16:
        return "float16"
    return "float32"


def get_device_info() -> dict:
    info = {
        "device": "cpu",
        "dtype": "float32",
        "cuda_available": torch.cuda.is_available(),
        "model_dir": str(MODEL_DIR),
        "hf_cache": str(HF_CACHE_DIR),
        "ct2_cache": str(CT2_CACHE_DIR),
    }
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
        info["bf16_support"] = torch.cuda.is_bf16_supported()
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

