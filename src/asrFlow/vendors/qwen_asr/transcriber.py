from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from asrFlow.vendors.whisper.core.config import HF_CACHE_DIR, get_device, get_dtype
from asrFlow.vendors.whisper.core.types import Segment

_LANGUAGE_ALIASES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    raw = language.strip()
    if not raw:
        return None

    lowered = raw.lower()
    if lowered in {"auto", "none"}:
        return None
    return _LANGUAGE_ALIASES.get(lowered, raw)


def _flatten_timestamps(time_stamps: Any) -> list[Any]:
    if time_stamps is None:
        return []

    if isinstance(time_stamps, (list, tuple)):
        flattened: list[Any] = []
        for item in time_stamps:
            flattened.extend(_flatten_timestamps(item))
        return flattened

    return [time_stamps]


class QwenAsrTranscriber:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        model_path: str | None = None,
        device: str = "auto",
        fp16: bool = True,
        max_inference_batch_size: int = 8,
        max_new_tokens: int = 256,
        forced_aligner: str | None = None,
        return_time_stamps: bool = False,
    ):
        from qwen_asr import Qwen3ASRModel

        token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
        if token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token

        checkpoint = str(Path(model_path).expanduser()) if model_path else model_name
        torch_device = get_device(device)
        device_map = "cpu"
        if torch_device.type == "cuda":
            device_map = f"cuda:{torch_device.index if torch_device.index is not None else 0}"
        elif torch_device.type != "cpu":
            device_map = str(device)

        torch_dtype = get_dtype(torch_device, prefer_fp16=fp16)

        # Keep Hugging Face cache under the same NeuroFlow model root used by Whisper backends.
        os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "hub"))

        self.input_model_name = model_name
        self.requested_model_path = str(Path(model_path).expanduser()) if model_path else None
        self.requested_device = device
        self.requested_fp16 = fp16
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.requested_forced_aligner = forced_aligner
        self.return_time_stamps = bool(return_time_stamps)
        self.model_id = checkpoint
        self.device = device_map
        self.compute_type = str(torch_dtype).replace("torch.", "")
        self.load_source = "explicit_model_path" if model_path else "hf_hub_or_cache"
        self.resolved_local_path = checkpoint if model_path else None
        self.effective_model_path = checkpoint if model_path else None
        self.cache_dir = str(HF_CACHE_DIR)

        forced_aligner_kwargs = None
        if forced_aligner:
            forced_aligner_kwargs = {
                "dtype": torch_dtype,
                "device_map": device_map,
            }

        self.model = Qwen3ASRModel.from_pretrained(
            checkpoint,
            dtype=torch_dtype,
            device_map=device_map,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=forced_aligner_kwargs,
            max_inference_batch_size=self.max_inference_batch_size,
            max_new_tokens=self.max_new_tokens,
        )
        self.backend = self

    def transcribe_full(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str | None = "ko",
        task: str = "transcribe",
    ) -> str:
        text, _ = self.transcribe_full_timeline(
            waveform=waveform,
            sample_rate=sample_rate,
            language=language,
            task=task,
        )
        return text

    def transcribe_full_timeline(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str | None = "ko",
        task: str = "transcribe",
    ) -> tuple[str, list[Segment]]:
        del task  # Qwen3-ASR transcribe API does not expose task switching like Whisper.

        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1)

        results = self.model.transcribe(
            audio=(audio, int(sample_rate)),
            language=_normalize_language(language),
            return_time_stamps=self.return_time_stamps,
        )
        if not results:
            return "", []

        result = results[0]
        text = (getattr(result, "text", "") or "").strip()
        segments: list[Segment] = []
        for item in _flatten_timestamps(getattr(result, "time_stamps", None)):
            start = getattr(item, "start_time", None)
            end = getattr(item, "end_time", None)
            seg_text = (getattr(item, "text", "") or "").strip()
            if start is None or end is None or not seg_text:
                continue
            segments.append(Segment(round(float(start), 2), round(float(end), 2), seg_text))

        if segments and not text:
            text = " ".join(seg.text for seg in segments).strip()

        return text, segments

    def print_model_info(self) -> None:
        print("=== QwenAsrTranscriber Model Info ===")
        print(f"input model name:      {self.input_model_name}")
        print(f"requested model path:  {self.requested_model_path or '-'}")
        print(f"resolved model id:     {self.model_id}")
        print(f"effective model path:  {self.effective_model_path or '-'}")
        print(f"load source:           {self.load_source}")
        print(f"cache dir used:        {self.cache_dir}")
        print(f"requested device:      {self.requested_device}")
        print(f"device map:            {self.device}")
        print(f"dtype:                 {self.compute_type}")
        print(f"max inference batch:   {self.max_inference_batch_size}")
        print(f"max new tokens:        {self.max_new_tokens}")
        print(f"forced aligner:        {self.requested_forced_aligner or '-'}")
