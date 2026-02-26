from __future__ import annotations

import importlib
from typing import Literal

import numpy as np

from voiceFlow.vendors.miso_stt.core.types import Segment

BackendName = Literal["hf_generate", "hf_pipeline", "ct2"]

_BACKEND_CLASS_PATHS: dict[BackendName, tuple[str, str]] = {
    "ct2": ("voiceFlow.vendors.miso_stt.backends.ct2", "CT2Transcriber"),
    "hf_generate": ("voiceFlow.vendors.miso_stt.backends.hf_generate", "GenerateTranscriber"),
    "hf_pipeline": ("voiceFlow.vendors.miso_stt.backends.hf_pipeline", "PipelineTranscriber"),
}


class WhisperTranscriber:
    def __init__(
        self,
        backend: BackendName = "hf_generate",
        model_name: str = "small",
        model_path: str | None = None,
        device: str = "auto",
        fp16: bool = True,
        **backend_kwargs,
    ):
        self.backend_name = backend
        class_path = _BACKEND_CLASS_PATHS.get(backend)
        if class_path is None:
            raise ValueError(f"Unsupported backend: {backend}")

        module_name, class_name = class_path
        try:
            backend_module = importlib.import_module(module_name)
            backend_cls = getattr(backend_module, class_name)
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                f"STT backend '{backend}' is unavailable in this build "
                f"({module_name}.{class_name} import failed: {exc})"
            ) from exc

        self.backend = backend_cls(
            model_name=model_name,
            model_path=model_path,
            device=device,
            fp16=fp16,
            **backend_kwargs,
        )

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def transcribe_full(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str | None = "ko",
        task: str = "transcribe",
    ) -> str:
        if hasattr(self.backend, "transcribe_full"):
            return self.backend.transcribe_full(waveform=waveform, sample_rate=sample_rate, language=language, task=task)
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
        return self.backend.transcribe_full_timeline(
            waveform=waveform,
            sample_rate=sample_rate,
            language=language,
            task=task,
        )

    def print_model_info(self) -> None:
        self.backend.print_model_info()

