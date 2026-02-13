from __future__ import annotations

from typing import Literal

import numpy as np

from voiceFlow.vendors.miso_stt.backends.ct2 import CT2Transcriber
from voiceFlow.vendors.miso_stt.backends.hf_generate import GenerateTranscriber
from voiceFlow.vendors.miso_stt.backends.hf_pipeline import PipelineTranscriber
from voiceFlow.vendors.miso_stt.core.types import Segment

BackendName = Literal["hf_generate", "hf_pipeline", "ct2"]


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
        if backend == "hf_generate":
            self.backend = GenerateTranscriber(
                model_name=model_name,
                model_path=model_path,
                device=device,
                fp16=fp16,
                **backend_kwargs,
            )
        elif backend == "hf_pipeline":
            self.backend = PipelineTranscriber(
                model_name=model_name,
                model_path=model_path,
                device=device,
                fp16=fp16,
                **backend_kwargs,
            )
        elif backend == "ct2":
            self.backend = CT2Transcriber(
                model_name=model_name,
                model_path=model_path,
                device=device,
                fp16=fp16,
                **backend_kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

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

