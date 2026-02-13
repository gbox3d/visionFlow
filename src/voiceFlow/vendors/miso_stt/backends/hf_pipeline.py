from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import torch
from transformers import pipeline

from voiceFlow.vendors.miso_stt.backends.hf_generate import GenerateTranscriber
from voiceFlow.vendors.miso_stt.core.audio import sliding_windows
from voiceFlow.vendors.miso_stt.core.filters import is_repetition_hallucination
from voiceFlow.vendors.miso_stt.core.types import Segment


class PipelineTranscriber:
    def __init__(
        self,
        model_name: str = "small",
        model_path: str | None = None,
        device: str = "auto",
        fp16: bool = True,
    ):
        self.input_model_name = model_name
        self.requested_model_path = model_path
        self.requested_device = device
        self.requested_fp16 = fp16
        generate_backend = GenerateTranscriber(model_name=model_name, model_path=model_path, device=device, fp16=fp16)
        model_dtype = next(generate_backend.model.parameters()).dtype
        asr = pipeline(
            task="automatic-speech-recognition",
            model=generate_backend.model,
            tokenizer=generate_backend.processor.tokenizer,
            feature_extractor=generate_backend.processor.feature_extractor,
            device=generate_backend.torch_device,
            dtype=model_dtype,
        )

        self.asr = asr
        self.model_id = generate_backend.model_id
        self.backend = generate_backend
        self.model_dtype = model_dtype

    def print_model_info(self) -> None:
        total_params = sum(p.numel() for p in self.backend.model.parameters())
        trainable_params = sum(p.numel() for p in self.backend.model.parameters() if p.requires_grad)

        print("=== PipelineTranscriber Model Info ===")
        print(f"input model name:      {self.input_model_name}")
        print(f"requested model path:  {self.requested_model_path or '-'}")
        print(f"resolved model id:     {self.model_id}")
        print(f"effective model path:  {self.backend.effective_model_path or '-'}")
        print(f"pipeline task:         automatic-speech-recognition")
        print(f"pipeline impl:         {type(self.asr).__name__}")
        print(f"load source:           {self.backend.load_source}")
        print(f"resolved local path:   {self.backend.resolved_local_path or '-'}")
        print(f"local files only:      {self.backend.local_files_only}")
        print(f"cache dir used:        {self.backend.cache_dir or '-'}")
        print(f"HF_HOME env:           {os.environ.get('HF_HOME', '-')}")
        print(f"TRANSFORMERS_CACHE env:{os.environ.get('TRANSFORMERS_CACHE', '-')}")
        print(f"requested device:      {self.requested_device}")
        print(f"torch device:          {self.backend.torch_device}")
        print(f"requested fp16:        {self.requested_fp16}")
        print(f"model param dtype:     {self.model_dtype}")
        print(f"total params:          {total_params:,}")
        print(f"trainable params:      {trainable_params:,}")
        if self.backend.torch_device.type == "cuda":
            idx = self.backend.torch_device.index if self.backend.torch_device.index is not None else 0
            print(f"cuda device name:      {torch.cuda.get_device_name(idx)}")
            props = torch.cuda.get_device_properties(idx)
            print(f"cuda total memory GB:  {props.total_memory / 1e9:.2f}")

    def _transcribe_chunk(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        generate_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return self.asr(
            {"array": waveform, "sampling_rate": sample_rate},
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

    def transcribe_full(self, waveform: np.ndarray, sample_rate: int, language: str | None = "ko", task: str = "transcribe") -> str:
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
        generate_kwargs: dict[str, Any] = {}
        if language is not None:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = task

        out = self._transcribe_chunk(
            waveform=waveform,
            sample_rate=sample_rate,
            generate_kwargs=generate_kwargs,
        )

        segments: list[Segment] = []
        for ch in out.get("chunks", []):
            ts = ch.get("timestamp")
            seg_text = (ch.get("text") or "").strip()
            if not ts or len(ts) != 2 or not seg_text:
                continue
            if is_repetition_hallucination(seg_text):
                continue

            start = float(ts[0] or 0.0)
            end = float(ts[1] or ts[0] or 0.0)
            segments.append(Segment(round(start, 2), round(end, 2), seg_text))

        fulltext = " ".join(seg.text for seg in segments).strip()
        if not fulltext:
            fulltext = (out.get("text") or "").strip()
        return fulltext, segments

    def transcribe_streaming(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str | None = "ko",
        task: str = "transcribe",
        window_sec: float = 5.0,
        overlap_sec: float = 2.0,
        on_chunk: Callable[[float, str], Any] | None = None,
    ) -> tuple[str, list[Segment]]:
        all_segments: list[Segment] = []
        last_end = 0.0

        for chunk, offset in sliding_windows(
            waveform=waveform,
            sample_rate=sample_rate,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
        ):
            generate_kwargs: dict[str, Any] = {}
            if language is not None:
                generate_kwargs["language"] = language
                generate_kwargs["task"] = task

            out = self._transcribe_chunk(
                waveform=chunk,
                sample_rate=sample_rate,
                generate_kwargs=generate_kwargs,
            )
            chunk_kept_texts: list[str] = []

            for ch in out.get("chunks", []):
                ts = ch.get("timestamp")
                seg_text = (ch.get("text") or "").strip()
                if not ts or len(ts) != 2 or not seg_text:
                    continue
                if is_repetition_hallucination(seg_text):
                    continue

                start = float(ts[0] or 0.0) + offset
                end = float(ts[1] or ts[0] or 0.0) + offset
                if start < (last_end - 0.20):
                    continue

                last_end = max(last_end, end)
                all_segments.append(Segment(round(start, 2), round(end, 2), seg_text))
                chunk_kept_texts.append(seg_text)

            if chunk_kept_texts and on_chunk is not None:
                on_chunk(offset, " ".join(chunk_kept_texts))

        final_text = " ".join(seg.text for seg in all_segments).strip()
        return final_text, all_segments

