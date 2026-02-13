from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from voiceFlow.vendors.miso_stt.core.config import HF_CACHE_DIR, MODEL_DIR, get_device, get_dtype, resolve_model_id
from voiceFlow.vendors.miso_stt.core.filters import is_repetition_hallucination
from voiceFlow.vendors.miso_stt.core.model_resolver import _resolve_local_hf_model, describe_hf_model_path_issue
from voiceFlow.vendors.miso_stt.core.types import Segment


class GenerateTranscriber:
    def __init__(
        self,
        model_name: str = "small",
        model_path: str | None = None,
        device: str = "auto",
        fp16: bool = True,
    ):
        self.input_model_name = model_name
        self.requested_model_path = str(Path(model_path).expanduser()) if model_path else None
        self.requested_device = device
        self.requested_fp16 = fp16
        torch_device = get_device(device)
        torch_dtype = get_dtype(torch_device, prefer_fp16=fp16)

        if model_path:
            explicit_path = Path(model_path).expanduser()
            resolved_local = _resolve_local_hf_model(explicit_path)
            if resolved_local is None:
                reason = describe_hf_model_path_issue(explicit_path)
                raise ValueError(f"Invalid HF model_path '{explicit_path}': {reason}")

            model_id = str(resolved_local)
            cache_dir = None
            local_files_only = True
            load_source = "explicit_model_path"
            resolved_local_path = resolved_local
        else:
            model_name_path = Path(model_name).expanduser()
            resolved_local = _resolve_local_hf_model(model_name_path)

            if resolved_local is not None:
                model_id = str(resolved_local)
                cache_dir = None
                local_files_only = True
                load_source = "model_name_local_path"
                resolved_local_path = resolved_local
            else:
                model_id = resolve_model_id(model_name)
                cache_dir = str(HF_CACHE_DIR)
                local_files_only = False
                load_source = "hf_hub_or_cache"
                resolved_local_path = None

        processor = WhisperProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        model.to(torch_device)
        model.eval()

        self.processor = processor
        self.model = model
        self.torch_device = torch_device
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.model_dtype = next(model.parameters()).dtype
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.load_source = load_source
        self.resolved_local_path = resolved_local_path
        self.effective_model_path = str(resolved_local_path) if resolved_local_path is not None else None

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
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            truncation=False,
            padding="max_length",
        )

        input_features = inputs.input_features.to(device=self.torch_device, dtype=self.model_dtype)

        generate_kwargs = {
            "return_timestamps": True,
            "return_segments": True,
        }
        if language is not None:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = task

        with torch.no_grad():
            output = self.model.generate(input_features, **generate_kwargs)

        segments: list[Segment] = []
        for seg_group in output.get("segments", []):
            for seg in seg_group:
                text = self.processor.tokenizer.decode(seg["tokens"].tolist(), skip_special_tokens=True).strip()
                if text and not is_repetition_hallucination(text):
                    start = float(seg["start"].item())
                    end = float(seg["end"].item())
                    segments.append(Segment(round(start, 2), round(end, 2), text))

        fulltext = " ".join(seg.text for seg in segments).strip()
        if not fulltext:
            fulltext = self.processor.batch_decode(
                output["sequences"],
                skip_special_tokens=True,
            )[0].strip()

        return fulltext, segments

    def print_model_info(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("=== GenerateTranscriber Model Info ===")
        print(f"input model name:      {self.input_model_name}")
        print(f"requested model path:  {self.requested_model_path or '-'}")
        print(f"resolved model id:     {self.model_id}")
        print(f"effective model path:  {self.effective_model_path or '-'}")
        print(f"load source:           {self.load_source}")
        print(f"resolved local path:   {self.resolved_local_path or '-'}")
        print(f"local files only:      {self.local_files_only}")
        print(f"cache dir used:        {self.cache_dir or '-'}")
        print(f"project model dir:     {MODEL_DIR}")
        print(f"HF_CACHE_DIR:          {HF_CACHE_DIR}")
        print(f"HF_HOME env:           {os.environ.get('HF_HOME', '-')}")
        print(f"TRANSFORMERS_CACHE env:{os.environ.get('TRANSFORMERS_CACHE', '-')}")
        print(f"requested device:      {self.requested_device}")
        print(f"torch device:          {self.torch_device}")
        print(f"requested fp16:        {self.requested_fp16}")
        print(f"load torch dtype:      {self.torch_dtype}")
        print(f"model param dtype:     {self.model_dtype}")
        print(f"total params:          {total_params:,}")
        print(f"trainable params:      {trainable_params:,}")
        if self.torch_device.type == "cuda":
            idx = self.torch_device.index if self.torch_device.index is not None else 0
            print(f"cuda device name:      {torch.cuda.get_device_name(idx)}")
            props = torch.cuda.get_device_properties(idx)
            print(f"cuda total memory GB:  {props.total_memory / 1e9:.2f}")

