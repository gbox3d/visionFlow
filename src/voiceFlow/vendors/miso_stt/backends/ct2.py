from __future__ import annotations

from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from voiceFlow.vendors.miso_stt.core.config import CT2_CACHE_DIR, MODEL_DIR, get_compute_type, get_device, resolve_ct2_model_id
from voiceFlow.vendors.miso_stt.core.filters import is_repetition_hallucination
from voiceFlow.vendors.miso_stt.core.model_resolver import _resolve_local_ct2_model, describe_ct2_model_path_issue
from voiceFlow.vendors.miso_stt.core.types import Segment


class CT2Transcriber:
    def __init__(
        self,
        model_name: str = "small",
        model_path: str | None = None,
        device: str = "auto",
        fp16: bool = True,
        beam_size: int = 5,
        temperature: float = 0.0,
        vad_filter: bool = True,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float | None = None,
        log_prob_threshold: float | None = None,
        compression_ratio_threshold: float | None = None,
    ):
        self.input_model_name = model_name
        self.requested_model_path = str(Path(model_path).expanduser()) if model_path else None
        self.requested_device = device
        self.requested_fp16 = fp16
        self.beam_size = int(beam_size)
        self.temperature = float(temperature)
        self.vad_filter = bool(vad_filter)
        self.condition_on_previous_text = bool(condition_on_previous_text)
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        torch_device = get_device(device)
        compute_type = get_compute_type(torch_device, fp16=fp16)

        if model_path:
            explicit_path = Path(model_path).expanduser()
            resolved_local = _resolve_local_ct2_model(explicit_path)
            if resolved_local is None:
                reason = describe_ct2_model_path_issue(explicit_path)
                raise ValueError(f"Invalid CT2 model_path '{explicit_path}': {reason}")

            model_id = str(resolved_local)
            download_root = None
            local_files_only = True
            load_source = "explicit_model_path"
            resolved_local_path = resolved_local
        else:
            model_name_path = Path(model_name).expanduser()
            resolved_local = _resolve_local_ct2_model(model_name_path)

            if resolved_local is not None:
                model_id = str(resolved_local)
                download_root = None
                local_files_only = True
                load_source = "model_name_local_path"
                resolved_local_path = resolved_local
            else:
                model_id = resolve_ct2_model_id(model_name)
                download_root = str(CT2_CACHE_DIR)
                local_files_only = False
                load_source = "hf_hub_or_cache"
                resolved_local_path = None

        model = WhisperModel(
            model_id,
            device=torch_device.type,
            compute_type=compute_type,
            download_root=download_root,
            local_files_only=local_files_only,
        )

        self.model = model
        self.model_id = model_id
        self.device = torch_device
        self.compute_type = compute_type
        self.download_root = download_root
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
        transcribe_kwargs = {
            "audio": waveform,
            "language": language,
            "task": task,
            "beam_size": self.beam_size,
            "temperature": self.temperature,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
        }
        if self.no_speech_threshold is not None:
            transcribe_kwargs["no_speech_threshold"] = float(self.no_speech_threshold)
        if self.log_prob_threshold is not None:
            transcribe_kwargs["log_prob_threshold"] = float(self.log_prob_threshold)
        if self.compression_ratio_threshold is not None:
            transcribe_kwargs["compression_ratio_threshold"] = float(self.compression_ratio_threshold)

        segments_gen, _info = self.model.transcribe(**transcribe_kwargs)

        segments: list[Segment] = []
        for seg in segments_gen:
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue
            if is_repetition_hallucination(seg_text):
                continue
            segments.append(Segment(round(float(seg.start), 2), round(float(seg.end), 2), seg_text))

        text = " ".join(seg.text for seg in segments).strip()
        return text, segments

    def print_model_info(self) -> None:
        print("=== CT2Transcriber Model Info ===")
        print(f"input model name:      {self.input_model_name}")
        print(f"requested model path:  {self.requested_model_path or '-'}")
        print(f"resolved model id:     {self.model_id}")
        print(f"effective model path:  {self.effective_model_path or '-'}")
        print(f"load source:           {self.load_source}")
        print(f"resolved local path:   {self.resolved_local_path or '-'}")
        print(f"local files only:      {self.local_files_only}")
        print(f"download root:         {self.download_root or '-'}")
        print(f"project model dir:     {MODEL_DIR}")
        print(f"CT2_CACHE_DIR:         {CT2_CACHE_DIR}")
        print(f"requested device:      {self.requested_device}")
        print(f"ct2 device:            {self.device.type}")
        print(f"requested fp16:        {self.requested_fp16}")
        print(f"compute type:          {self.compute_type}")
        print(f"beam size:             {self.beam_size}")
        print(f"temperature:           {self.temperature}")
        print(f"vad filter:            {self.vad_filter}")
        print(f"condition prev text:   {self.condition_on_previous_text}")
        print(f"no speech threshold:   {self.no_speech_threshold}")
        print(f"log prob threshold:    {self.log_prob_threshold}")
        print(f"compression ratio thr: {self.compression_ratio_threshold}")

