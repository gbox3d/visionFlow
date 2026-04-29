from __future__ import annotations

import io
import time
import unicodedata
import wave
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    PreTrainedTokenizerFast,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)

from ttsFlow.engines.base import TtsEngine, TtsEngineResult


class SpeechT5KoreanTtsEngine(TtsEngine):
    name = "speecht5-ko"

    def __init__(
        self,
        *,
        model_id: str = "ahnhs2k/speecht5-korean",
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device_name = device.strip().lower()
        self.device = self._resolve_device(self.device_name)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._vocoder: Any | None = None
        self._speaker_embedding: torch.Tensor | None = None
        self.samplerate = 16000

    def service_meta(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "model_id": self.model_id,
            "device": str(self.device),
            "model_loaded": self._model is not None,
            "samplerate": self.samplerate,
        }

    def warmup(self) -> None:
        self._ensure_loaded()

    def _resolve_device(self, requested: str) -> torch.device:
        if requested in ("", "auto"):
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("NF_TTS_DEVICE=cuda requested but CUDA is not available")
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")
        raise ValueError(f"unsupported NF_TTS_DEVICE: {requested!r}")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._model = SpeechT5ForTextToSpeech.from_pretrained(self.model_id).to(self.device).eval()
        self._tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_id)
        self._vocoder = SpeechT5HifiGan.from_pretrained(self.model_id, subfolder="vocoder").to(self.device).eval()
        speaker_path = hf_hub_download(self.model_id, "speaker_embedding.pth")
        self._speaker_embedding = torch.load(speaker_path, map_location=self.device)

    def synthesize(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        speed: float = 1.0,
        audio_format: str = "wav",
    ) -> TtsEngineResult:
        if audio_format != "wav":
            raise ValueError(f"speecht5-ko only supports wav, got: {audio_format}")
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._vocoder is not None
        assert self._speaker_embedding is not None

        encoded = self._tokenizer(
            _decompose_jamo(text),
            is_split_into_words=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        speaker_embedding = self._speaker_embedding.to(self.device).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.inference_mode():
            audio = self._model.generate_speech(
                input_ids,
                speaker_embeddings=speaker_embedding,
                vocoder=self._vocoder,
            )
        inference_ms = int((time.perf_counter() - t0) * 1000)

        audio_np = audio.detach().cpu().numpy().astype(np.float32)
        pcm = np.clip(audio_np * 32767.0, -32767, 32767).astype(np.int16)
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wav_file:
            wav_file.setframerate(self.samplerate)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(pcm.tobytes())

        duration_ms = int((len(pcm) / self.samplerate) * 1000) if pcm.size else 0
        rtf = (inference_ms / 1000.0) / (duration_ms / 1000.0) if duration_ms else 0.0
        return TtsEngineResult(
            audio_bytes=wav_bytes.getvalue(),
            samplerate=self.samplerate,
            channels=1,
            duration_ms=duration_ms,
            meta={
                "model_id": self.model_id,
                "device": str(self.device),
                "inference_ms": inference_ms,
                "rtf": rtf,
                "speed_supported": False,
            },
        )


def _decompose_jamo(text: str) -> list[str]:
    result: list[str] = []
    for char in text:
        name = unicodedata.name(char, "")
        if "HANGUL SYLLABLE" not in name:
            result.append(char)
            continue

        code = ord(char) - 0xAC00
        result.append(chr(0x1100 + (code // 588)))
        result.append(chr(0x1161 + ((code % 588) // 28)))
        jong = code % 28
        if jong > 0:
            result.append(chr(0x11A7 + jong))

    return result
