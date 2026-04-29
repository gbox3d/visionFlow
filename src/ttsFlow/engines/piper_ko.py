from __future__ import annotations

import contextlib
import io
import json
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ttsFlow.engines.base import TtsEngine, TtsEngineResult


class PiperKoreanTtsEngine(TtsEngine):
    name = "piper-ko"

    def __init__(
        self,
        *,
        model_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "cpu",
        samplerate: int | None = None,
        default_speed: float = 1.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else Path(f"{self.model_path}.json")
        self.device = device.strip().lower()
        self.default_speed = float(default_speed)
        self._samplerate_override = samplerate
        self._voice: Any | None = None
        self._phonemizer: Any | None = None
        self._config: Any | None = None
        self._phoneme_id_map: dict[str, list[int]] = {}
        self._actual_provider = "unloaded"

    def service_meta(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "model_path": str(self.model_path),
            "config_path": str(self.config_path),
            "device": self.device,
            "provider": self._actual_provider,
            "model_loaded": self._voice is not None,
        }

    def warmup(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._voice is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"missing TTS model: {self.model_path}. "
                "Run `uv run nf-tts-models-download` or set NF_TTS_MODEL_PATH."
            )
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"missing TTS model config: {self.config_path}. "
                "Run `uv run nf-tts-models-download` or set NF_TTS_CONFIG_PATH."
            )

        from piper.config import PiperConfig
        from piper.voice import PiperVoice

        config_dict = json.loads(self.config_path.read_text(encoding="utf-8"))
        # The Korean KSS model uses a Rust/gruut phonemizer marker that the Python
        # piper package cannot parse. We do Korean G2P ourselves and use Piper only
        # for ONNX synthesis, so the enum value is normalized here.
        python_config_dict = dict(config_dict)
        python_config_dict["phoneme_type"] = "espeak"
        config = PiperConfig.from_dict(python_config_dict)
        if self._samplerate_override:
            config.sample_rate = int(self._samplerate_override)

        providers = self._resolve_providers()
        session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._actual_provider = session.get_providers()[0] if session.get_providers() else providers[0]
        self._voice = PiperVoice(session=session, config=config)
        self._config = config
        self._phoneme_id_map = {str(k): list(v) for k, v in config.phoneme_id_map.items()}
        self._phonemizer = self._load_korean_phonemizer()

    def _resolve_providers(self) -> list[str]:
        available = set(ort.get_available_providers())
        if self.device in ("", "auto", "cpu"):
            return ["CPUExecutionProvider"]
        if self.device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "NF_TTS_DEVICE=cuda was requested but onnxruntime has no CUDAExecutionProvider. "
                    "Use cpu or install a compatible onnxruntime-gpu stack."
                )
            return ["CUDAExecutionProvider"]
        raise ValueError(f"unsupported NF_TTS_DEVICE: {self.device!r}")

    def _load_korean_phonemizer(self) -> Any:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from piper_plus_g2p import get_phonemizer

            return get_phonemizer("ko")

    def _phonemize(self, text: str) -> list[str]:
        if self._phonemizer is None:
            self._phonemizer = self._load_korean_phonemizer()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tokens = self._phonemizer.phonemize(text)
        return [str(token) for token in tokens]

    def _tokens_to_ids(self, tokens: list[str]) -> tuple[list[int], list[str]]:
        id_map = self._phoneme_id_map
        pad = id_map.get("_", [0])
        ids: list[int] = []
        missing: list[str] = []
        ids.extend(id_map.get("^", [1]))
        ids.extend(pad)

        for token in tokens:
            # The KSS config maps IPA codepoints directly, while some G2P tokens are
            # multi-codepoint strings such as "tɕ" or "kʰ". Feed known codepoints
            # individually and report the dropped diacritics in metadata.
            for char in token:
                if char in id_map:
                    ids.extend(id_map[char])
                    ids.extend(pad)
                else:
                    missing.append(char)

        ids.extend(id_map.get("$", [2]))
        return ids, missing

    def synthesize(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        speed: float = 1.0,
        audio_format: str = "wav",
    ) -> TtsEngineResult:
        if audio_format != "wav":
            raise ValueError(f"piper-ko only supports wav, got: {audio_format}")
        self._ensure_loaded()
        assert self._voice is not None
        assert self._config is not None

        speed = float(speed or self.default_speed)
        length_scale = 1.0 / max(0.25, min(4.0, speed))
        tokens = self._phonemize(text)
        phoneme_ids, missing = self._tokens_to_ids(tokens)

        from piper.config import SynthesisConfig

        t0 = time.perf_counter()
        audio = self._voice.phoneme_ids_to_audio(
            phoneme_ids,
            SynthesisConfig(speaker_id=speaker_id, length_scale=length_scale),
        )
        inference_ms = int((time.perf_counter() - t0) * 1000)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        max_value = float(np.max(np.abs(audio))) if audio.size else 0.0
        if max_value < 1e-8:
            pcm = np.zeros_like(audio, dtype=np.int16)
        else:
            pcm = np.clip(audio / max_value * 32767.0, -32767, 32767).astype(np.int16)

        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wav_file:
            wav_file.setframerate(int(self._config.sample_rate))
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(pcm.tobytes())

        duration_ms = int((len(pcm) / int(self._config.sample_rate)) * 1000) if pcm.size else 0
        rtf = (inference_ms / 1000.0) / (duration_ms / 1000.0) if duration_ms else 0.0
        return TtsEngineResult(
            audio_bytes=wav_bytes.getvalue(),
            samplerate=int(self._config.sample_rate),
            channels=1,
            duration_ms=duration_ms,
            meta={
                "tokens": len(tokens),
                "phoneme_ids": len(phoneme_ids),
                "missing_phonemes": sorted(set(missing)),
                "inference_ms": inference_ms,
                "rtf": rtf,
                "provider": self._actual_provider,
            },
        )
