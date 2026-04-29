from __future__ import annotations

from pathlib import Path

from ttsFlow.engines.base import TtsEngine
from ttsFlow.engines.piper_ko import PiperKoreanTtsEngine
from ttsFlow.engines.speecht5_ko import SpeechT5KoreanTtsEngine
from ttsFlow.engines.stub import StubTtsEngine


def build_tts_engine(
    *,
    engine: str,
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    model_id: str | None = None,
    device: str = "cpu",
    speed: float = 1.0,
) -> TtsEngine:
    engine_key = engine.strip().lower()
    if engine_key == "stub":
        return StubTtsEngine()
    if engine_key in ("piper", "piper-ko", "piper_korean"):
        if model_path is None:
            raise ValueError("piper-ko requires model_path")
        return PiperKoreanTtsEngine(
            model_path=model_path,
            config_path=config_path,
            device=device,
            default_speed=speed,
        )
    if engine_key in ("speecht5", "speecht5-ko", "speecht5_korean"):
        return SpeechT5KoreanTtsEngine(
            model_id=model_id or "ahnhs2k/speecht5-korean",
            device=device,
        )
    raise ValueError(f"unsupported TTS engine: {engine!r}")
