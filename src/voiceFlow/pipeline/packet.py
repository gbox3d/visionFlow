from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import time
import numpy as np


@dataclass(frozen=True)
class AudioPacket:
    """
    오디오 전용 payload
    """
    audio: np.ndarray
    ts_ms: int
    seq: int
    source_id: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AsrResultPacket:
    """
    ASR 추론 결과 payload
    - text: 전체 인식 텍스트
    - segments: faster-whisper Segment 리스트 (start, end, text 등)
    - language: 감지된 언어 코드
    """
    text: str
    segments: List[Dict[str, Any]]
    language: str
    ts_ms: int
    seq: int
    source_id: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Packet:
    """
    VoiceFlow 표준 Bus 패킷 (유일)
    """
    topic: str
    data: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    seq: int = 0
    source_id: str = ""
    timestamp: float = field(default_factory=time.time)
