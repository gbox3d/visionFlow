from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

import numpy as np


@dataclass(frozen=True)
class AudioPacket:
    audio: np.ndarray
    ts_ms: int
    seq: int
    source_id: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AsrResultPacket:
    text: str
    segments: list[dict[str, Any]]
    language: str
    ts_ms: int
    seq: int
    source_id: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Packet:
    topic: str
    data: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    source_id: str = ""
    timestamp: float = field(default_factory=time.time)
