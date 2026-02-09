from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import time
import numpy as np

ROI = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass(frozen=True)
class FramePacket:
    """
    프레임 전용 payload
    """
    image: np.ndarray
    ts_ms: int
    seq: int
    source_id: str
    roi: Optional[ROI] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Packet:
    """
    VisionFlow 표준 Bus 패킷 (유일)
    """
    topic: str
    data: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    seq: int = 0
    source_id: str = ""
    timestamp: float = field(default_factory=time.time)
