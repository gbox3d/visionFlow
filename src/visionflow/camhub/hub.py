"""
nf-vision-camhub: Camera Image Hub

Multiple camera clients push JPEG frames identified by a unique name.
AI clients fetch the latest frame by camera name.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CameraFrame:
    """Single stored frame for a camera."""

    name: str
    jpeg: bytes
    width: int = 0
    height: int = 0
    timestamp: float = field(default_factory=time.time)
    seq: int = 0
    meta: Dict[str, object] = field(default_factory=dict)


class FrameHub:
    """Thread-safe in-memory store that keeps the latest frame per camera name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: Dict[str, CameraFrame] = {}
        self._seq_counters: Dict[str, int] = {}

    def put(
        self,
        name: str,
        jpeg: bytes,
        width: int = 0,
        height: int = 0,
        meta: Optional[Dict[str, object]] = None,
    ) -> CameraFrame:
        with self._lock:
            seq = self._seq_counters.get(name, 0) + 1
            self._seq_counters[name] = seq
            frame = CameraFrame(
                name=name,
                jpeg=jpeg,
                width=width,
                height=height,
                timestamp=time.time(),
                seq=seq,
                meta=meta or {},
            )
            self._frames[name] = frame
            return frame

    def get(self, name: str) -> Optional[CameraFrame]:
        with self._lock:
            return self._frames.get(name)

    def list_cameras(self) -> List[Dict[str, object]]:
        with self._lock:
            result = []
            for name, frame in self._frames.items():
                result.append(
                    {
                        "name": name,
                        "width": frame.width,
                        "height": frame.height,
                        "seq": frame.seq,
                        "timestamp": frame.timestamp,
                        "age_s": round(time.time() - frame.timestamp, 2),
                    }
                )
            return result

    def remove(self, name: str) -> bool:
        with self._lock:
            removed = self._frames.pop(name, None)
            self._seq_counters.pop(name, None)
            return removed is not None
