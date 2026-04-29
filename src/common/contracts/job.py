from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobState(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class JobRequest:
    service: str
    command: str
    request_id: int
    session_id: int = 0
    timeout_ms: int = 30_000
    meta: dict[str, Any] = field(default_factory=dict)
    data: bytes = b""


@dataclass(slots=True)
class JobResult:
    service: str
    command: str
    request_id: int
    session_id: int
    state: JobState
    meta: dict[str, Any] = field(default_factory=dict)
    data: bytes = b""
