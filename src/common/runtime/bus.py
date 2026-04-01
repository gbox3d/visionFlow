from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple


class TopicBus:
    """
    최신값(latest) 유지형 토픽 버스 (멀티 구독 가능)

    - publish(topic, obj): 최신값 갱신 + version 증가
    - get_latest(topic): 최신값을 "소비 없이" 조회
    - wait_latest(topic, last_version): last_version 이후 새 데이터가 오면 반환
      -> 여러 스레드가 동일 토픽을 안전하게 구독 가능
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._cond: Dict[str, threading.Condition] = {}
        self._slots: Dict[str, Any] = {}
        self._ver: Dict[str, int] = {}

    def _ensure(self, topic: str) -> None:
        with self._lock:
            if topic not in self._cond:
                self._cond[topic] = threading.Condition(self._lock)
                self._slots[topic] = None
                self._ver[topic] = 0

    def publish(self, topic: str, obj: Any) -> int:
        self._ensure(topic)
        with self._lock:
            self._slots[topic] = obj
            self._ver[topic] += 1
            v = self._ver[topic]
            self._cond[topic].notify_all()
            return v

    def get_latest(self, topic: str) -> Optional[Any]:
        self._ensure(topic)
        with self._lock:
            return self._slots.get(topic)

    def get_version(self, topic: str) -> int:
        self._ensure(topic)
        with self._lock:
            return self._ver.get(topic, 0)

    def wait_latest(
        self,
        topic: str,
        last_version: int,
        timeout: float = 0.2,
    ) -> Tuple[Optional[Any], int]:
        """
        last_version 이후 새 데이터가 들어오면 (obj, new_version) 반환
        timeout 시 (None, last_version) 반환
        """
        self._ensure(topic)
        with self._lock:
            if self._ver[topic] != last_version:
                return self._slots[topic], self._ver[topic]

            ok = self._cond[topic].wait(timeout=timeout)
            if not ok:
                return None, last_version

            return self._slots[topic], self._ver[topic]
