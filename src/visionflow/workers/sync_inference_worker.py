from __future__ import annotations

import threading
import time
from typing import Optional, Protocol

from visionflow.pipeline.bus import TopicBus
from visionflow.pipeline.packet import FramePacket


class Processor(Protocol):
    def process(self, pkt: FramePacket) -> Optional[FramePacket]: ...
    def close(self) -> None: ...


class SyncInferenceWorker:
    """
    - CameraSource가 만든 프레임을 구독(in_topic)
    - processor.process() 실행
    - 결과를 out_topic으로 publish

    렌더링과 완전 분리 (별도 Thread)
    """

    def __init__(
        self,
        bus: TopicBus,
        in_topic: str,
        out_topic: str,
        processor: Processor,
        name: str = "inference",
        sleep_s: float = 0.0,
    ):
        self.bus = bus
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.processor = processor
        self.name = name
        self.sleep_s = float(sleep_s)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_ver = 0

        # infer fps
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._infer_fps = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._last_ver = self.bus.get_version(self.in_topic)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.processor.close()
        except Exception:
            pass

    def _loop(self):
        while self._running:
            pkt, v = self.bus.wait_latest(self.in_topic, self._last_ver, timeout=0.2)
            if pkt is None:
                continue
            self._last_ver = v

            t0 = time.time()
            out = self.processor.process(pkt)
            dt = time.time() - t0

            # infer fps
            self._fps_n += 1
            now = time.time()
            d = now - self._fps_t0
            if d >= 1.0:
                self._infer_fps = self._fps_n / d
                self._fps_n = 0
                self._fps_t0 = now

            if out is not None:
                out = FramePacket(
                    image=out.image,
                    ts_ms=out.ts_ms,
                    seq=out.seq,
                    source_id=out.source_id,
                    roi=out.roi,
                    meta={**out.meta, "infer_fps": self._infer_fps, "infer_ms": dt * 1000.0},
                )
                self.bus.publish(self.out_topic, out)

            if self.sleep_s > 0:
                time.sleep(self.sleep_s)
