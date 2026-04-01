from __future__ import annotations

import threading
import time
from typing import Optional, Protocol

import numpy as np

from common.contracts.packets import AsrResultPacket
from common.runtime.bus import TopicBus


class StreamingAsrProcessorProtocol(Protocol):
    def feed_audio(
        self,
        audio_chunk: np.ndarray,
        *,
        samplerate: int = 16000,
        seq: int = 0,
        source_id: str = "",
    ) -> Optional[AsrResultPacket]:
        ...

    def flush(self, *, seq: int = 0, source_id: str = "") -> Optional[AsrResultPacket]:
        ...

    def close(self) -> None:
        ...


class StreamingAsrWorker:
    def __init__(
        self,
        bus: TopicBus,
        processor: StreamingAsrProcessorProtocol,
        in_topic: str = "audio/raw",
        out_topic: str = "text/asr",
        samplerate: int = 16000,
        name: str = "streaming-asr-worker",
    ):
        self.bus = bus
        self.processor = processor
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.samplerate = int(samplerate)
        self.name = name

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_ver = 0
        self._seq = 0
        self._last_source_id = ""

        self._fps_t0 = time.time()
        self._fps_n = 0
        self._infer_fps = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_ver = self.bus.get_version(self.in_topic)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[StreamingAsrWorker] 시작: sr={self.samplerate}Hz")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        try:
            self._publish_result(
                self.processor.flush(seq=self._seq + 1, source_id=self._last_source_id),
                infer_ms=0.0,
            )
        except Exception as exc:
            print(f"[StreamingAsrWorker] flush 실패: {exc}")
        finally:
            try:
                self.processor.close()
            except Exception:
                pass
        print("[StreamingAsrWorker] 종료")

    def _publish_result(self, result: Optional[AsrResultPacket], *, infer_ms: float) -> None:
        if result is None:
            return

        enriched = AsrResultPacket(
            text=result.text,
            segments=result.segments,
            language=result.language,
            ts_ms=result.ts_ms,
            seq=result.seq,
            source_id=result.source_id,
            meta={
                **result.meta,
                "infer_fps": self._infer_fps,
                "infer_ms": infer_ms,
            },
        )
        self.bus.publish(self.out_topic, enriched)

    def _loop(self) -> None:
        while self._running:
            pkt, ver = self.bus.wait_latest(self.in_topic, self._last_ver, timeout=0.2)
            if pkt is None:
                continue
            self._last_ver = ver
            self._last_source_id = pkt.source_id

            audio = pkt.audio
            if audio.ndim > 1:
                audio = audio[:, 0]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            self._seq += 1
            t0 = time.time()
            try:
                result = self.processor.feed_audio(
                    audio,
                    samplerate=self.samplerate,
                    seq=self._seq,
                    source_id=pkt.source_id,
                )
            except Exception as exc:
                print(f"[StreamingAsrWorker] ASR 처리 실패(seq={self._seq}): {exc}")
                continue
            infer_ms = (time.time() - t0) * 1000.0

            self._fps_n += 1
            now = time.time()
            elapsed = now - self._fps_t0
            if elapsed >= 1.0:
                self._infer_fps = self._fps_n / elapsed
                self._fps_n = 0
                self._fps_t0 = now

            self._publish_result(result, infer_ms=infer_ms)
