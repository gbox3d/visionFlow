from __future__ import annotations

import threading
import time
from typing import Optional, Protocol

import numpy as np

from visionflow.pipeline.bus import TopicBus
from voiceFlow.pipeline.packet import AudioPacket, AsrResultPacket


class AsrProcessorProtocol(Protocol):
    def process(
        self,
        audio_chunk: np.ndarray,
        samplerate: int = 16000,
        seq: int = 0,
        source_id: str = "",
    ) -> Optional[AsrResultPacket]:
        ...

    def close(self) -> None:
        ...


class AsrWorker:
    """
    오디오 버퍼링 → ASR 추론 워커

    - SyncInferenceWorker 패턴과 동일
    - in_topic에서 AudioPacket을 구독
    - chunk_duration_s 만큼 오디오를 버퍼에 누적
    - 버퍼가 차면 WhisperAsrProcessor.process() 호출
    - 결과를 out_topic으로 publish

    렌더링과 완전 분리 (별도 Thread)
    """

    def __init__(
        self,
        bus: TopicBus,
        processor: AsrProcessorProtocol,
        in_topic: str = "audio/raw",
        out_topic: str = "text/asr",
        chunk_duration_s: float = 3.0,
        samplerate: int = 16000,
        name: str = "asr-worker",
    ):
        self.bus = bus
        self.processor = processor
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.chunk_duration_s = float(chunk_duration_s)
        self.samplerate = int(samplerate)
        self.name = name

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_ver = 0

        # 오디오 버퍼
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._chunk_samples = int(self.samplerate * self.chunk_duration_s)

        # 시퀀스
        self._seq = 0

        # infer fps (추론 횟수/초)
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
        print(
            f"[AsrWorker] 시작: chunk={self.chunk_duration_s}s "
            f"({self._chunk_samples} samples @ {self.samplerate}Hz)"
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        try:
            self.processor.close()
        except Exception:
            pass
        print("[AsrWorker] 종료")

    def _loop(self) -> None:
        source_id = ""

        while self._running:
            pkt, v = self.bus.wait_latest(self.in_topic, self._last_ver, timeout=0.2)
            if pkt is None:
                continue
            self._last_ver = v

            # AudioPacket에서 오디오 추출
            audio = pkt.audio
            source_id = pkt.source_id

            self._buffer.append(audio)
            self._buffer_samples += audio.shape[0]

            # 버퍼가 충분히 차면 추론
            if self._buffer_samples >= self._chunk_samples:
                chunk = np.concatenate(self._buffer, axis=0)

                # shape 정규화: (N, C) -> (N,) mono
                if chunk.ndim > 1:
                    chunk = chunk[:, 0]

                self._buffer.clear()
                self._buffer_samples = 0

                self._seq += 1

                t0 = time.time()
                try:
                    result = self.processor.process(
                        audio_chunk=chunk,
                        samplerate=self.samplerate,
                        seq=self._seq,
                        source_id=source_id,
                    )
                except Exception as e:
                    print(f"[AsrWorker] ASR 처리 실패(seq={self._seq}): {e}")
                    continue
                dt = time.time() - t0

                # infer fps
                self._fps_n += 1
                now = time.time()
                d = now - self._fps_t0
                if d >= 1.0:
                    self._infer_fps = self._fps_n / d
                    self._fps_n = 0
                    self._fps_t0 = now

                if result is not None:
                    # meta에 추론 성능 정보 추가
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
                            "infer_ms": dt * 1000.0,
                        },
                    )
                    self.bus.publish(self.out_topic, enriched)
