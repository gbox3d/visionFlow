from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional

import numpy as np
import sounddevice as sd

from visionflow.pipeline.bus import TopicBus
from voiceFlow.pipeline.packet import AudioPacket


class MicrophoneSource:
    """
    마이크 캡처 전용 소스
    - 오디오 데이터를 out_topic으로 계속 publish
    - 오디오 장치 연결 끊김 시 자동 재연결
    """

    def __init__(
        self,
        bus: TopicBus,
        out_topic: str = "audio/raw",
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 1024,
        device: Optional[int | str] = None,
        max_fail: int = 60,
        reconnect_sleep_s: float = 1.0,
        source_id: Optional[str] = None,
    ):
        self.bus = bus
        self.out_topic = out_topic
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self.device = device
        self.max_fail = int(max_fail)
        self.reconnect_sleep_s = float(reconnect_sleep_s)
        self.source_id = source_id or f"mic_{device if device is not None else 'default'}"

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._seq = 0
        self._last_ok_ts = 0
        self._mic_fps = 0.0 # Placeholder for audio "fps" equivalent (blocks per second)
        self._fps_t0 = time.time()
        self._fps_n = 0

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        if status:
            print(f"[MicrophoneSource] Audio callback status: {status}")

        if not self._running:
            return

        self._seq += 1
        self._last_ok_ts = int(time.time() * 1000)

        self._fps_n += 1
        now = time.time()
        dt = now - self._fps_t0
        if dt >= 1.0:
            self._mic_fps = self._fps_n / dt
            self._fps_n = 0
            self._fps_t0 = now

        pkt = AudioPacket(
            audio=indata.copy(),  # Copy to ensure immutability
            ts_ms=self._last_ok_ts,
            seq=self._seq,
            source_id=self.source_id,
            meta={
                "samplerate": self.samplerate,
                "channels": self.channels,
                "blocksize": self.blocksize,
                "device": self.device,
                "mic_fps": self._mic_fps,
            },
        )
        self.bus.publish(self.out_topic, pkt)

    def _open_stream(self) -> bool:
        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            print(
                f"[MicrophoneSource] 오디오 스트림 오픈 성공: "
                f"장치={self.device if self.device is not None else '기본'} "
                f"샘플링레이트={self.samplerate}Hz, 채널={self.channels}, 블록크기={self.blocksize}"
            )
            return True
        except Exception as e:
            print(f"[MicrophoneSource] 오디오 스트림 오픈 실패: {e}")
            self._stream = None
            return False

    def _close_stream(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print("[MicrophoneSource] 오디오 스트림 종료")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_stream()
        print("[MicrophoneSource] MicrophoneSource 종료")

    def _loop(self) -> None:
        fail = 0
        while self._running:
            if self._stream is None or not self._stream.active:
                self._close_stream()  # Ensure it's fully closed before attempting to reopen
                if self._open_stream():
                    fail = 0  # Reset fail count on successful open
                else:
                    fail += 1
                    if fail % 10 == 0:
                        print(f"[MicrophoneSource] 오디오 스트림 재연결 실패 {fail}회 누적")
                    if fail >= self.max_fail:
                        print("[MicrophoneSource] 스트림 연결 지속 실패 → 재시도 대기")
                        time.sleep(self.reconnect_sleep_s)
            else:
                # Stream is active, just keep the thread alive
                time.sleep(0.1)  # Small sleep to prevent busy-waiting

        self._close_stream()
