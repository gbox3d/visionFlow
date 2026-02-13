from __future__ import annotations

import time
import threading

from visionflow.pipeline.bus import TopicBus
from voiceFlow.sources.microphone_source import MicrophoneSource
from voiceFlow.pipeline.packet import AudioPacket
from typing import Optional


class AudioTestApplication:
    """
    마이크 입력을 테스트하는 간단한 애플리케이션
    """

    def __init__(self):
        self.bus = TopicBus()
        self.mic_source = MicrophoneSource(bus=self.bus, out_topic="audio/raw")
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None

    def _consumer_loop(self):
        last_version = 0
        while self._running:
            # Use a short timeout to allow for graceful shutdown
            packet, new_version = self.bus.wait_latest("audio/raw", last_version, timeout=0.1)
            if packet is not None and new_version != last_version:
                print(
                    f"[AudioListener] Topic: audio/raw, "
                    f"Source ID: {packet.source_id}, "
                    f"Seq: {packet.seq}, "
                    f"Timestamp: {packet.ts_ms}, "
                    f"Audio Shape: {packet.audio.shape}"
                )
                last_version = new_version

    def start(self):
        self.mic_source.start()
        self._running = True
        self._consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self._consumer_thread.start()
        print("[AudioTestApplication] 마이크 테스트 애플리케이션 시작. Ctrl+C를 눌러 종료하세요.")
        try:
            while self._running:
                time.sleep(1) # Keep main thread alive
        except KeyboardInterrupt:
            print("[AudioTestApplication] 종료 요청 받음.")
        finally:
            self.stop()

    def stop(self):
        if self._running:
            self._running = False
            self.mic_source.stop()
            if self._consumer_thread:
                self._consumer_thread.join(timeout=1.0) # Wait for consumer thread to finish
            # Note: TopicBus.shutdown() is not present in the provided bus.py
            # If resource cleanup is needed for TopicBus, it should be added there.
            print("[AudioTestApplication] 마이크 테스트 애플리케이션 종료.")


def main():
    app = AudioTestApplication()
    app.start()


if __name__ == "__main__":
    main()
