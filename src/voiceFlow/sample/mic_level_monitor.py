"""
file : sample/mic_level_monitor.py

VoiceFlow - Microphone Level Monitor (PySide6)

Rules:
- This file NEVER performs inference.
- This file NEVER blocks on processing.
- This file ONLY renders what exists on the TopicBus.
- This file is SAFE to copy for other samples.

DO NOT edit this block.
"""

from __future__ import annotations

import sys
import time
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QColor, QPainter, QFont, QPen, QBrush, QLinearGradient
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from visionflow.pipeline.bus import TopicBus
from voiceFlow.sources.microphone_source import MicrophoneSource
from voiceFlow.pipeline.packet import AudioPacket


# ---------------------------------------------------------------------------
# Level Gauge Widget
# ---------------------------------------------------------------------------

class LevelGaugeWidget(QWidget):
    """
    실시간 음량 레벨을 표시하는 커스텀 위젯
    - 수평 바 형태
    - 녹색 → 노란색 → 빨간색 그라디언트
    - 피크 홀드 마커
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(36)
        self.setMinimumWidth(200)

        self._level: float = 0.0      # 0.0 ~ 1.0
        self._peak: float = 0.0       # 피크 홀드
        self._peak_ts: float = 0.0    # 피크 갱신 시각
        self._peak_hold_s: float = 1.5  # 피크 홀드 유지 시간

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))
        now = time.time()
        if self._level >= self._peak:
            self._peak = self._level
            self._peak_ts = now
        elif now - self._peak_ts > self._peak_hold_s:
            # 피크 서서히 감쇄
            self._peak = max(self._level, self._peak * 0.92)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 2
        bar_h = h - margin * 2
        bar_w = w - margin * 2

        # 배경
        p.fillRect(margin, margin, bar_w, bar_h, QColor(30, 30, 30))

        # 레벨 바 (그라디언트)
        level_w = int(bar_w * self._level)
        if level_w > 0:
            grad = QLinearGradient(margin, 0, margin + bar_w, 0)
            grad.setColorAt(0.0, QColor(0, 200, 0))
            grad.setColorAt(0.6, QColor(200, 200, 0))
            grad.setColorAt(1.0, QColor(220, 0, 0))
            p.fillRect(margin, margin, level_w, bar_h, QBrush(grad))

        # 피크 마커
        peak_x = margin + int(bar_w * self._peak)
        if peak_x > margin + 2:
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(peak_x, margin, peak_x, margin + bar_h)

        # dB 표시
        if self._level > 0:
            db = 20.0 * np.log10(self._level + 1e-10)
        else:
            db = -60.0
        db = max(-60.0, db)

        p.setPen(QColor(220, 220, 220))
        font = QFont("Consolas", 10)
        p.setFont(font)
        p.drawText(
            margin + 4, margin, bar_w - 8, bar_h,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{db:.1f} dB",
        )

        p.end()


# ---------------------------------------------------------------------------
# Bus → Qt Signal Bridge
# ---------------------------------------------------------------------------

class AudioBridge(QObject):
    """
    TopicBus에서 AudioPacket을 읽어 Qt Signal로 전달하는 브릿지
    (별도 스레드에서 동작)
    """

    level_updated = Signal(float)

    def __init__(self, bus: TopicBus, topic: str = "audio/raw"):
        super().__init__()
        self._bus = bus
        self._topic = topic
        self._running = False
        self._thread: Optional[threading.Thread] = None

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

    def _loop(self) -> None:
        last_ver = self._bus.get_version(self._topic)
        while self._running:
            pkt, new_ver = self._bus.wait_latest(self._topic, last_ver, timeout=0.1)
            if pkt is not None and new_ver != last_ver:
                last_ver = new_ver
                # RMS 계산
                rms = float(np.sqrt(np.mean(pkt.audio.astype(np.float64) ** 2)))
                # 0~1 범위로 정규화 (float32 기준 sounddevice 출력은 -1~1)
                level = min(1.0, rms * 3.0)  # 약간의 gain
                self.level_updated.emit(level)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MicLevelMonitorWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceFlow - Mic Level Monitor")
        self.setMinimumSize(480, 180)
        self.resize(520, 200)

        # --- 마이크 목록 구성 ---
        self._devices = self._list_input_devices()

        # --- UI ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 마이크 선택
        row = QHBoxLayout()
        row.addWidget(QLabel("마이크:"))
        self._combo = QComboBox()
        for dev in self._devices:
            api_name = sd.query_hostapis(dev["hostapi"])["name"]
            self._combo.addItem(f"{dev['index']}: {dev['name']}  ({api_name})")
        row.addWidget(self._combo, 1)
        layout.addLayout(row)

        # 상태 라벨
        self._status_label = QLabel("마이크를 선택하면 자동으로 모니터링이 시작됩니다.")
        self._status_label.setStyleSheet("color: #888;")
        layout.addWidget(self._status_label)

        # 레벨 게이지
        self._gauge = LevelGaugeWidget()
        layout.addWidget(self._gauge)

        # 스트레치
        layout.addStretch()

        # --- Pipeline ---
        self._bus = TopicBus()
        self._mic_source: Optional[MicrophoneSource] = None
        self._bridge = AudioBridge(self._bus, topic="audio/raw")
        self._bridge.level_updated.connect(self._on_level)

        # 콤보 변경 시 마이크 전환
        self._combo.currentIndexChanged.connect(self._on_device_changed)

        # 초기 선택이 있으면 시작
        if self._devices:
            self._on_device_changed(0)

    # ---- helpers ----

    @staticmethod
    def _list_input_devices() -> list:
        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                d = dict(dev)
                d["index"] = i
                result.append(d)
        return result

    # ---- slots ----

    def _on_device_changed(self, combo_index: int) -> None:
        if combo_index < 0 or combo_index >= len(self._devices):
            return

        dev = self._devices[combo_index]
        dev_index = dev["index"]

        # 기존 소스 중지
        self._stop_pipeline()

        # 새 소스 시작
        self._mic_source = MicrophoneSource(
            bus=self._bus,
            out_topic="audio/raw",
            samplerate=16000,
            channels=1,
            blocksize=1024,
            device=dev_index,
            source_id=f"mic_{dev_index}",
        )
        self._mic_source.start()
        self._bridge.start()

        self._status_label.setText(
            f"모니터링 중: {dev['name']}  "
            f"(SR={int(dev['default_samplerate'])}Hz, "
            f"Ch={dev['max_input_channels']})"
        )
        self._status_label.setStyleSheet("color: #0a0;")

    def _on_level(self, level: float) -> None:
        self._gauge.set_level(level)

    # ---- cleanup ----

    def _stop_pipeline(self) -> None:
        self._bridge.stop()
        if self._mic_source is not None:
            self._mic_source.stop()
            self._mic_source = None
        self._gauge.set_level(0.0)

    def closeEvent(self, event):
        self._stop_pipeline()
        event.accept()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MicLevelMonitorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
