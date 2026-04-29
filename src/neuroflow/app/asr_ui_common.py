from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QColor, QPainter, QFont, QPen, QBrush, QLinearGradient
from PySide6.QtWidgets import QWidget

from common.runtime.bus import TopicBus


class LevelGaugeWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(28)
        self.setMaximumHeight(32)
        self.setMinimumWidth(200)

        self._level: float = 0.0
        self._peak: float = 0.0
        self._peak_ts: float = 0.0
        self._peak_hold_s: float = 1.5

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))
        now = time.time()
        if self._level >= self._peak:
            self._peak = self._level
            self._peak_ts = now
        elif now - self._peak_ts > self._peak_hold_s:
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

        p.fillRect(margin, margin, bar_w, bar_h, QColor(30, 30, 30))

        level_w = int(bar_w * self._level)
        if level_w > 0:
            grad = QLinearGradient(margin, 0, margin + bar_w, 0)
            grad.setColorAt(0.0, QColor(0, 200, 0))
            grad.setColorAt(0.6, QColor(200, 200, 0))
            grad.setColorAt(1.0, QColor(220, 0, 0))
            p.fillRect(margin, margin, level_w, bar_h, QBrush(grad))

        peak_x = margin + int(bar_w * self._peak)
        if peak_x > margin + 2:
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(peak_x, margin, peak_x, margin + bar_h)

        db = max(-60.0, 20.0 * np.log10(self._level + 1e-10))
        p.setPen(QColor(220, 220, 220))
        p.setFont(QFont("Consolas", 9))
        p.drawText(
            margin + 4,
            margin,
            bar_w - 8,
            bar_h,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{db:.1f} dB",
        )
        p.end()


class AudioLevelBridge(QObject):
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
                rms = float(np.sqrt(np.mean(pkt.audio.astype(np.float64) ** 2)))
                self.level_updated.emit(min(1.0, rms * 3.0))
