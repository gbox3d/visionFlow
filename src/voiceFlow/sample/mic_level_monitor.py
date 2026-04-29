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

import json
import os
import re
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from PySide6.QtCore import Qt, Signal, QObject
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

from common.runtime.bus import TopicBus
from voiceFlow.sources.microphone_source import MicrophoneSource
from voiceFlow.utils.audio_device import resolve_mic_device_from_camera_path
from asrFlow.utils.env import env_str_any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)


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
    _AUDIO_NAME_STOPWORDS = {
        "microsoft",
        "input",
        "output",
        "primary",
        "default",
        "sound",
        "mapper",
        "capture",
        "driver",
        "마이크",
        "수화기",
        "머리",
        "거는",
        "헤드폰",
        "주",
        "사운드",
        "캡처",
        "드라이버",
        "stereo",
        "mix",
        "audio",
        "hands",
        "free",
        "system32",
        "drivers",
        "sys",
        "bthhfenum",
        "bthhfenumsys",
        "머리에",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceFlow - Mic Level Monitor")
        self.setMinimumSize(480, 180)
        self.resize(760, 240)

        # --- 마이크 목록 구성 ---
        self._devices = self._list_input_devices()
        self._preferred_camera_path = env_str_any(("DEVICE_PATH", "CAMERA_DEVICE_PATH", "DEMO_DEVICE_PATH"), "").strip()
        self._preferred_mic_selector = (
            resolve_mic_device_from_camera_path(self._preferred_camera_path)
            if self._preferred_camera_path
            else None
        )

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
        self._path_label = QLabel("device path: -")
        self._path_label.setStyleSheet("color: #666; font-size: 11px;")
        self._path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)

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
            initial_combo_index = self._select_preferred_device_index()
            self._combo.setCurrentIndex(initial_combo_index)
            self._on_device_changed(initial_combo_index)

    # ---- helpers ----

    @staticmethod
    def _list_input_devices() -> list:
        camera_paths_by_name: dict[str, list[str]] = {}
        try:
            from cv2_enumerate_cameras import enumerate_cameras

            for cam in enumerate_cameras():
                cam_name = str(getattr(cam, "name", "") or "").strip()
                cam_path = str(getattr(cam, "path", "") or "").strip()
                if not cam_name or not cam_path:
                    continue
                key_name = MicLevelMonitorWindow._normalize_audio_name_key(cam_name)
                existing = camera_paths_by_name.setdefault(key_name, [])
                norm = MicLevelMonitorWindow._normalize_camera_path_key(cam_path)
                if all(MicLevelMonitorWindow._normalize_camera_path_key(p) != norm for p in existing):
                    existing.append(cam_path)
        except Exception:
            camera_paths_by_name = {}

        endpoint_rows = MicLevelMonitorWindow._list_windows_audio_endpoints()
        default_endpoint_row: Optional[dict] = None
        try:
            default_input_name = str(sd.query_devices(kind="input").get("name", "")).strip()
            if default_input_name:
                default_endpoint_row = MicLevelMonitorWindow._match_audio_endpoint_row(default_input_name, endpoint_rows)
        except Exception:
            default_endpoint_row = None

        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                d = dict(dev)
                d["index"] = i
                dev_name = str(d.get("name", "")).strip()
                dev_key = MicLevelMonitorWindow._normalize_audio_name_key(dev_name)
                matched_paths: list[str] = []
                for cam_name, paths in camera_paths_by_name.items():
                    if cam_name and (cam_name in dev_key or dev_key in cam_name):
                        matched_paths.extend(paths)
                # Remove duplicates while preserving order.
                dedup: list[str] = []
                seen = set()
                for p in matched_paths:
                    k = MicLevelMonitorWindow._normalize_camera_path_key(p)
                    if k in seen:
                        continue
                    seen.add(k)
                    dedup.append(p)
                camera_path = " | ".join(dedup) if dedup else "-"
                endpoint_row = MicLevelMonitorWindow._match_audio_endpoint_row(dev_name, endpoint_rows)
                if endpoint_row is None and MicLevelMonitorWindow._is_default_capture_alias(dev_name):
                    endpoint_row = default_endpoint_row

                endpoint_path = "-"
                endpoint_name = "-"
                if endpoint_row:
                    endpoint_path = str(endpoint_row.get("instance_id") or "-")
                    endpoint_name = str(endpoint_row.get("friendly_name") or "-")

                d["camera_device_path"] = camera_path
                d["audio_endpoint_path"] = endpoint_path
                d["audio_endpoint_name"] = endpoint_name
                d["audio_parent_pnp_path"] = "-"
                d["device_path"] = camera_path if camera_path != "-" else endpoint_path
                result.append(d)
        return result

    @staticmethod
    def _normalize_camera_path_key(path: str) -> str:
        normalized = (path or "").strip().lower()
        return normalized.split("#{")[0] if "#{" in normalized else normalized

    @staticmethod
    def _normalize_audio_name_key(name: str) -> str:
        # Match names across APIs (MME/DSHOW/WASAPI/WDM-KS) by removing spacing/punctuation variance.
        normalized = (name or "").strip().lower()
        normalized = normalized.replace("\r", " ").replace("\n", " ")
        normalized = re.sub(r"[^0-9a-zA-Z가-힣]+", "", normalized)
        return normalized

    @staticmethod
    def _tokenize_audio_name(name: str) -> set[str]:
        cleaned = (name or "").strip().lower()
        cleaned = cleaned.replace("\r", " ").replace("\n", " ")
        tokens = re.split(r"[^0-9a-zA-Z가-힣]+", cleaned)
        return {t for t in tokens if t}

    @staticmethod
    def _extract_specific_tokens(tokens: set[str]) -> set[str]:
        return {t for t in tokens if len(t) >= 3 and t not in MicLevelMonitorWindow._AUDIO_NAME_STOPWORDS}

    @staticmethod
    def _is_default_capture_alias(device_name: str) -> bool:
        key = MicLevelMonitorWindow._normalize_audio_name_key(device_name)
        return (
            "soundmapper" in key
            or "사운드매퍼" in key
            or "primarysoundcapturedriver" in key
            or "주사운드캡처드라이버" in key
        )

    @staticmethod
    def _run_powershell_json(ps_script: str, timeout_s: float = 8.0):
        exe_candidates = ["powershell", "powershell.exe"]
        system_root = os.environ.get("SystemRoot", "").strip()
        if system_root:
            exe_candidates.append(
                str(Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")
            )
        exe_candidates.append("pwsh")

        seen = set()
        for exe in exe_candidates:
            if not exe:
                continue
            exe_key = exe.lower()
            if exe_key in seen:
                continue
            seen.add(exe_key)

            try:
                cp = subprocess.run(
                    [exe, "-NoProfile", "-Command", ps_script],
                    capture_output=True,
                    timeout=timeout_s,
                )
            except Exception:
                continue

            if cp.returncode != 0:
                continue

            raw = cp.stdout or b""
            text = ""
            for enc in ("utf-8-sig", "utf-16", "cp949", "latin-1"):
                try:
                    decoded = raw.decode(enc).strip()
                except Exception:
                    continue
                if decoded:
                    text = decoded
                    break

            if not text:
                continue

            try:
                return json.loads(text)
            except Exception:
                continue

        return None

    @staticmethod
    def _list_windows_audio_endpoints() -> list[dict]:
        if sys.platform != "win32":
            return []

        ps_script = """
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
Get-PnpDevice -Class AudioEndpoint | Select-Object FriendlyName,InstanceId | ConvertTo-Json -Depth 4
"""
        data = MicLevelMonitorWindow._run_powershell_json(ps_script, timeout_s=4.0)
        if data is None:
            return []

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        rows: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            friendly = str(item.get("FriendlyName", "")).strip()
            instance_id = str(item.get("InstanceId", "")).strip()
            if not friendly or not instance_id:
                continue
            if "{0.0.1." not in instance_id.lower():
                continue
            tokens = MicLevelMonitorWindow._tokenize_audio_name(friendly)
            rows.append(
                {
                    "friendly_name": friendly,
                    "name_key": MicLevelMonitorWindow._normalize_audio_name_key(friendly),
                    "tokens": tokens,
                    "specific_tokens": MicLevelMonitorWindow._extract_specific_tokens(tokens),
                    "instance_id": instance_id,
                }
            )
        return rows

    @staticmethod
    def _match_audio_endpoint_row(device_name: str, endpoint_rows: list[dict]) -> Optional[dict]:
        if not endpoint_rows:
            return None

        name_key = MicLevelMonitorWindow._normalize_audio_name_key(device_name)
        name_tokens = MicLevelMonitorWindow._tokenize_audio_name(device_name)
        name_specific = MicLevelMonitorWindow._extract_specific_tokens(name_tokens)

        # 1) exact normalized key
        for row in endpoint_rows:
            if row.get("name_key") == name_key:
                return row

        # 2) containment
        for row in endpoint_rows:
            endpoint_key = str(row.get("name_key") or "")
            if not endpoint_key:
                continue
            row_specific = row.get("specific_tokens")
            if (
                endpoint_key
                and (endpoint_key in name_key or name_key in endpoint_key)
                and isinstance(row_specific, set)
                and (not name_specific or bool(name_specific.intersection(row_specific)))
            ):
                return row

        # 3) token overlap with specific-token guard (reduce false positives)
        best_score = 0
        best_row: Optional[dict] = None
        for row in endpoint_rows:
            tokens = row.get("tokens")
            row_specific = row.get("specific_tokens")
            if not isinstance(tokens, set) or not tokens or not isinstance(row_specific, set):
                continue
            spec_overlap = name_specific.intersection(row_specific)
            if not spec_overlap:
                continue
            score = len(spec_overlap) * 100 + len(name_tokens.intersection(tokens)) * 10
            if score > best_score:
                best_score = score
                best_row = row

        return best_row

    def _select_preferred_device_index(self) -> int:
        if not self._devices:
            return 0

        sel = self._preferred_mic_selector
        if isinstance(sel, int):
            for combo_idx, dev in enumerate(self._devices):
                if int(dev.get("index", -1)) == sel:
                    return combo_idx
        elif isinstance(sel, str):
            target = sel.strip().lower()
            if target:
                for combo_idx, dev in enumerate(self._devices):
                    if target in str(dev.get("name", "")).lower():
                        return combo_idx

        return 0

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
        self._path_label.setText(
            "camera device path: "
            f"{dev.get('camera_device_path', '-')}\n"
            "audio endpoint path: "
            f"{dev.get('audio_endpoint_path', '-')}"
        )

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
