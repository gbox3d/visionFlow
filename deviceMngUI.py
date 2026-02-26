from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import dotenv_values, find_dotenv, set_key, unset_key
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visionflow.pipeline.bus import TopicBus
from visionflow.sources.camera_source import CameraSource
from visionflow.utils.etc import generate_resolution_candidates, normalize_device_path_key, parse_resolution
from voiceFlow.sources.microphone_source import MicrophoneSource
from voiceFlow.utils.audio_device import parse_mic_device


TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
ENV_DROPDOWN_OPTIONS: dict[str, tuple[str, ...]] = {
    "VOICEFLOW_STT_BACKEND": ("ct2", "hf_generate", "hf_pipeline"),
    "VOICEFLOW_STT_DEVICE": ("auto", "cuda", "cpu"),
}

_CAMERA_ID_KEYS = {"CAMERA_ID", "DEMO_CAMERA_ID"}
_PATH_KEYS = {"DEVICE_PATH", "DEMO_DEVICE_PATH", "CAMERA_DEVICE_PATH", "DEMO_CAMERA_DEVICE_PATH"}
_MIC_DEVICE_KEYS = {"MIC_DEVICE", "DEMO_MIC_DEVICE"}
_CAMERA_RESOLUTION_KEYS = {"CAMERA_RESOLUTION", "DEMO_CAMERA_RESOLUTION"}


@dataclass
class CameraDevice:
    camera_id: int
    name: str
    vid_pid: str
    backend: str
    path: str


@dataclass
class MicrophoneDevice:
    mic_id: int
    name: str
    hostapi: str
    channels: int
    is_default: bool


def _camera_id_candidates() -> tuple[str, ...]:
    candidates: set[int] = set()
    dshow_code = int(getattr(cv2, "CAP_DSHOW", 700))
    msmf_code = getattr(cv2, "CAP_MSMF", None)
    msmf_code_int = int(msmf_code) if msmf_code is not None else None

    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception:
        enumerate_cameras = None

    if enumerate_cameras is not None:
        try:
            for cam in enumerate_cameras():
                backend = getattr(cam, "backend", None)
                idx = int(cam.index)
                if backend == dshow_code or (msmf_code_int is not None and backend == msmf_code_int):
                    cam_id = int(cam.index) % 100
                elif msmf_code_int is not None and idx >= msmf_code_int:
                    cam_id = idx - msmf_code_int
                elif idx >= dshow_code:
                    cam_id = idx - dshow_code
                else:
                    cam_id = idx
                if cam_id >= 0:
                    candidates.add(cam_id)
        except Exception:
            pass

    if not candidates:
        candidates.update(range(10))

    return tuple(str(i) for i in sorted(candidates))


def _mic_id_candidates() -> tuple[str, ...]:
    mics = _list_mics()
    candidates = sorted({int(m.mic_id) for m in mics if int(m.mic_id) >= 0})
    if not candidates:
        candidates = list(range(6))
    return tuple(str(i) for i in candidates)


def _probe_camera(camera_id: int, use_dshow: bool = True) -> bool:
    backend = cv2.CAP_DSHOW if use_dshow else 0
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def _decode_camera_index(raw_index: int) -> int:
    index = int(raw_index)
    dshow_code = int(getattr(cv2, "CAP_DSHOW", 700))
    msmf_code = getattr(cv2, "CAP_MSMF", None)
    if msmf_code is not None and index >= int(msmf_code):
        return index - int(msmf_code)
    if index >= dshow_code:
        return index - dshow_code
    return index


def _resolve_camera_id_from_path(device_path: str, use_dshow: bool) -> Optional[int]:
    target_key = normalize_device_path_key(device_path)
    if not target_key:
        return None

    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception:
        return None

    dshow_code = int(getattr(cv2, "CAP_DSHOW", 700))
    msmf_code = getattr(cv2, "CAP_MSMF", None)
    prefer_base = dshow_code if use_dshow else int(msmf_code) if msmf_code is not None else dshow_code

    matched_indices: list[int] = []
    try:
        for cam in enumerate_cameras():
            cam_path = str(getattr(cam, "path", None) or "")
            if not cam_path:
                continue
            if normalize_device_path_key(cam_path) != target_key:
                continue
            matched_indices.append(int(getattr(cam, "index", -1)))
    except Exception:
        return None

    if not matched_indices:
        return None

    for index in matched_indices:
        if prefer_base <= index < prefer_base + 100:
            return index - prefer_base

    return _decode_camera_index(matched_indices[0])


def _probe_supported_camera_resolutions(camera_id: int, use_dshow: bool) -> list[tuple[int, int]]:
    backend = cv2.CAP_DSHOW if use_dshow else 0
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap.release()
        return []

    resolutions: set[tuple[int, int]] = set()

    def _record_size(frame: Optional[np.ndarray]) -> None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w <= 0 or h <= 0) and frame is not None and getattr(frame, "size", 0) > 0:
            h, w = frame.shape[:2]
        if w >= 160 and h >= 120:
            resolutions.add((w, h))

    try:
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            _record_size(frame)

        for req_w, req_h in generate_resolution_candidates(max_w=3840, max_h=2160):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(req_w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(req_h))

            success = False
            captured: Optional[np.ndarray] = None
            for _ in range(4):
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    success = True
                    captured = frame
                    break
                time.sleep(0.02)

            if not success:
                continue

            _record_size(captured)
    finally:
        cap.release()

    return sorted(resolutions, key=lambda s: (s[0] * s[1], s[0], s[1]))


def _list_cameras_hwinfo() -> list[CameraDevice]:
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception:
        return []

    cameras = list(enumerate_cameras())
    backend_ranges = {
        int(getattr(cv2, "CAP_DSHOW", 700)): "DSHOW",
    }
    if hasattr(cv2, "CAP_MSMF"):
        backend_ranges[int(cv2.CAP_MSMF)] = "MSMF"

    out: list[CameraDevice] = []
    for cam in cameras:
        idx = int(getattr(cam, "index", 0))
        backend = int(getattr(cam, "backend", 0)) if getattr(cam, "backend", None) is not None else None

        if backend is not None and backend in backend_ranges:
            backend_name = backend_ranges[backend]
            cam_id = idx % 100
        else:
            matched = False
            cam_id = idx
            backend_name = "AUTO"
            for base, name in sorted(backend_ranges.items(), reverse=True):
                if idx >= base:
                    cam_id = idx - base
                    backend_name = name
                    matched = True
                    break
            if not matched:
                backend_name = "AUTO"

        vid_val = getattr(cam, "vid", None)
        pid_val = getattr(cam, "pid", None)
        vid = f"{int(vid_val):04X}" if vid_val else "----"
        pid = f"{int(pid_val):04X}" if pid_val else "----"

        out.append(
            CameraDevice(
                camera_id=int(cam_id),
                name=str(getattr(cam, "name", None) or "-"),
                vid_pid=f"{vid}:{pid}",
                backend=backend_name,
                path=str(getattr(cam, "path", None) or "-"),
            )
        )
    return out


def _list_mics() -> list[MicrophoneDevice]:
    try:
        import sounddevice as sd
    except Exception:
        return []

    try:
        devices = sd.query_devices()
    except Exception:
        return []

    default_idx = -1
    try:
        default_pair = getattr(sd.default, "device", None)
        if isinstance(default_pair, (list, tuple)) and len(default_pair) >= 1 and default_pair[0] is not None:
            default_idx = int(default_pair[0])
    except Exception:
        pass

    out: list[MicrophoneDevice] = []
    for idx, device in enumerate(devices):
        max_input = int(device.get("max_input_channels", 0))
        if max_input <= 0:
            continue
        hostapi_name = "-"
        try:
            hostapi_index = int(device.get("hostapi", -1))
            hostapi_name = str(sd.query_hostapis(hostapi_index).get("name", "-"))
        except Exception:
            pass
        out.append(
            MicrophoneDevice(
                mic_id=idx,
                name=str(device.get("name", "-")),
                hostapi=hostapi_name,
                channels=max_input,
                is_default=(idx == default_idx),
            )
        )
    return out


class _SelectTableDialog(QDialog):
    def __init__(self, title: str, headers: list[str], rows: list[list[str]], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 620)

        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(lambda _r, _c: self.accept())
        layout.addWidget(self.table)

        self.table.setRowCount(len(rows))
        for r, row_values in enumerate(rows):
            for c, value in enumerate(row_values):
                self.table.setItem(r, c, QTableWidgetItem(value))

        if rows:
            self.table.selectRow(0)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_row(self) -> int:
        selected = self.table.selectedItems()
        if not selected:
            return -1
        return selected[0].row()


@dataclass
class RuntimeConfig:
    camera_id: int
    camera_path: Optional[str]
    camera_width: int
    camera_height: int
    camera_use_dshow: bool
    mic_device: Optional[int | str]
    mic_samplerate: int
    mic_blocksize: int
    note: str = ""


def _parse_int(value: str, default: int) -> int:
    try:
        return int((value or "").strip())
    except ValueError:
        return default


def _parse_bool(value: str, default: bool) -> bool:
    raw = (value or "").strip().lower()
    if not raw:
        return default
    return raw in TRUTHY_VALUES


def _suppress_opencv_warnings() -> None:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        return
    except Exception:
        pass
    try:
        cv2.setLogLevel(3)
        return
    except Exception:
        pass


class DeviceMngUI(QMainWindow):
    def __init__(self, dotenv_path: Path) -> None:
        super().__init__()
        self.setWindowTitle("NeuroFlow deviceMngUI")
        self.resize(1380, 820)

        self.dotenv_path = dotenv_path
        self._loaded_keys: set[str] = set()
        self._camera_resolution_options: tuple[str, ...] = ()

        self._bus: Optional[TopicBus] = None
        self._camera_source: Optional[CameraSource] = None
        self._mic_source: Optional[MicrophoneSource] = None
        self._last_frame_ver = 0
        self._last_audio_ver = 0
        self._raw_camera_pixmap: Optional[QPixmap] = None

        self._last_audio_level = 0.0
        self._last_audio_update_ts = 0.0

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(33)
        self._poll_timer.timeout.connect(self._poll_sources)

        self._build_ui()
        self._load_env_table()
        self._set_status(f"Loaded .env: {self.dotenv_path}")

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)

        splitter.setSizes([820, 560])

        preview_group = QGroupBox("Source Test (No AI Processor)")
        preview_layout = QVBoxLayout(preview_group)
        left_layout.addWidget(preview_group, 1)

        controls = QHBoxLayout()
        self.btn_start = QPushButton("Start Test")
        self.btn_start.clicked.connect(self.start_sources)
        self.btn_stop = QPushButton("Stop Test")
        self.btn_stop.clicked.connect(self.stop_sources)
        self.btn_restart = QPushButton("Restart Test")
        self.btn_restart.clicked.connect(self.restart_sources)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_restart)
        controls.addStretch(1)
        preview_layout.addLayout(controls)

        self.camera_label = QLabel("Camera preview is stopped.")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(760, 430)
        self.camera_label.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        preview_layout.addWidget(self.camera_label, 1)

        self.camera_info_label = QLabel("camera: -")
        self.camera_info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        preview_layout.addWidget(self.camera_info_label)

        mic_group = QGroupBox("Microphone RMS")
        mic_layout = QVBoxLayout(mic_group)
        preview_layout.addWidget(mic_group)

        self.rms_bar = QProgressBar()
        self.rms_bar.setRange(0, 100)
        self.rms_bar.setValue(0)
        self.rms_bar.setTextVisible(False)
        self.rms_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #666; background: #1a1a1a; }"
            "QProgressBar::chunk { background-color: #46d25a; }"
        )
        mic_layout.addWidget(self.rms_bar)

        self.rms_label = QLabel("RMS 0.000 (-60.0 dB)")
        self.rms_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mic_layout.addWidget(self.rms_label)

        env_group = QGroupBox(".env Editor")
        env_layout = QVBoxLayout(env_group)
        right_layout.addWidget(env_group, 1)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self.env_path_label = QLabel(str(self.dotenv_path))
        self.env_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        path_row.addWidget(self.env_path_label, 1)
        env_layout.addLayout(path_row)

        env_controls = QHBoxLayout()
        self.btn_reload = QPushButton("Reload .env")
        self.btn_reload.clicked.connect(self._load_env_table)
        self.btn_add_row = QPushButton("Add Row")
        self.btn_add_row.clicked.connect(self._add_env_row)
        self.btn_del_row = QPushButton("Remove Row")
        self.btn_del_row.clicked.connect(self._remove_env_rows)
        self.btn_pick_path = QPushButton("Pick Path")
        self.btn_pick_path.clicked.connect(self._open_path_picker)
        self.btn_pick_camera_id = QPushButton("Pick Camera ID")
        self.btn_pick_camera_id.clicked.connect(self._open_camera_id_picker)
        self.btn_pick_camera_res = QPushButton("Pick Camera Res")
        self.btn_pick_camera_res.clicked.connect(self._open_camera_resolution_picker)
        self.btn_pick_mic = QPushButton("Pick Mic")
        self.btn_pick_mic.clicked.connect(self._open_mic_picker)
        self.btn_save = QPushButton("Save .env")
        self.btn_save.clicked.connect(self._save_env_table)
        env_controls.addWidget(self.btn_reload)
        env_controls.addWidget(self.btn_add_row)
        env_controls.addWidget(self.btn_del_row)
        env_controls.addWidget(self.btn_pick_path)
        env_controls.addWidget(self.btn_pick_camera_id)
        env_controls.addWidget(self.btn_pick_camera_res)
        env_controls.addWidget(self.btn_pick_mic)
        env_controls.addStretch(1)
        env_controls.addWidget(self.btn_save)
        env_layout.addLayout(env_controls)

        self.env_table = QTableWidget()
        self.env_table.setColumnCount(2)
        self.env_table.setHorizontalHeaderLabels(["KEY", "VALUE"])
        self.env_table.horizontalHeader().setStretchLastSection(True)
        self.env_table.setAlternatingRowColors(True)
        self.env_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.env_table.itemChanged.connect(self._on_env_item_changed)
        self.env_table.cellDoubleClicked.connect(self._on_env_cell_double_clicked)
        env_layout.addWidget(self.env_table, 1)

        self.status_label = QLabel("")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        main_layout.addWidget(self.status_label)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _load_env_table(self) -> None:
        if not self.dotenv_path.exists():
            self.dotenv_path.write_text("# NeuroFlow environment\n", encoding="utf-8")

        values = dotenv_values(self.dotenv_path)
        items = [(str(k), "" if v is None else str(v)) for k, v in values.items() if k is not None]

        self.env_table.blockSignals(True)
        self.env_table.setRowCount(len(items))
        for row, (key, value) in enumerate(items):
            self.env_table.setItem(row, 0, QTableWidgetItem(key))
            self._set_value_cell(row, key, value)
        self.env_table.blockSignals(False)
        self.env_table.resizeColumnsToContents()

        self._loaded_keys = {k for k, _ in items}
        self._set_status(f"Reloaded .env ({len(items)} keys)")

    def _add_env_row(self) -> None:
        row = self.env_table.rowCount()
        self.env_table.insertRow(row)
        self.env_table.setItem(row, 0, QTableWidgetItem(""))
        self._set_value_cell(row, "", "")
        self.env_table.setCurrentCell(row, 0)

    def _remove_env_rows(self) -> None:
        selected_rows = sorted({index.row() for index in self.env_table.selectedIndexes()}, reverse=True)
        if not selected_rows:
            self._set_status("Select rows to remove.")
            return
        for row in selected_rows:
            self.env_table.removeRow(row)
        self._set_status(f"Removed {len(selected_rows)} row(s).")

    def _find_row_by_key(self, key: str) -> int:
        target = key.strip()
        if not target:
            return -1
        for row in range(self.env_table.rowCount()):
            key_item = self.env_table.item(row, 0)
            if key_item is None:
                continue
            if key_item.text().strip() == target:
                return row
        return -1

    def _set_env_key_value(self, key: str, value: str) -> None:
        row = self._find_row_by_key(key)
        if row < 0:
            row = self.env_table.rowCount()
            self.env_table.insertRow(row)
            self.env_table.setItem(row, 0, QTableWidgetItem(key))
        else:
            self.env_table.setItem(row, 0, QTableWidgetItem(key))

        self.env_table.blockSignals(True)
        self._set_value_cell(row, key, value)
        self.env_table.blockSignals(False)
        self.env_table.selectRow(row)

    def _collect_env_map(self) -> dict[str, str]:
        env_map: dict[str, str] = {}
        for row in range(self.env_table.rowCount()):
            key_item = self.env_table.item(row, 0)
            key = (key_item.text() if key_item else "").strip()
            value_combobox = self._get_value_combobox(row)
            if value_combobox is not None:
                value = value_combobox.currentText().strip()
            else:
                value_checkbox = self._get_value_checkbox(row)
                if value_checkbox is not None:
                    value = "true" if value_checkbox.isChecked() else "false"
                else:
                    val_item = self.env_table.item(row, 1)
                    value = (val_item.text() if val_item else "").strip()
            if not key:
                continue
            if key in env_map:
                raise ValueError(f"Duplicate key in table: {key}")
            env_map[key] = value
        return env_map

    def _parse_bool_literal(self, value: str) -> Optional[bool]:
        raw = (value or "").strip().lower()
        if raw == "true":
            return True
        if raw == "false":
            return False
        return None

    def _get_value_checkbox(self, row: int) -> Optional[QCheckBox]:
        widget = self.env_table.cellWidget(row, 1)
        if widget is None:
            return None
        return widget.findChild(QCheckBox)

    def _get_value_combobox(self, row: int) -> Optional[QComboBox]:
        widget = self.env_table.cellWidget(row, 1)
        if widget is None:
            return None
        if isinstance(widget, QComboBox):
            return widget
        return widget.findChild(QComboBox)

    def _read_row_value(self, row: int) -> str:
        value_combobox = self._get_value_combobox(row)
        if value_combobox is not None:
            return value_combobox.currentText().strip()

        value_checkbox = self._get_value_checkbox(row)
        if value_checkbox is not None:
            return "true" if value_checkbox.isChecked() else "false"

        value_item = self.env_table.item(row, 1)
        return (value_item.text() if value_item else "").strip()

    def _set_value_cell(self, row: int, key: str, value: str) -> None:
        self.env_table.removeCellWidget(row, 1)

        norm_key = (key or "").strip()
        dropdown_options = ENV_DROPDOWN_OPTIONS.get(norm_key, ())
        if norm_key in _CAMERA_ID_KEYS:
            dropdown_options = _camera_id_candidates()
        elif norm_key in _MIC_DEVICE_KEYS:
            dropdown_options = _mic_id_candidates()
        elif norm_key in _CAMERA_RESOLUTION_KEYS and self._camera_resolution_options:
            dropdown_options = self._camera_resolution_options
        if dropdown_options:
            combobox = QComboBox()
            for option in dropdown_options:
                combobox.addItem(option)

            selected = (value or "").strip()
            if selected and selected not in dropdown_options:
                combobox.addItem(selected)

            if not selected:
                selected = dropdown_options[0]
            selected_index = combobox.findText(selected)
            if selected_index < 0:
                selected_index = 0
            combobox.setCurrentIndex(selected_index)

            self.env_table.setItem(row, 1, QTableWidgetItem(""))
            self.env_table.item(row, 1).setFlags(
                self.env_table.item(row, 1).flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            self.env_table.setCellWidget(row, 1, combobox)
            return

        bool_value = self._parse_bool_literal(value)
        if bool_value is None:
            self.env_table.setItem(row, 1, QTableWidgetItem(value))
            return

        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        checkbox = QCheckBox()
        checkbox.setChecked(bool_value)
        layout.addWidget(checkbox)
        self.env_table.setItem(row, 1, QTableWidgetItem(""))
        self.env_table.item(row, 1).setFlags(
            self.env_table.item(row, 1).flags() & ~Qt.ItemFlag.ItemIsEditable
        )
        self.env_table.setCellWidget(row, 1, holder)

    def _on_env_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            key = item.text().strip()
            value = self._read_row_value(item.row())
            self.env_table.blockSignals(True)
            self._set_value_cell(item.row(), key, value)
            self.env_table.blockSignals(False)
            return

        if item.column() != 1:
            return

        key_item = self.env_table.item(item.row(), 0)
        key = (key_item.text() if key_item else "").strip()
        if key in ENV_DROPDOWN_OPTIONS:
            return

        if self._get_value_combobox(item.row()) is not None:
            return

        if self._get_value_checkbox(item.row()) is not None:
            return

        bool_value = self._parse_bool_literal(item.text())
        if bool_value is None:
            return

        self.env_table.blockSignals(True)
        self._set_value_cell(item.row(), key, "true" if bool_value else "false")
        self.env_table.blockSignals(False)

    def _on_env_cell_double_clicked(self, row: int, col: int) -> None:
        if row < 0:
            return
        key_item = self.env_table.item(row, 0)
        key = (key_item.text() if key_item else "").strip()
        if not key:
            return
        if key in _PATH_KEYS:
            self._open_path_picker(target_key=key)
        elif key in _CAMERA_ID_KEYS:
            self._open_camera_id_picker(target_key=key)
        elif key in _CAMERA_RESOLUTION_KEYS:
            self._open_camera_resolution_picker(target_key=key)
        elif key in _MIC_DEVICE_KEYS:
            self._open_mic_picker(target_key=key)

    def _resolve_target_key(self, preferred: tuple[str, ...], fallback: str) -> str:
        for key in preferred:
            if self._find_row_by_key(key) >= 0:
                return key
        return fallback

    def _set_camera_resolution_options(self, options: list[str]) -> None:
        self._camera_resolution_options = tuple(options)
        self.env_table.blockSignals(True)
        for row in range(self.env_table.rowCount()):
            key_item = self.env_table.item(row, 0)
            key = (key_item.text() if key_item else "").strip()
            if key not in _CAMERA_RESOLUTION_KEYS:
                continue
            current = self._read_row_value(row)
            self._set_value_cell(row, key, current)
        self.env_table.blockSignals(False)

    def _clear_camera_resolution_options(self) -> None:
        if not self._camera_resolution_options:
            return
        self._camera_resolution_options = ()
        self.env_table.blockSignals(True)
        for row in range(self.env_table.rowCount()):
            key_item = self.env_table.item(row, 0)
            key = (key_item.text() if key_item else "").strip()
            if key not in _CAMERA_RESOLUTION_KEYS:
                continue
            current = self._read_row_value(row)
            self._set_value_cell(row, key, current)
        self.env_table.blockSignals(False)

    def _current_camera_probe_target(self) -> tuple[int, bool, Optional[str], bool]:
        env_map = self._collect_env_map()
        camera_id = _parse_int(self._env_first(env_map, ("CAMERA_ID", "DEMO_CAMERA_ID"), "0"), 0)
        use_dshow = _parse_bool(
            self._env_first(env_map, ("CAMERA_USE_DSHOW", "DEMO_CAMERA_USE_DSHOW"), "true"),
            True,
        )
        unified_path = self._env_first(env_map, ("DEVICE_PATH", "DEMO_DEVICE_PATH"), "")
        legacy_path = self._env_first(env_map, ("CAMERA_DEVICE_PATH", "DEMO_CAMERA_DEVICE_PATH"), "")
        camera_path = unified_path or legacy_path or None

        resolved_from_path = False
        if camera_path:
            resolved = _resolve_camera_id_from_path(camera_path, use_dshow=use_dshow)
            if resolved is not None:
                camera_id = resolved
                resolved_from_path = True

        return camera_id, use_dshow, camera_path, resolved_from_path

    def _open_camera_resolution_picker(self, target_key: Optional[str] = None) -> None:
        try:
            camera_id, use_dshow, camera_path, resolved_from_path = self._current_camera_probe_target()
        except ValueError as exc:
            self._set_status(str(exc))
            return

        backend_name = "DSHOW" if use_dshow else "AUTO"
        self._set_status(f"Probing camera resolutions... id={camera_id}, backend={backend_name}")
        QApplication.processEvents()

        probed = _probe_supported_camera_resolutions(camera_id=camera_id, use_dshow=use_dshow)
        if not probed:
            if camera_path and not resolved_from_path:
                self._set_status(
                    f"Resolution probe failed: path not matched. "
                    f"fallback camera_id={camera_id}, backend={backend_name}"
                )
            else:
                self._set_status(f"No selectable camera resolutions found (id={camera_id}, backend={backend_name}).")
            self._clear_camera_resolution_options()
            return

        resolution_values = [f"{w}x{h}" for w, h in probed]
        self._set_camera_resolution_options(resolution_values)

        rows: list[list[str]] = []
        for w, h in probed:
            ratio = (w / h) if h > 0 else 0.0
            rows.append([f"{w}x{h}", str(w), str(h), f"{ratio:.3f}"])

        dialog = _SelectTableDialog(
            title=f"Select CAMERA_RESOLUTION (id={camera_id}, backend={backend_name})",
            headers=["resolution", "width", "height", "aspect"],
            rows=rows,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        row = dialog.selected_row()
        if row < 0:
            self._set_status("No camera resolution selected.")
            return

        key = target_key or self._resolve_target_key(("CAMERA_RESOLUTION", "DEMO_CAMERA_RESOLUTION"), "CAMERA_RESOLUTION")
        selected = resolution_values[row]
        self._set_env_key_value(key, selected)
        self._set_status(f"{key} updated from dialog: {selected}")

    def _open_path_picker(self, target_key: Optional[str] = None) -> None:
        cameras = _list_cameras_hwinfo()
        rows: list[list[str]] = []
        valid_paths: list[str] = []
        for cam in cameras:
            path = (cam.path or "").strip()
            if path in ("", "-", "--"):
                continue
            rows.append(
                [
                    str(cam.camera_id),
                    cam.name,
                    cam.vid_pid,
                    cam.backend,
                    path,
                ]
            )
            valid_paths.append(path)

        if not rows:
            self._set_status("No selectable camera path found.")
            return

        dialog = _SelectTableDialog(
            title="Select DEVICE_PATH",
            headers=["camera_id", "name", "VID:PID", "backend", "path"],
            rows=rows,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        row = dialog.selected_row()
        if row < 0:
            self._set_status("No path selected.")
            return
        selected_path = valid_paths[row]
        key = target_key or self._resolve_target_key(
            ("DEVICE_PATH", "CAMERA_DEVICE_PATH", "DEMO_DEVICE_PATH", "DEMO_CAMERA_DEVICE_PATH"),
            "DEVICE_PATH",
        )
        self._set_env_key_value(key, selected_path)
        self._clear_camera_resolution_options()
        self._set_status(f"{key} updated from dialog.")

    def _open_camera_id_picker(self, target_key: Optional[str] = None) -> None:
        cameras = _list_cameras_hwinfo()
        rows: list[list[str]] = []
        ids: list[int] = []

        if cameras:
            seen: set[int] = set()
            for cam in cameras:
                if cam.camera_id in seen:
                    continue
                seen.add(cam.camera_id)
                rows.append(
                    [
                        str(cam.camera_id),
                        cam.name,
                        cam.backend,
                        cam.path,
                    ]
                )
                ids.append(cam.camera_id)
        else:
            for cam_id in range(10):
                ok = _probe_camera(cam_id, use_dshow=True)
                rows.append([str(cam_id), "(probe)", "DSHOW", "OK" if ok else "--"])
                ids.append(cam_id)

        if not rows:
            self._set_status("No camera IDs found.")
            return

        dialog = _SelectTableDialog(
            title="Select CAMERA_ID (used when path is unavailable)",
            headers=["camera_id", "name", "backend", "path/status"],
            rows=rows,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        row = dialog.selected_row()
        if row < 0:
            self._set_status("No camera ID selected.")
            return

        key = target_key or self._resolve_target_key(("CAMERA_ID", "DEMO_CAMERA_ID"), "CAMERA_ID")
        self._set_env_key_value(key, str(ids[row]))
        self._clear_camera_resolution_options()
        self._set_status(f"{key} updated from dialog.")

    def _open_mic_picker(self, target_key: Optional[str] = None) -> None:
        mics = _list_mics()
        if not mics:
            self._set_status("No microphone input device found.")
            return

        rows: list[list[str]] = []
        ids: list[int] = []
        for mic in mics:
            rows.append(
                [
                    str(mic.mic_id),
                    mic.name,
                    mic.hostapi,
                    str(mic.channels),
                    "Y" if mic.is_default else "",
                ]
            )
            ids.append(mic.mic_id)

        dialog = _SelectTableDialog(
            title="Select MIC_DEVICE",
            headers=["mic_id", "name", "hostapi", "channels", "default"],
            rows=rows,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        row = dialog.selected_row()
        if row < 0:
            self._set_status("No microphone selected.")
            return

        key = target_key or self._resolve_target_key(("MIC_DEVICE", "DEMO_MIC_DEVICE"), "MIC_DEVICE")
        self._set_env_key_value(key, str(ids[row]))
        self._set_status(f"{key} updated from dialog.")

    def _save_env_table(self) -> None:
        try:
            env_map = self._collect_env_map()
        except ValueError as exc:
            self._set_status(str(exc))
            return

        if not self.dotenv_path.exists():
            self.dotenv_path.write_text("# NeuroFlow environment\n", encoding="utf-8")

        removed = self._loaded_keys - set(env_map.keys())
        for key in sorted(removed):
            unset_key(self.dotenv_path, key)

        for key, value in env_map.items():
            set_key(self.dotenv_path, key, value)

        self._loaded_keys = set(env_map.keys())
        self._set_status(f"Saved .env ({len(env_map)} keys)")

    def _env_first(self, env_map: dict[str, str], keys: tuple[str, ...], default: str) -> str:
        for key in keys:
            value = (env_map.get(key) or "").strip()
            if value:
                return value
        return default

    def _build_runtime_config(self) -> RuntimeConfig:
        env_map = self._collect_env_map()

        camera_id = _parse_int(self._env_first(env_map, ("CAMERA_ID", "DEMO_CAMERA_ID"), "0"), 0)
        unified_path = self._env_first(env_map, ("DEVICE_PATH", "DEMO_DEVICE_PATH"), "")
        legacy_path = self._env_first(env_map, ("CAMERA_DEVICE_PATH", "DEMO_CAMERA_DEVICE_PATH"), "")
        camera_path = unified_path or legacy_path or None

        resolution_text = self._env_first(
            env_map,
            ("CAMERA_RESOLUTION", "DEMO_CAMERA_RESOLUTION"),
            "1280x720",
        )
        resolution = parse_resolution(resolution_text)
        if resolution is None:
            camera_width, camera_height = 1280, 720
        else:
            camera_width, camera_height = resolution

        camera_use_dshow = _parse_bool(
            self._env_first(env_map, ("CAMERA_USE_DSHOW", "DEMO_CAMERA_USE_DSHOW"), "true"),
            True,
        )

        mic_device_parsed = parse_mic_device(self._env_first(env_map, ("MIC_DEVICE", "DEMO_MIC_DEVICE"), ""))
        mic_device: Optional[int] = mic_device_parsed if isinstance(mic_device_parsed, int) else None
        note = ""
        if isinstance(mic_device_parsed, str):
            note = f"MIC_DEVICE index only mode: ignored '{mic_device_parsed}'"

        mic_samplerate = _parse_int(
            self._env_first(
                env_map,
                ("MIC_SAMPLERATE", "DEMO_MIC_SAMPLERATE", "VOICEFLOW_STT_SAMPLERATE"),
                "16000",
            ),
            16000,
        )
        mic_blocksize = _parse_int(
            self._env_first(env_map, ("MIC_BLOCKSIZE", "DEMO_MIC_BLOCKSIZE"), "1024"),
            1024,
        )

        return RuntimeConfig(
            camera_id=camera_id,
            camera_path=camera_path,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_use_dshow=camera_use_dshow,
            mic_device=mic_device,
            mic_samplerate=mic_samplerate,
            mic_blocksize=mic_blocksize,
            note=note,
        )

    def start_sources(self) -> None:
        try:
            config = self._build_runtime_config()
        except ValueError as exc:
            self._set_status(str(exc))
            return

        self.stop_sources(clear_status=False)

        self._bus = TopicBus()
        self._last_frame_ver = self._bus.get_version("frame/raw")
        self._last_audio_ver = self._bus.get_version("audio/raw")
        self._last_audio_level = 0.0
        self._last_audio_update_ts = time.time()

        self._camera_source = CameraSource(
            bus=self._bus,
            out_topic="frame/raw",
            camera_id=config.camera_id,
            device_path=config.camera_path,
            request_width=config.camera_width,
            request_height=config.camera_height,
            use_dshow=config.camera_use_dshow,
            source_id="deviceMngUI-camera",
        )
        self._mic_source = MicrophoneSource(
            bus=self._bus,
            out_topic="audio/raw",
            samplerate=config.mic_samplerate,
            channels=1,
            blocksize=config.mic_blocksize,
            device=config.mic_device,
            source_id="deviceMngUI-mic",
        )

        started_parts: list[str] = []
        warnings: list[str] = []

        try:
            self._camera_source.start()
            started_parts.append("camera")
        except Exception as exc:
            warnings.append(f"camera start failed: {exc}")
            self._camera_source = None

        try:
            self._mic_source.start()
            started_parts.append("mic")
        except Exception as exc:
            warnings.append(f"mic start failed: {exc}")
            self._mic_source = None

        if not started_parts:
            self.stop_sources(clear_status=False)
            self._set_status("Source test start failed. " + " | ".join(warnings))
            return

        self._poll_timer.start()
        base = (
            f"Started {', '.join(started_parts)} | "
            f"cam id={config.camera_id} res={config.camera_width}x{config.camera_height} | "
            f"mic={config.mic_device if config.mic_device is not None else 'default'}"
        )
        if config.note:
            base += f" | {config.note}"
        if warnings:
            base += " | " + " | ".join(warnings)
        self._set_status(base)

    def stop_sources(self, clear_status: bool = True) -> None:
        self._poll_timer.stop()

        if self._mic_source is not None:
            try:
                self._mic_source.stop()
            except Exception:
                pass
            self._mic_source = None

        if self._camera_source is not None:
            try:
                self._camera_source.stop()
            except Exception:
                pass
            self._camera_source = None

        self._bus = None
        self._raw_camera_pixmap = None
        self.camera_label.setPixmap(QPixmap())
        self.camera_label.setText("Camera preview is stopped.")
        self.camera_info_label.setText("camera: -")
        self.rms_bar.setValue(0)
        self.rms_label.setText("RMS 0.000 (-60.0 dB)")

        if clear_status:
            self._set_status("Source test stopped.")

    def restart_sources(self) -> None:
        self.stop_sources(clear_status=False)
        self.start_sources()

    def _poll_sources(self) -> None:
        if self._bus is None:
            return

        frame_pkt, frame_ver = self._bus.wait_latest("frame/raw", self._last_frame_ver, timeout=0.0)
        if frame_pkt is not None:
            self._last_frame_ver = frame_ver
            self._update_camera_preview(frame_pkt.image)
            meta = frame_pkt.meta if isinstance(frame_pkt.meta, dict) else {}
            cam_id = meta.get("camera_id", "-")
            cam_fps = float(meta.get("cam_fps", 0.0))
            actual_w = int(meta.get("actual_width", 0))
            actual_h = int(meta.get("actual_height", 0))
            self.camera_info_label.setText(
                f"camera_id={cam_id} | actual={actual_w}x{actual_h} | fps={cam_fps:.1f}"
            )

        audio_pkt, audio_ver = self._bus.wait_latest("audio/raw", self._last_audio_ver, timeout=0.0)
        if audio_pkt is not None:
            self._last_audio_ver = audio_ver
            self._update_rms(audio_pkt.audio)
        else:
            self._decay_rms()

    def _update_camera_preview(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return

        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._raw_camera_pixmap = QPixmap.fromImage(qimage)
        self._refresh_camera_pixmap()

    def _refresh_camera_pixmap(self) -> None:
        if self._raw_camera_pixmap is None:
            return
        scaled = self._raw_camera_pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_label.setPixmap(scaled)
        self.camera_label.setText("")

    def _update_rms(self, audio: np.ndarray) -> None:
        if audio is None:
            return
        signal = np.asarray(audio)
        if signal.ndim > 1:
            signal = signal[:, 0]
        if signal.size == 0:
            return

        rms_raw = float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))
        level = max(0.0, min(1.0, rms_raw * 3.0))
        self._last_audio_level = level
        self._last_audio_update_ts = time.time()
        self._render_rms(level)

    def _decay_rms(self) -> None:
        if self._last_audio_update_ts <= 0:
            return
        if time.time() - self._last_audio_update_ts < 0.2:
            return
        self._last_audio_level *= 0.90
        if self._last_audio_level < 0.003:
            self._last_audio_level = 0.0
        self._render_rms(self._last_audio_level)

    def _render_rms(self, level: float) -> None:
        self.rms_bar.setValue(int(level * 100))
        if level < 0.6:
            color = "#46d25a"
        elif level < 0.85:
            color = "#dcc850"
        else:
            color = "#e65050"
        self.rms_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #666; background: #1a1a1a; }"
            f"QProgressBar::chunk {{ background-color: {color}; }}"
        )
        db = max(-60.0, 20.0 * np.log10(level + 1e-10))
        self.rms_label.setText(f"RMS {level:.3f} ({db:.1f} dB)")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_camera_pixmap()

    def closeEvent(self, event) -> None:
        self.stop_sources(clear_status=False)
        event.accept()


def resolve_dotenv_path(dotenv_path_arg: Optional[str]) -> Path:
    if dotenv_path_arg:
        return Path(dotenv_path_arg).expanduser().resolve()

    found = find_dotenv(usecwd=True)
    if found:
        return Path(found).resolve()
    return (PROJECT_ROOT / ".env").resolve()


def main() -> None:
    _suppress_opencv_warnings()

    parser = argparse.ArgumentParser(description="NeuroFlow deviceMngUI")
    parser.add_argument("--dotenv-path", type=str, default=None, help="Path to .env file")
    args = parser.parse_args()

    dotenv_path = resolve_dotenv_path(args.dotenv_path)
    app = QApplication(sys.argv)
    win = DeviceMngUI(dotenv_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
