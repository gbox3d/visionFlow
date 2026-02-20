from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cv2

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


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
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except Exception:
        pass


def _probe_camera(camera_id: int, use_dshow: bool) -> bool:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def list_cameras_hwinfo() -> list[CameraDevice]:
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception:
        return []

    cameras = list(enumerate_cameras())
    backend_ranges = {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_MSMF: "MSMF",
    }
    out: list[CameraDevice] = []

    for cam in cameras:
        if cam.backend and cam.backend in backend_ranges:
            backend_name = backend_ranges[cam.backend]
            cam_id = cam.index % 100
        else:
            matched = False
            for base, name in sorted(backend_ranges.items(), reverse=True):
                if cam.index >= base:
                    backend_name = name
                    cam_id = cam.index - base
                    matched = True
                    break
            if not matched:
                backend_name = "AUTO"
                cam_id = cam.index

        vid = f"{cam.vid:04X}" if cam.vid else "----"
        pid = f"{cam.pid:04X}" if cam.pid else "----"
        out.append(
            CameraDevice(
                camera_id=cam_id,
                name=str(cam.name or "-"),
                vid_pid=f"{vid}:{pid}",
                backend=backend_name,
                path=str(cam.path or "-"),
            )
        )
    return out


def list_cameras_probe(max_devices: int, use_dshow: bool) -> list[CameraDevice]:
    out: list[CameraDevice] = []
    for cam_id in range(max(0, max_devices)):
        ok = _probe_camera(cam_id, use_dshow=use_dshow)
        out.append(
            CameraDevice(
                camera_id=cam_id,
                name="(probe)",
                vid_pid="-",
                backend="DSHOW" if use_dshow else "AUTO",
                path="OK" if ok else "--",
            )
        )
    return out


def list_microphones() -> list[MicrophoneDevice]:
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
        default_input = sd.query_devices(kind="input")
        default_idx = int(default_input.get("index", -1))
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


class DeviceListerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NeuroFlow Device Lister (UI)")
        self.resize(1100, 700)

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)

        top_row = QHBoxLayout()
        self.chk_probe = QCheckBox("Camera Probe Mode")
        self.chk_probe.setChecked(False)
        self.chk_dshow = QCheckBox("Use DSHOW")
        self.chk_dshow.setChecked(True)

        self.spin_max = QSpinBox()
        self.spin_max.setRange(1, 64)
        self.spin_max.setValue(10)
        self.spin_max.setFixedWidth(80)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_all)
        self.btn_copy_camera_path = QPushButton("Copy Camera Path")
        self.btn_copy_camera_path.clicked.connect(self.copy_selected_camera_path)

        top_row.addWidget(self.chk_probe)
        top_row.addWidget(self.chk_dshow)
        top_row.addWidget(QLabel("Max Camera Index"))
        top_row.addWidget(self.spin_max)
        top_row.addStretch(1)
        top_row.addWidget(self.btn_copy_camera_path)
        top_row.addWidget(self.btn_refresh)

        main_layout.addLayout(top_row)

        self.tabs = QTabWidget()
        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(5)
        self.camera_table.setHorizontalHeaderLabels(["camera_id", "name", "VID:PID", "backend", "path/status"])
        self.camera_table.horizontalHeader().setStretchLastSection(True)
        self.camera_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.camera_table.setSelectionMode(QTableWidget.SingleSelection)
        self.camera_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.camera_table.cellDoubleClicked.connect(self._on_camera_cell_double_clicked)

        self.mic_table = QTableWidget()
        self.mic_table.setColumnCount(5)
        self.mic_table.setHorizontalHeaderLabels(["mic_id", "name", "hostapi", "channels", "default"])
        self.mic_table.horizontalHeader().setStretchLastSection(True)
        self.mic_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.mic_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.tabs.addTab(self.camera_table, "Camera")
        self.tabs.addTab(self.mic_table, "Microphone")
        main_layout.addWidget(self.tabs)

        self.status_label = QLabel("")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        main_layout.addWidget(self.status_label)
        self._status_base = ""

        self.refresh_all()

    def _set_camera_rows(self, items: list[CameraDevice]) -> None:
        self.camera_table.setRowCount(len(items))
        for row, item in enumerate(items):
            values = [
                str(item.camera_id),
                item.name,
                item.vid_pid,
                item.backend,
                item.path,
            ]
            for col, value in enumerate(values):
                self.camera_table.setItem(row, col, QTableWidgetItem(value))
        self.camera_table.resizeColumnsToContents()

    def _set_mic_rows(self, items: list[MicrophoneDevice]) -> None:
        self.mic_table.setRowCount(len(items))
        for row, item in enumerate(items):
            values = [
                str(item.mic_id),
                item.name,
                item.hostapi,
                str(item.channels),
                "Y" if item.is_default else "",
            ]
            for col, value in enumerate(values):
                self.mic_table.setItem(row, col, QTableWidgetItem(value))
        self.mic_table.resizeColumnsToContents()

    def refresh_all(self) -> None:
        probe_mode = self.chk_probe.isChecked()
        use_dshow = self.chk_dshow.isChecked()
        max_devices = self.spin_max.value()

        cameras = (
            list_cameras_probe(max_devices=max_devices, use_dshow=use_dshow)
            if probe_mode
            else list_cameras_hwinfo()
        )
        microphones = list_microphones()

        self._set_camera_rows(cameras)
        self._set_mic_rows(microphones)

        self._status_base = (
            f"cameras={len(cameras)} | microphones={len(microphones)} | "
            f"probe={'ON' if probe_mode else 'OFF'}"
        )
        self.status_label.setText(self._status_base)

    def _on_camera_cell_double_clicked(self, row: int, col: int) -> None:
        if col == 4:
            self.copy_selected_camera_path()

    def copy_selected_camera_path(self) -> None:
        selected = self.camera_table.selectedItems()
        if not selected:
            self.status_label.setText(f"{self._status_base} | select a camera row first")
            return

        row = selected[0].row()
        path_item = self.camera_table.item(row, 4)
        if path_item is None:
            self.status_label.setText(f"{self._status_base} | selected row has no path")
            return

        path_text = (path_item.text() or "").strip()
        if path_text in ("", "-", "--", "OK"):
            self.status_label.setText(f"{self._status_base} | path is not available in this mode")
            return

        QApplication.clipboard().setText(path_text)
        self.status_label.setText(f"{self._status_base} | copied camera path to clipboard")


def main() -> None:
    _suppress_opencv_warnings()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--max-devices", type=int, default=10)
    parser.add_argument("--dshow", type=int, default=1)
    args, _ = parser.parse_known_args()

    if args.dump:
        cameras = (
            list_cameras_probe(max_devices=args.max_devices, use_dshow=(args.dshow != 0))
            if args.probe
            else list_cameras_hwinfo()
        )
        microphones = list_microphones()
        print(f"cameras={len(cameras)}")
        for c in cameras:
            print(f"camera_id={c.camera_id} name={c.name} backend={c.backend} path={c.path}")
        print(f"microphones={len(microphones)}")
        for m in microphones:
            print(f"mic_id={m.mic_id} name={m.name} hostapi={m.hostapi} channels={m.channels} default={m.is_default}")
        return

    app = QApplication(sys.argv)
    win = DeviceListerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
