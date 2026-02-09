from __future__ import annotations

import sys
from typing import Optional

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
)

from cv2_enumerate_cameras import enumerate_cameras

from visionflow.pipeline.bus import TopicBus
from visionflow.pipeline.packet import FramePacket
from visionflow.sources.camera_source import CameraSource


# =====================================================
# enumerate-cameras 결과를 "있는 그대로" 노출
# =====================================================

def enumerate_all_camera_entries():
    cams = enumerate_cameras()
    results = []

    for cam in cams:
        label = (
            f"[idx={cam.index}] {cam.name} "
            f"(vid={cam.vid} pid={cam.pid})"
        )
        results.append(
            {
                "label": label,
                "camera_id": cam.index,
                "vid": cam.vid,
                "pid": cam.pid,
                "name": cam.name,
            }
        )

    # 보기 좋게 index 기준 정렬
    results.sort(key=lambda x: x["camera_id"])
    return results


# =====================================================
# PySide6 UI
# =====================================================

class CameraViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionFlow Camera Viewer (All Indices)")
        self.resize(900, 600)

        # UI
        self.combo = QComboBox()
        self.btn_play = QPushButton("Play")
        self.btn_stop = QPushButton("Stop")
        self.video = QLabel("No Camera")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:black; color:white;")

        top = QHBoxLayout()
        top.addWidget(self.combo)
        top.addWidget(self.btn_play)
        top.addWidget(self.btn_stop)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.video, 1)

        # VisionFlow
        self.bus = TopicBus()
        self.camera: Optional[CameraSource] = None

        self.topic = "frame/raw"
        self.last_ver = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_frame)

        # ---- populate ALL camera indices ----
        for cam in enumerate_all_camera_entries():
            self.combo.addItem(cam["label"], cam)

        # signals
        self.btn_play.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)

    # -------------------------------------------------

    def start_camera(self):
        self.stop_camera()

        cam = self.combo.currentData()
        if not cam:
            return

        camera_id = cam["camera_id"]

        self.camera = CameraSource(
            bus=self.bus,
            out_topic=self.topic,
            camera_id=camera_id,
            request_width=1280,
            request_height=720,
            use_dshow=False,  # enumerate index → auto backend 필수
            source_id=f"cam{camera_id}",
        )

        try:
            self.camera.start()
        except Exception as e:
            self.video.setText(str(e))
            self.camera = None
            return

        self.last_ver = self.bus.get_version(self.topic)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.camera:
            self.camera.stop()
            self.camera = None
        self.video.setText("Stopped")

    # -------------------------------------------------
    # TopicBus polling
    # -------------------------------------------------

    def poll_frame(self):
        pkt, ver = self.bus.wait_latest(
            self.topic,
            self.last_ver,
            timeout=0.0,  # non-blocking
        )
        if pkt is None:
            return

        self.last_ver = ver
        self.render(pkt)

    def render(self, pkt: FramePacket):
        frame = pkt.image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video.setPixmap(pix)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


# =====================================================
# main
# =====================================================

def main():
    app = QApplication(sys.argv)
    win = CameraViewer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
