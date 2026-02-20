from __future__ import annotations

import time
import threading
from typing import Optional

import cv2

from visionflow.pipeline.bus import TopicBus
from visionflow.pipeline.packet import FramePacket
from visionflow.utils.etc import normalize_device_path_key


class CameraSource:
    """
    카메라 캡처 전용 소스
    - 프레임을 out_topic으로 계속 publish
    - read 실패 누적 시 자동 재연결
    - device_path 지정 시 path 기반으로 카메라를 식별하여 연결
    """

    def __init__(
        self,
        bus: TopicBus,
        out_topic: str = "frame/raw",
        camera_id: int = 0,
        device_path: Optional[str] = None,
        request_width: int = 640,
        request_height: int = 480,
        max_fail: int = 60,
        reconnect_sleep_s: float = 1.0,
        use_dshow: bool = True,
        source_id: Optional[str] = None,
        quiet_opencv_log: bool = True,
    ):
        self.bus = bus
        self.out_topic = out_topic

        self.device_path = device_path
        self.request_width = int(request_width)
        self.request_height = int(request_height)

        self.max_fail = int(max_fail)
        self.reconnect_sleep_s = float(reconnect_sleep_s)
        self.use_dshow = bool(use_dshow)

        # device_path가 지정된 경우 path → index 해상
        if self.device_path:
            resolved = self._resolve_index_from_path(self.device_path)
            if resolved is not None:
                self.camera_id = resolved
            else:
                print(
                    f"[CameraSource] device_path 해상 실패, "
                    f"camera_id={camera_id} 사용: {self.device_path}"
                )
                self.camera_id = int(camera_id)
        else:
            self.camera_id = int(camera_id)

        self.source_id = source_id or f"camera{self.camera_id}"

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._seq = 0
        self._last_ok_ts = 0
        self._actual_width = 0
        self._actual_height = 0

        # cam fps
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._cam_fps = 0.0

        if quiet_opencv_log:
            self._suppress_opencv_warnings()

    def _resolve_index_from_path(self, device_path: str) -> Optional[int]:
        """
        device_path를 기반으로 현재 시스템의 OpenCV camera index를 찾는다.
        cv2-enumerate-cameras로 전체 디바이스를 열거한 뒤,
        path의 핵심 부분(GUID 앞까지)을 비교하여 매칭한다.

        Returns:
            매칭된 camera index (int) 또는 None
        """
        try:
            from cv2_enumerate_cameras import enumerate_cameras
        except ImportError:
            print("[CameraSource] cv2-enumerate-cameras 패키지가 필요합니다")
            return None

        target_key = normalize_device_path_key(device_path)
        prefer_backend = cv2.CAP_DSHOW if self.use_dshow else cv2.CAP_MSMF

        candidates = []
        for cam in enumerate_cameras():
            if cam.path and normalize_device_path_key(cam.path) == target_key:
                candidates.append(cam)

        if not candidates:
            return None

        # 선호 backend에 맞는 것 우선 선택
        for cam in candidates:
            if cam.index >= prefer_backend and cam.index < prefer_backend + 100:
                return cam.index - prefer_backend

        # 못 찾으면 첫 번째 것에서 index 추출
        cam = candidates[0]
        if cam.index >= cv2.CAP_MSMF:
            return cam.index - cv2.CAP_MSMF
        elif cam.index >= cv2.CAP_DSHOW:
            return cam.index - cv2.CAP_DSHOW
        return cam.index

    def _suppress_opencv_warnings(self):
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

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if not self._open():
            self._running = False
            raise RuntimeError(f"카메라 오픈 실패 (id={self.camera_id})")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release()

    def _backend(self) -> int:
        return cv2.CAP_DSHOW if self.use_dshow else 0

    def _open(self) -> bool:
        self._release()
        cap = cv2.VideoCapture(self.camera_id, self._backend())
        if not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.request_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.request_height)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with self._lock:
            self._cap = cap
            self._actual_width = w
            self._actual_height = h
        return True

    def _release(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = None
            self._actual_width = 0
            self._actual_height = 0

    def _reconnect(self) -> None:
        self._release()
        time.sleep(self.reconnect_sleep_s)

        # device_path가 있으면 재연결 시마다 index를 재해상
        # (USB 재연결 등으로 index가 바뀌었을 수 있음)
        if self.device_path:
            resolved = self._resolve_index_from_path(self.device_path)
            if resolved is not None and resolved != self.camera_id:
                print(
                    f"[CameraSource] device_path 재해상: "
                    f"id {self.camera_id} → {resolved}"
                )
                self.camera_id = resolved

        ok = self._open()
        if ok:
            print(
                f"[CameraSource] 재연결 성공: id={self.camera_id} "
                f"actual={self._actual_width}x{self._actual_height}"
            )
        else:
            print(f"[CameraSource] 재연결 실패: id={self.camera_id}")

    def _loop(self) -> None:
        fail = 0
        while self._running:
            with self._lock:
                cap = self._cap

            if cap is None:
                self._reconnect()
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                fail += 1
                if fail % 30 == 0:
                    print(f"[CameraSource] camera.read() 실패 {fail}회 누적")
                if fail >= self.max_fail:
                    print("[CameraSource] read 지속 실패 → 재연결 시도")
                    self._reconnect()
                    fail = 0
                time.sleep(0.05)
                continue

            fail = 0
            self._seq += 1
            self._last_ok_ts = int(time.time() * 1000)

            # cam fps
            self._fps_n += 1
            now = time.time()
            dt = now - self._fps_t0
            if dt >= 1.0:
                self._cam_fps = self._fps_n / dt
                self._fps_n = 0
                self._fps_t0 = now

            pkt = FramePacket(
                image=frame,
                ts_ms=self._last_ok_ts,
                seq=self._seq,
                source_id=self.source_id,
                meta={
                    "camera_id": self.camera_id,
                    "request_width": self.request_width,
                    "request_height": self.request_height,
                    "actual_width": self._actual_width,
                    "actual_height": self._actual_height,
                    "cam_fps": self._cam_fps,
                },
            )
            self.bus.publish(self.out_topic, pkt)
            time.sleep(0.001)
