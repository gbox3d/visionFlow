from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
from dotenv import load_dotenv

# Ensure local src/ imports work when running `python main.py` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_demo_env() -> Path | None:
    """
    Load .env with distribution-friendly priority:
    1) beside executable (frozen build)
    2) project root (dev / bundled fallback)
    3) current working directory
    """
    candidates: list[Path] = []

    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / ".env")

    candidates.append(PROJECT_ROOT / ".env")
    candidates.append(Path.cwd() / ".env")

    seen: set[Path] = set()
    for env_path in candidates:
        try:
            key = env_path.resolve()
        except Exception:
            key = env_path
        if key in seen:
            continue
        seen.add(key)
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return env_path
    return None


from visionflow.pipeline.bus import TopicBus
from visionflow.sources.camera_source import CameraSource
from visionflow.utils.draw_utils import Colors, DrawUtils
from visionflow.utils.etc import parse_resolution, resource_path
from visionflow.utils.image import cv_bgr_to_pygame_surface
from visionflow.workers.mp_face_detector_async_worker import MpFaceDetectorAsyncWorker
from visionflow.workers.mp_face_landmarker_async_worker import MpFaceLandmarkerAsyncWorker
from visionflow.workers.mp_pose_landmarker_async_worker import MpPoseLandmarkerAsyncWorker

from voiceFlow.utils.audio_device import parse_mic_device, resolve_mic_device_from_camera_path
from voiceFlow.utils.env import (
    env_bool_any,
    env_float_any,
    env_int_any,
    env_lang_any,
    env_str_any,
)
from voiceFlow.utils.text import wrap_text
from voiceFlow.sources.microphone_source import MicrophoneSource
from voiceFlow.processors.miso_stt_asr import MisoSttAsrProcessor
from voiceFlow.workers.accumulate_asr_worker import AccumulateAsrWorker

class DemoApplication:
    APP_NAME = "Kiosk Demo Vision + STT"
    APP_VERSION = "v1.0.0"
    APP_TITLE = f"{APP_NAME} {APP_VERSION}"

    def __init__(self) -> None:
        self._loaded_env_path = _load_demo_env()

        self.panel_height = env_int_any(("PANEL_HEIGHT", "DEMO_PANEL_HEIGHT"), 220)
        self.camera_top_ratio = max(
            0.35,
            min(0.75, env_float_any(("CAMERA_TOP_RATIO", "DEMO_CAMERA_TOP_RATIO"), 0.52)),
        )
        self.show_help = True
        self.enable_face_detection = True
        self.enable_face_landmark = True
        self.enable_pose_landmark = True
        self._is_fullscreen = False
        self._windowed_size = (1280, 860)

        self._bus: Optional[TopicBus] = None
        self._camera: Optional[CameraSource] = None
        self._face_worker: Optional[MpFaceDetectorAsyncWorker] = None
        self._landmark_worker: Optional[MpFaceLandmarkerAsyncWorker] = None
        self._pose_worker: Optional[MpPoseLandmarkerAsyncWorker] = None
        self._mic_source: Optional[MicrophoneSource] = None
        self._asr_processor: Optional[MisoSttAsrProcessor] = None
        self._asr_worker: Optional[AccumulateAsrWorker] = None

        self._running = False

        self._last_raw_ver = 0
        self._last_audio_ver = 0
        self._last_asr_ver = 0
        self._last_asr_status_ver = 0

        self._last_raw_pkt = None
        self._last_surface = None

        self._latest_asr_text = ""
        self._latest_lang = "-"
        self._latest_infer_ms = 0.0
        self._latest_buffer_s = 0.0
        self._latest_queue_chunks = 0
        self._latest_warmup_pct = 0.0
        self._latest_warmup_buffer_s = 0.0
        self._latest_warmup_target_s = 0.0
        self._latest_warmup_done = False
        self._latest_rms = 0.0
        self._latest_rms_peak = 0.0
        self._latest_rms_peak_ts = 0.0

        self._disp_count = 0
        self._disp_fps = 0.0
        self._disp_t0 = time.time()

        self._status_text = "대기 중"
        self._runtime_info = ""

        pygame.init()
        pygame.display.set_caption(self.APP_TITLE)

        self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.font_title, self.font_body, self.font_small = self._load_fonts()

    def _load_fonts(self) -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
        font_path = PROJECT_ROOT / "font" / "DungGeunMo.ttf"
        if font_path.exists():
            return (
                pygame.font.Font(str(font_path), 22),
                pygame.font.Font(str(font_path), 16),
                pygame.font.Font(str(font_path), 14),
            )
        return (
            pygame.font.SysFont("Consolas", 22),
            pygame.font.SysFont("Consolas", 16),
            pygame.font.SysFont("Consolas", 14),
        )

    def _start_pipeline(self) -> None:
        self._bus = TopicBus()
        self._last_raw_ver = self._bus.get_version("frame/raw")
        self._last_audio_ver = self._bus.get_version("audio/raw")
        self._last_asr_ver = self._bus.get_version("text/asr")
        self._last_asr_status_ver = self._bus.get_version("text/asr_status")
        self._latest_rms = 0.0
        self._latest_rms_peak = 0.0
        self._latest_rms_peak_ts = 0.0

        max_face = max(1, env_int_any(("MAX_FACE",), 1))
        max_pose = max(1, env_int_any(("MAX_POSE",), 1))

        camera_id = env_int_any(("CAMERA_ID", "DEMO_CAMERA_ID"), 0)
        unified_path = env_str_any(("DEVICE_PATH", "DEMO_DEVICE_PATH"), "").strip()
        camera_path = unified_path or env_str_any(("CAMERA_DEVICE_PATH", "DEMO_CAMERA_DEVICE_PATH"), "").strip() or None
        camera_res = parse_resolution(env_str_any(("CAMERA_RESOLUTION", "DEMO_CAMERA_RESOLUTION"), ""))
        if camera_res is not None:
            camera_width, camera_height = camera_res
        else:
            camera_width = env_int_any(("CAMERA_WIDTH", "DEMO_CAMERA_WIDTH"), 640)
            camera_height = env_int_any(("CAMERA_HEIGHT", "DEMO_CAMERA_HEIGHT"), 480)
        camera_dshow = env_bool_any(("CAMERA_USE_DSHOW", "DEMO_CAMERA_USE_DSHOW"), True)

        mic_device = parse_mic_device(env_str_any(("MIC_DEVICE", "DEMO_MIC_DEVICE"), ""))
        if mic_device is None and camera_path:
            mic_device = resolve_mic_device_from_camera_path(camera_path)
        mic_sr = env_int_any(("MIC_SAMPLERATE", "DEMO_MIC_SAMPLERATE", "VOICEFLOW_STT_SAMPLERATE"), 16000)
        mic_block = env_int_any(("MIC_BLOCKSIZE", "DEMO_MIC_BLOCKSIZE"), 1024)

        stt_backend = env_str_any(("VOICEFLOW_STT_BACKEND",), "ct2")
        stt_model = env_str_any(("VOICEFLOW_STT_MODEL",), "large-v3")
        stt_model_path = env_str_any(("VOICEFLOW_STT_MODEL_PATH",), "").strip() or None
        stt_device = env_str_any(("VOICEFLOW_STT_DEVICE",), "auto")
        stt_fp16 = env_bool_any(("VOICEFLOW_STT_FP16",), True)
        stt_lang = env_lang_any(("VOICEFLOW_STT_LANGUAGE",), "auto")
        stt_task = env_str_any(("VOICEFLOW_STT_TASK",), "transcribe")
        stt_beam = env_int_any(("VOICEFLOW_STT_BEAM_SIZE",), 5)
        stt_temp = env_float_any(("VOICEFLOW_STT_TEMPERATURE",), 0.0)
        stt_vad = env_bool_any(("VOICEFLOW_STT_CT2_VAD_FILTER",), False)

        asr_step_s = env_float_any(("ASR_STEP_S", "DEMO_ASR_STEP_S"), 1.5)
        asr_max_s = env_float_any(("ASR_MAX_WINDOW_S", "DEMO_ASR_MAX_WINDOW_S"), 25.0)
        enable_logging = env_bool_any(("ENABLE_LOGGING",), False)

        face_model_path = resource_path(
            env_str_any(("FACE_MODEL_PATH", "DEMO_FACE_MODEL_PATH"), "models/blaze_face_short_range.tflite")
        )
        landmark_model_path = resource_path(
            env_str_any(("FACE_LANDMARK_MODEL_PATH", "DEMO_FACE_LANDMARK_MODEL_PATH"), "models/face_landmarker.task")
        )
        pose_model_path = resource_path(
            env_str_any(("POSE_MODEL_PATH", "DEMO_POSE_MODEL_PATH"), "models/pose_landmarker.task")
        )

        self._runtime_info = (
            f"cam(path={camera_path or '-'}, id={camera_id}, req={camera_width}x{camera_height}) | "
            f"max_face={max_face} max_pose={max_pose} | "
            f"mic={mic_device if mic_device is not None else 'default'} | "
            f"stt={stt_backend}/{stt_model} | "
            f"env={self._loaded_env_path if self._loaded_env_path is not None else '-'}"
        )

        self._camera = CameraSource(
            bus=self._bus,
            out_topic="frame/raw",
            camera_id=camera_id,
            device_path=camera_path,
            request_width=camera_width,
            request_height=camera_height,
            use_dshow=camera_dshow,
            source_id=f"camera{camera_id}",
        )

        self._face_worker = MpFaceDetectorAsyncWorker(
            bus=self._bus,
            in_topic="frame/raw",
            out_topic="frame/face",
            model_path=face_model_path,
            model_options={"min_detection_confidence": 0.5},
            max_faces=max_face,
            name="demo-face-detector",
        )

        self._landmark_worker = MpFaceLandmarkerAsyncWorker(
            bus=self._bus,
            in_topic="frame/face",
            out_topic="frame/landmark",
            model_path=landmark_model_path,
            model_options={
                "num_faces": max_face,
                "min_face_detection_confidence": 0.5,
                "min_face_presence_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            use_roi_crop=(max_face <= 1),
            roi_padding=0.3,
            name="demo-face-landmarker",
        )

        self._pose_worker = MpPoseLandmarkerAsyncWorker(
            bus=self._bus,
            in_topic="frame/raw",
            out_topic="frame/pose",
            model_path=pose_model_path,
            model_options={
                "num_poses": max_pose,
                "min_pose_detection_confidence": 0.5,
                "min_pose_presence_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            name="demo-pose-landmarker",
        )

        self._mic_source = MicrophoneSource(
            bus=self._bus,
            out_topic="audio/raw",
            samplerate=mic_sr,
            channels=1,
            blocksize=mic_block,
            device=mic_device,
            source_id="demo-mic",
        )

        backend_kwargs = {"temperature": stt_temp}
        if stt_backend == "ct2":
            backend_kwargs["vad_filter"] = stt_vad

        self._asr_processor = MisoSttAsrProcessor(
            backend=stt_backend,
            model_name=stt_model,
            model_path=stt_model_path,
            device=stt_device,
            fp16=stt_fp16,
            language=stt_lang,
            task=stt_task,
            beam_size=stt_beam,
            backend_kwargs=backend_kwargs,
        )

        self._status_text = f"ASR 모델 로딩 중... ({stt_backend}/{stt_model})"
        self._asr_processor.warmup()

        self._asr_worker = AccumulateAsrWorker(
            bus=self._bus,
            processor=self._asr_processor,
            in_topic="audio/raw",
            out_topic="text/asr",
            status_topic="text/asr_status",
            step_s=asr_step_s,
            max_window_s=asr_max_s,
            samplerate=mic_sr,
            enable_logging=enable_logging,
            log_dir=str(PROJECT_ROOT / "logs"),
            name="demo-asr-worker",
        )

        self._camera.start()
        self._face_worker.start()
        self._landmark_worker.start()
        self._pose_worker.start()
        self._mic_source.start()
        self._asr_worker.start()

        self._running = True
        self._status_text = "실행 중"

    def _stop_pipeline(self) -> None:
        if self._asr_worker is not None:
            self._asr_worker.stop()
            self._asr_worker = None

        if self._mic_source is not None:
            self._mic_source.stop()
            self._mic_source = None

        if self._pose_worker is not None:
            self._pose_worker.stop()
            self._pose_worker = None

        if self._landmark_worker is not None:
            self._landmark_worker.stop()
            self._landmark_worker = None

        if self._face_worker is not None:
            self._face_worker.stop()
            self._face_worker = None

        if self._camera is not None:
            self._camera.stop()
            self._camera = None

        if self._asr_processor is not None:
            try:
                self._asr_processor.close()
            except Exception:
                pass
            self._asr_processor = None

        self._bus = None
        self._running = False

    def _update_from_bus(self) -> None:
        if self._bus is None:
            return

        pkt, ver = self._bus.wait_latest("frame/raw", self._last_raw_ver, timeout=0.0)
        if pkt is not None:
            self._last_raw_pkt = pkt
            self._last_raw_ver = ver
            try:
                self._last_surface = cv_bgr_to_pygame_surface(pkt.image)
            except Exception:
                self._last_surface = None

        audio_pkt, audio_ver = self._bus.wait_latest("audio/raw", self._last_audio_ver, timeout=0.0)
        if audio_pkt is not None:
            self._last_audio_ver = audio_ver
            audio = audio_pkt.audio
            if audio.ndim > 1:
                audio = audio[:, 0]
            if len(audio) > 0:
                rms_raw = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
                level = max(0.0, min(1.0, rms_raw * 3.0))
                self._latest_rms = level
                now = time.time()
                if level >= self._latest_rms_peak:
                    self._latest_rms_peak = level
                    self._latest_rms_peak_ts = now
                elif now - self._latest_rms_peak_ts > 1.2:
                    self._latest_rms_peak = max(level, self._latest_rms_peak * 0.94)

        status_payload, status_ver = self._bus.wait_latest(
            "text/asr_status", self._last_asr_status_ver, timeout=0.0
        )
        if status_payload is not None:
            self._last_asr_status_ver = status_ver
            data = status_payload if isinstance(status_payload, dict) else {}
            self._latest_warmup_pct = float(data.get("warmup_pct", 0.0))
            self._latest_warmup_buffer_s = float(data.get("buffer_s", 0.0))
            self._latest_warmup_target_s = float(data.get("target_s", 0.0))
            self._latest_warmup_done = bool(data.get("warmed", False))

        asr_pkt, asr_ver = self._bus.wait_latest("text/asr", self._last_asr_ver, timeout=0.0)
        if asr_pkt is not None:
            self._last_asr_ver = asr_ver
            self._latest_asr_text = str(asr_pkt.meta.get("full_text") or asr_pkt.text or "").strip()
            self._latest_lang = str(asr_pkt.language or "-")
            self._latest_infer_ms = float(asr_pkt.meta.get("infer_ms", 0.0))
            self._latest_buffer_s = float(asr_pkt.meta.get("buffer_s", 0.0))
            self._latest_queue_chunks = int(asr_pkt.meta.get("queue_chunks", 0))

    def _draw_rms_bar(self, panel_rect: pygame.Rect, line_y: int) -> int:
        bar_x = panel_rect.x + 12
        bar_w = max(180, panel_rect.w - 24)
        bar_h = 16
        bar_y = line_y

        pygame.draw.rect(self.screen, (40, 40, 40), pygame.Rect(bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * self._latest_rms)
        if fill_w > 0:
            if self._latest_rms < 0.6:
                fill_color = (70, 210, 90)
            elif self._latest_rms < 0.85:
                fill_color = (220, 200, 80)
            else:
                fill_color = (230, 80, 80)
            pygame.draw.rect(self.screen, fill_color, pygame.Rect(bar_x, bar_y, fill_w, bar_h))

        peak_x = bar_x + int(bar_w * self._latest_rms_peak)
        if peak_x > bar_x + 1:
            pygame.draw.line(self.screen, (255, 255, 255), (peak_x, bar_y), (peak_x, bar_y + bar_h), 2)

        db = max(-60.0, 20.0 * np.log10(self._latest_rms + 1e-10))
        txt = f"mic RMS {self._latest_rms:.3f} ({db:.1f} dB)"
        self.screen.blit(self.font_small.render(txt, True, (190, 190, 230)), (bar_x, bar_y + bar_h + 4))

        return bar_y + bar_h + 24

    def _draw_vision(self, camera_rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (0, 0, 0), camera_rect)

        if self._last_surface is None or self._last_raw_pkt is None:
            txt = self.font_title.render("카메라 프레임 대기 중...", True, (180, 180, 180))
            self.screen.blit(
                txt,
                (
                    camera_rect.x + (camera_rect.w - txt.get_width()) // 2,
                    camera_rect.y + (camera_rect.h - txt.get_height()) // 2,
                ),
            )
            return

        raw_pkt = self._last_raw_pkt
        img_h, img_w = raw_pkt.image.shape[:2]
        if img_w <= 0 or img_h <= 0:
            return

        scale = min(camera_rect.w / img_w, camera_rect.h / img_h)
        draw_w = max(1, int(img_w * scale))
        draw_h = max(1, int(img_h * scale))
        draw_x = camera_rect.x + (camera_rect.w - draw_w) // 2
        draw_y = camera_rect.y + (camera_rect.h - draw_h) // 2
        draw_rect = pygame.Rect(draw_x, draw_y, draw_w, draw_h)

        camera_layer = pygame.transform.scale(self._last_surface, (draw_rect.w, draw_rect.h))

        sx = draw_rect.w / max(1, img_w)
        sy = draw_rect.h / max(1, img_h)

        face_pkt = self._bus.get_latest("frame/face") if self._bus else None
        landmark_pkt = self._bus.get_latest("frame/landmark") if self._bus else None
        pose_pkt = self._bus.get_latest("frame/pose") if self._bus else None

        if self.enable_face_detection and face_pkt is not None:
            face_rois = face_pkt.meta.get("face_rois")
            if isinstance(face_rois, list) and len(face_rois) > 0:
                for idx, face_info in enumerate(face_rois):
                    try:
                        x, y, rw, rh, score = face_info
                    except Exception:
                        continue
                    label = f"Face{idx+1}: {float(score):.2f}" if idx < 3 else None
                    DrawUtils.draw_face_roi(
                        screen=camera_layer,
                        roi=(int(x), int(y), int(rw), int(rh)),
                        scale_x=sx,
                        scale_y=sy,
                        color=Colors.GREEN,
                        thickness=2,
                        label=label,
                        font=self.font_small,
                    )
            elif face_pkt.roi is not None:
                score = float(face_pkt.meta.get("face_score", 0.0))
                DrawUtils.draw_face_roi(
                    screen=camera_layer,
                    roi=face_pkt.roi,
                    scale_x=sx,
                    scale_y=sy,
                    color=Colors.GREEN,
                    thickness=2,
                    label=f"Face: {score:.2f}",
                    font=self.font_small,
                )

        if self.enable_face_landmark and landmark_pkt is not None:
            face_landmarks_data = landmark_pkt.meta.get("face_landmarks") or []
            DrawUtils.draw_face_landmarks(
                screen=camera_layer,
                face_landmarks_data=face_landmarks_data,
                img_w=img_w,
                img_h=img_h,
                scale_x=sx,
                scale_y=sy,
                draw_points=True,
                draw_oval=True,
                draw_eyes=True,
                draw_lips=True,
            )

        if self.enable_pose_landmark and pose_pkt is not None:
            pose_landmarks = pose_pkt.meta.get("pose_landmarks") or []
            DrawUtils.draw_pose_landmarks(
                screen=camera_layer,
                pose_landmarks_data=pose_landmarks,
                screen_w=draw_rect.w,
                screen_h=draw_rect.h,
                draw_points=True,
                draw_connections=True,
            )

        self.screen.blit(camera_layer, draw_rect.topleft)

    def _draw_info_panel(self, panel_rect: pygame.Rect) -> None:
        panel = pygame.Surface((panel_rect.w, panel_rect.h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 185))
        self.screen.blit(panel, panel_rect.topleft)

        if self._last_raw_pkt is not None:
            self._disp_count += 1
            now = time.time()
            dt = now - self._disp_t0
            if dt >= 1.0:
                self._disp_fps = self._disp_count / dt
                self._disp_count = 0
                self._disp_t0 = now

        cam_meta = self._last_raw_pkt.meta if self._last_raw_pkt is not None else {}
        cam_fps = float(cam_meta.get("cam_fps", 0.0)) if cam_meta else 0.0
        req_w = int(cam_meta.get("request_width", 0)) if cam_meta else 0
        req_h = int(cam_meta.get("request_height", 0)) if cam_meta else 0
        act_w = int(cam_meta.get("actual_width", 0)) if cam_meta else 0
        act_h = int(cam_meta.get("actual_height", 0)) if cam_meta else 0
        face_fps = 0.0
        pose_fps = 0.0
        if self._bus is not None:
            face_pkt = self._bus.get_latest("frame/face")
            pose_pkt = self._bus.get_latest("frame/pose")
            if face_pkt is not None:
                face_fps = float(face_pkt.meta.get("infer_fps", 0.0))
            if pose_pkt is not None:
                pose_fps = float(pose_pkt.meta.get("infer_fps", 0.0))

        line_y = panel_rect.y + 10
        self.screen.blit(self.font_title.render(self.APP_TITLE, True, (255, 230, 90)), (panel_rect.x + 12, line_y))
        line_y += 28

        status_line = f"status={self._status_text} | disp={self._disp_fps:.1f}"
        self.screen.blit(self.font_body.render(status_line, True, (220, 220, 220)), (panel_rect.x + 12, line_y))
        line_y += 22
        fps_line = f"FPS raw={cam_fps:.1f} | face={face_fps:.1f} | pose={pose_fps:.1f}"
        self.screen.blit(self.font_body.render(fps_line, True, (220, 220, 220)), (panel_rect.x + 12, line_y))
        line_y += 22
        res_line = f"res req/actual={req_w}x{req_h} / {act_w}x{act_h}"
        self.screen.blit(self.font_small.render(res_line, True, (180, 180, 180)), (panel_rect.x + 12, line_y))
        line_y += 20
        self.screen.blit(self.font_small.render(self._runtime_info, True, (180, 180, 180)), (panel_rect.x + 12, line_y))
        line_y += 20

        toggles = (
            f"[1]POSE={'ON' if self.enable_pose_landmark else 'OFF'}  "
            f"[2]FACE={'ON' if self.enable_face_detection else 'OFF'}  "
            f"[3]LAND={'ON' if self.enable_face_landmark else 'OFF'}"
        )
        self.screen.blit(self.font_small.render(toggles, True, (130, 240, 130)), (panel_rect.x + 12, line_y))
        line_y += 20

        line_y = self._draw_rms_bar(panel_rect, line_y)

        warmup_line = (
            "warmup=done"
            if self._latest_warmup_done
            else f"warmup={self._latest_warmup_pct:.0f}% ({self._latest_warmup_buffer_s:.1f}/{self._latest_warmup_target_s:.1f}s)"
        )
        stt_meta_line = (
            f"lang={self._latest_lang} infer={self._latest_infer_ms:.0f}ms "
            f"buffer={self._latest_buffer_s:.1f}s q={self._latest_queue_chunks}"
        )
        self.screen.blit(self.font_small.render(warmup_line, True, (190, 190, 255)), (panel_rect.x + 12, line_y))
        line_y += 20
        self.screen.blit(self.font_small.render(stt_meta_line, True, (190, 190, 255)), (panel_rect.x + 12, line_y))
        line_y += 24

        text_w = panel_rect.w - 24
        max_lines = max(2, (panel_rect.bottom - line_y - 8) // 22)
        stt_lines = wrap_text(self._latest_asr_text, self.font_body, text_w, max_lines=max_lines)
        for t in stt_lines:
            self.screen.blit(self.font_body.render(t, True, (245, 245, 245)), (panel_rect.x + 12, line_y))
            line_y += 22

        if self.show_help:
            help_text = "[ENTER]fullscreen [H]help [ESC]quit"
            hint = self.font_small.render(help_text, True, (170, 170, 170))
            self.screen.blit(hint, (panel_rect.right - hint.get_width() - 12, panel_rect.y + 10))

    def _toggle_fullscreen(self) -> None:
        if self._is_fullscreen:
            self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
            self._is_fullscreen = False
            return

        self._windowed_size = self.screen.get_size()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self._is_fullscreen = True

    def _handle_events(self) -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self._toggle_fullscreen()
                    continue
                if e.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif e.key == pygame.K_1:
                    self.enable_pose_landmark = not self.enable_pose_landmark
                elif e.key == pygame.K_2:
                    self.enable_face_detection = not self.enable_face_detection
                elif e.key == pygame.K_3:
                    self.enable_face_landmark = not self.enable_face_landmark
                elif e.key == pygame.K_r:
                    self._status_text = "재시작 중..."
                    self._stop_pipeline()
                    self._start_pipeline()
            if e.type == pygame.VIDEORESIZE and not self._is_fullscreen:
                self._windowed_size = (max(800, e.w), max(600, e.h))
                self.screen = pygame.display.set_mode((max(800, e.w), max(600, e.h)), pygame.RESIZABLE)
        return True

    def run(self) -> None:
        try:
            self._start_pipeline()

            running = True
            while running:
                running = self._handle_events()
                self._update_from_bus()

                sw, sh = self.screen.get_size()
                camera_h = int(sh * self.camera_top_ratio)
                camera_h = max(220, min(camera_h, sh - 160))
                panel_h = sh - camera_h

                camera_rect = pygame.Rect(0, 0, sw, camera_h)
                panel_rect = pygame.Rect(0, camera_h, sw, panel_h)

                self.screen.fill((0, 0, 0))
                self._draw_vision(camera_rect)
                self._draw_info_panel(panel_rect)

                pygame.display.flip()
                self.clock.tick(60)

        finally:
            self._stop_pipeline()
            pygame.quit()


def main() -> None:
    app = DemoApplication()
    app.run()


if __name__ == "__main__":
    main()
