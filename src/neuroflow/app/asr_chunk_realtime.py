"""
file : neuroflow/app/asr_chunk_realtime.py

NeuroFlow chunk realtime ASR app with microphone source (PySide6)

Pipeline:
  MicrophoneSource -> TopicBus("audio/raw") -> AsrWorker -> TopicBus("text/asr") -> UI

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
from pathlib import Path
from typing import Optional

import sounddevice as sd
from dotenv import load_dotenv

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGroupBox,
)

from common.runtime.bus import TopicBus
from voiceFlow.sources.microphone_source import MicrophoneSource
from asrFlow.processors.miso_stt_asr import MisoSttAsrProcessor
from asrFlow.utils.env import (
    env_bool as _get_env_bool,
    env_float as _get_env_float,
    env_int as _get_env_int,
    env_lang as _get_env_optional_language,
    env_str as _get_env_str,
)
from asrFlow.workers.asr_worker import AsrWorker
from neuroflow.app.asr_model_catalog import (
    ALL_BACKENDS,
    default_backend_for_model,
    describe_model_experiment,
    infer_supported_backends,
    preset_model_names,
)
from neuroflow.app.asr_ui_common import AudioLevelBridge, LevelGaugeWidget

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)
ENV_STT_BACKEND = _get_env_str("ASRFLOW_STT_BACKEND", "ct2")
ENV_STT_MODEL = _get_env_str("ASRFLOW_STT_MODEL", "large-v3")
ENV_STT_MODEL_PATH = _get_env_str("ASRFLOW_STT_MODEL_PATH", "")
ENV_STT_DEVICE = _get_env_str("ASRFLOW_STT_DEVICE", "auto")
ENV_STT_FP16 = _get_env_bool("ASRFLOW_STT_FP16", True)
ENV_STT_LANGUAGE = _get_env_optional_language("ASRFLOW_STT_LANGUAGE", "auto")
ENV_STT_TASK = _get_env_str("ASRFLOW_STT_TASK", "transcribe")
ENV_STT_BEAM_SIZE = _get_env_int("ASRFLOW_STT_BEAM_SIZE", 5)
ENV_STT_TEMPERATURE = _get_env_float("ASRFLOW_STT_TEMPERATURE", 0.0)
ENV_STT_CT2_VAD_FILTER = _get_env_bool("ASRFLOW_STT_CT2_VAD_FILTER", False)
ENV_STT_CHUNK_SEC = _get_env_float("ASRFLOW_STT_CHUNK_SEC", 3.0)
ENV_STT_SAMPLERATE = _get_env_int("ASRFLOW_STT_SAMPLERATE", 16000)
ENV_STT_MAX_NEW_TOKENS = _get_env_int("ASRFLOW_STT_MAX_NEW_TOKENS", 256)
ENV_STT_MAX_INFERENCE_BATCH_SIZE = _get_env_int("ASRFLOW_STT_MAX_INFERENCE_BATCH_SIZE", 8)


class AsrResultBridge(QObject):
    """TopicBus("text/asr") -> ASR 결과 Signal"""

    asr_received = Signal(str, str, float, str, str)

    def __init__(self, bus: TopicBus, topic: str = "text/asr"):
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
            pkt, new_ver = self._bus.wait_latest(self._topic, last_ver, timeout=0.2)
            if pkt is not None and new_ver != last_ver:
                last_ver = new_ver
                infer_ms = float(pkt.meta.get("infer_ms", 0.0))
                fallback_note = ""
                if pkt.meta.get("ct2_fallback_cpu"):
                    reason = str(pkt.meta.get("fallback_reason") or "").strip()
                    fallback_note = "CT2 GPU 실패 -> CPU fallback"
                    if reason:
                        fallback_note = f"{fallback_note} ({reason})"
                load_source = str(pkt.meta.get("load_source") or "").strip()
                resolved_local_path = str(
                    pkt.meta.get("resolved_local_path_abs")
                    or pkt.meta.get("resolved_local_path")
                    or ""
                ).strip()
                effective_model_path = str(
                    pkt.meta.get("effective_model_path_abs")
                    or pkt.meta.get("effective_model_path")
                    or ""
                ).strip()
                cache_root = str(pkt.meta.get("cache_root_abs") or "").strip()
                model_id = str(pkt.meta.get("model_id") or "").strip()
                if resolved_local_path:
                    model_source = f"{load_source}: {resolved_local_path}" if load_source else resolved_local_path
                elif effective_model_path:
                    model_source = f"{load_source}: {effective_model_path}" if load_source else effective_model_path
                elif cache_root and model_id:
                    model_source = f"{load_source}: {cache_root} ({model_id})" if load_source else f"{cache_root} ({model_id})"
                elif model_id:
                    model_source = f"{load_source}: {model_id}" if load_source else model_id
                else:
                    model_source = load_source
                self.asr_received.emit(pkt.text, pkt.language, infer_ms, fallback_note, model_source)


class AsrRealtimeWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroFlow microphone -> ASR chunk realtime")
        self.setMinimumSize(600, 480)
        self.resize(700, 560)

        self._devices = self._list_input_devices()
        self._pipeline_running = False

        self._bus = TopicBus()
        self._mic_source: Optional[MicrophoneSource] = None
        self._processor: Optional[MisoSttAsrProcessor] = None
        self._asr_worker: Optional[AsrWorker] = None

        self._audio_bridge = AudioLevelBridge(self._bus, "audio/raw")
        self._asr_bridge = AsrResultBridge(self._bus, "text/asr")
        self._audio_bridge.level_updated.connect(self._on_level)
        self._asr_bridge.asr_received.connect(self._on_asr_result)
        self._model_source_note = ""
        self._model_help_label: QLabel | None = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        settings_group = QGroupBox("설정")
        settings_layout = QVBoxLayout(settings_group)

        row_mic = QHBoxLayout()
        row_mic.addWidget(QLabel("마이크:"))
        self._combo_mic = QComboBox()
        for dev in self._devices:
            api_name = sd.query_hostapis(dev["hostapi"])["name"]
            self._combo_mic.addItem(f"{dev['index']}: {dev['name']}  ({api_name})")
        row_mic.addWidget(self._combo_mic, 1)
        settings_layout.addLayout(row_mic)

        row_opts = QHBoxLayout()
        row_opts.addWidget(QLabel("백엔드:"))
        self._combo_backend = QComboBox()
        for b in ALL_BACKENDS:
            self._combo_backend.addItem(b)
        row_opts.addWidget(self._combo_backend)

        row_opts.addSpacing(16)
        row_opts.addWidget(QLabel("모델:"))
        self._combo_model = QComboBox()
        self._combo_model.setEditable(True)
        for m in preset_model_names():
            self._combo_model.addItem(m)
        if self._combo_model.findText(ENV_STT_MODEL) < 0:
            self._combo_model.addItem(ENV_STT_MODEL)
        self._combo_model.setCurrentText(ENV_STT_MODEL)
        row_opts.addWidget(self._combo_model)

        row_opts.addSpacing(16)
        row_opts.addWidget(QLabel("청크 간격(초):"))
        self._spin_chunk = QDoubleSpinBox()
        self._spin_chunk.setRange(1.0, 30.0)
        self._spin_chunk.setSingleStep(0.5)
        self._spin_chunk.setValue(ENV_STT_CHUNK_SEC)
        self._spin_chunk.setDecimals(1)
        row_opts.addWidget(self._spin_chunk)
        row_opts.addStretch()
        settings_layout.addLayout(row_opts)

        row_model_path = QHBoxLayout()
        row_model_path.addWidget(QLabel("모델 경로:"))
        self._edit_model_path = QLineEdit()
        self._edit_model_path.setPlaceholderText("optional, 예: C:/models/my-merged-whisper or C:/models/Qwen3-ASR")
        self._edit_model_path.setText(ENV_STT_MODEL_PATH)
        row_model_path.addWidget(self._edit_model_path, 1)
        settings_layout.addLayout(row_model_path)

        self._model_help_label = QLabel("")
        self._model_help_label.setWordWrap(True)
        self._model_help_label.setStyleSheet("color: #666; font-size: 11px;")
        settings_layout.addWidget(self._model_help_label)

        layout.addWidget(settings_group)

        row_ctrl = QHBoxLayout()
        self._btn_start = QPushButton("시작")
        self._btn_start.setMinimumHeight(36)
        self._btn_start.clicked.connect(self._on_start)
        row_ctrl.addWidget(self._btn_start)

        self._btn_stop = QPushButton("중지")
        self._btn_stop.setMinimumHeight(36)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        row_ctrl.addWidget(self._btn_stop)
        layout.addLayout(row_ctrl)

        self._gauge = LevelGaugeWidget()
        layout.addWidget(self._gauge)

        self._status_label = QLabel("대기 중")
        self._status_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self._status_label)

        result_group = QGroupBox("인식 결과")
        result_layout = QVBoxLayout(result_group)
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Consolas", 11))
        self._text_edit.setStyleSheet(
            "background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #444;"
        )
        result_layout.addWidget(self._text_edit)
        layout.addWidget(result_group, 1)

        self._combo_model.currentTextChanged.connect(self._sync_model_controls)
        self._edit_model_path.textChanged.connect(self._sync_model_controls)
        self._sync_model_controls(self._combo_model.currentText())

    def _current_model_name(self) -> str:
        return self._combo_model.currentText().strip()

    def _sync_model_controls(self, *_args) -> None:
        model_name = self._current_model_name()
        model_path = self._edit_model_path.text().strip() or None
        supported = infer_supported_backends(model_name, model_path=model_path)
        desired = default_backend_for_model(
            model_name,
            preferred_backend=self._combo_backend.currentText().strip() or ENV_STT_BACKEND,
            model_path=model_path,
        )

        self._combo_backend.blockSignals(True)
        self._combo_backend.clear()
        for backend in supported:
            self._combo_backend.addItem(backend)
        self._combo_backend.setCurrentText(desired)
        self._combo_backend.blockSignals(False)

        if self._model_help_label is not None:
            self._model_help_label.setText(describe_model_experiment(model_name, model_path=model_path))

    @staticmethod
    def _build_backend_kwargs(backend: str) -> dict[str, object]:
        if backend == "ct2":
            return {
                "temperature": ENV_STT_TEMPERATURE,
                "vad_filter": ENV_STT_CT2_VAD_FILTER,
            }
        if backend in {"hf_generate", "hf_pipeline"}:
            return {"temperature": ENV_STT_TEMPERATURE}
        if backend == "qwen_transformers":
            return {
                "max_new_tokens": ENV_STT_MAX_NEW_TOKENS,
                "max_inference_batch_size": ENV_STT_MAX_INFERENCE_BATCH_SIZE,
            }
        return {}

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

    def _on_start(self) -> None:
        if self._pipeline_running:
            return
        if not self._devices:
            self._status_label.setText("입력 장치가 없습니다")
            self._status_label.setStyleSheet("color: #f00;")
            return

        mic_idx = self._devices[self._combo_mic.currentIndex()]["index"]
        backend = self._combo_backend.currentText().strip()
        model_size = self._current_model_name()
        if not model_size:
            self._status_label.setText("모델 이름이 비어 있습니다")
            self._status_label.setStyleSheet("color: #f00;")
            return
        model_path_raw = self._edit_model_path.text().strip()
        model_path = model_path_raw or None
        chunk_s = self._spin_chunk.value()
        backend_kwargs = self._build_backend_kwargs(backend)

        self._combo_backend.setEnabled(False)
        self._combo_mic.setEnabled(False)
        self._combo_model.setEnabled(False)
        self._edit_model_path.setEnabled(False)
        self._spin_chunk.setEnabled(False)
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._status_label.setText(f"모델 로딩 중... ({backend}/{model_size})")
        self._status_label.setStyleSheet("color: #fa0;")
        QApplication.processEvents()

        try:
            self._mic_source = MicrophoneSource(
                bus=self._bus,
                out_topic="audio/raw",
                samplerate=ENV_STT_SAMPLERATE,
                channels=1,
                blocksize=1024,
                device=mic_idx,
                source_id=f"mic_{mic_idx}",
            )

            self._processor = MisoSttAsrProcessor(
                backend=backend,
                model_name=model_size,
                model_path=model_path,
                device=ENV_STT_DEVICE,
                fp16=ENV_STT_FP16,
                language=ENV_STT_LANGUAGE,
                task=ENV_STT_TASK,
                beam_size=ENV_STT_BEAM_SIZE,
                backend_kwargs=backend_kwargs,
            )

            self._asr_worker = AsrWorker(
                bus=self._bus,
                processor=self._processor,
                in_topic="audio/raw",
                out_topic="text/asr",
                chunk_duration_s=chunk_s,
                samplerate=ENV_STT_SAMPLERATE,
            )

            self._mic_source.start()
            self._asr_worker.start()
            self._audio_bridge.start()
            self._asr_bridge.start()
        except Exception as e:
            self._stop_pipeline()
            self._combo_backend.setEnabled(True)
            self._combo_mic.setEnabled(True)
            self._combo_model.setEnabled(True)
            self._edit_model_path.setEnabled(True)
            self._spin_chunk.setEnabled(True)
            self._btn_start.setEnabled(True)
            self._btn_stop.setEnabled(False)

            msg = str(e).strip() or repr(e)
            if "out of memory" in msg.lower():
                msg = f"{msg} | 메모리 부족 가능성: 더 작은 모델 또는 fp16 비활성화를 고려하세요."
            self._status_label.setText(f"시작 실패: {msg}")
            self._status_label.setStyleSheet("color: #f00;")
            return

        self._pipeline_running = True
        self._status_label.setText(
            f"모니터링 중: mic={mic_idx}, backend={backend}, model={model_size}, chunk={chunk_s}s, device={ENV_STT_DEVICE}"
        )
        self._status_label.setStyleSheet("color: #0a0;")

    def _on_stop(self) -> None:
        self._stop_pipeline()

        self._combo_backend.setEnabled(True)
        self._combo_mic.setEnabled(True)
        self._combo_model.setEnabled(True)
        self._edit_model_path.setEnabled(True)
        self._spin_chunk.setEnabled(True)
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)

        self._status_label.setText("중지됨")
        self._status_label.setStyleSheet("color: #888;")

    def _on_level(self, level: float) -> None:
        self._gauge.set_level(level)

    def _on_asr_result(self, text: str, language: str, infer_ms: float, fallback_note: str, model_source: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] [{language}] ({infer_ms:.0f}ms) {text}"
        self._text_edit.append(line)
        if model_source:
            self._model_source_note = model_source
        if fallback_note:
            self._status_label.setText(f"모니터링 중 (주의): {fallback_note}")
            self._status_label.setStyleSheet("color: #fa0;")
        elif self._model_source_note:
            self._status_label.setText(f"모니터링 중 | model source: {self._model_source_note}")
            self._status_label.setStyleSheet("color: #0a0;")

        scrollbar = self._text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _stop_pipeline(self) -> None:
        self._asr_bridge.stop()
        self._audio_bridge.stop()
        if self._asr_worker is not None:
            self._asr_worker.stop()
            self._asr_worker = None
        if self._mic_source is not None:
            self._mic_source.stop()
            self._mic_source = None
        self._processor = None
        self._gauge.set_level(0.0)
        self._pipeline_running = False

    def closeEvent(self, event):
        self._stop_pipeline()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = AsrRealtimeWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
