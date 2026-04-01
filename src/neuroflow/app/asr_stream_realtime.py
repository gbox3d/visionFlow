"""
file : neuroflow/app/asr_stream_realtime.py

NeuroFlow native streaming ASR app with microphone source (PySide6)

Pipeline:
  MicrophoneSource -> TopicBus("audio/raw") -> StreamingAsrWorker -> TopicBus("text/asr") -> UI

Rules:
- This file NEVER performs inference.
- This file NEVER blocks on processing.
- This file ONLY renders what exists on the TopicBus.
- Native streaming capable model only.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import sounddevice as sd
from dotenv import load_dotenv
from PySide6.QtCore import Signal, QObject
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
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGroupBox,
)

from common.runtime.bus import TopicBus
from voiceFlow.sources.microphone_source import MicrophoneSource
from asrFlow.processors.qwen_streaming_asr import QwenStreamingAsrProcessor
from asrFlow.utils.env import (
    env_float as _get_env_float,
    env_int as _get_env_int,
    env_lang as _get_env_optional_language,
    env_str as _get_env_str,
)
from asrFlow.workers.streaming_asr_worker import StreamingAsrWorker
from neuroflow.app.asr_model_catalog import (
    default_native_stream_backend_for_model,
    describe_native_stream_experiment,
    infer_native_stream_backends,
    native_stream_model_names,
)
from neuroflow.app.asr_ui_common import AudioLevelBridge, LevelGaugeWidget


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)

ENV_STREAM_MODEL = _get_env_str("ASRFLOW_STREAM_MODEL", "Qwen/Qwen3-ASR-0.6B")
ENV_STREAM_MODEL_PATH = _get_env_str("ASRFLOW_STREAM_MODEL_PATH", "")
ENV_STREAM_LANGUAGE = _get_env_optional_language("ASRFLOW_STREAM_LANGUAGE", "auto")
ENV_STREAM_CHUNK_SEC = _get_env_float("ASRFLOW_STREAM_CHUNK_SEC", 2.0)
ENV_STREAM_SAMPLERATE = _get_env_int("ASRFLOW_STREAM_SAMPLERATE", 16000)
ENV_STREAM_MAX_NEW_TOKENS = _get_env_int("ASRFLOW_STREAM_MAX_NEW_TOKENS", 512)
ENV_STREAM_UNFIXED_CHUNK_NUM = _get_env_int("ASRFLOW_STREAM_UNFIXED_CHUNK_NUM", 2)
ENV_STREAM_UNFIXED_TOKEN_NUM = _get_env_int("ASRFLOW_STREAM_UNFIXED_TOKEN_NUM", 5)


class NativeStreamResultBridge(QObject):
    asr_received = Signal(str, str, float, str, str, bool)

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
            if pkt is None or new_ver == last_ver:
                continue
            last_ver = new_ver

            infer_ms = float(pkt.meta.get("infer_ms", 0.0))
            full_text = str(pkt.meta.get("full_text", pkt.text))
            new_text = str(pkt.meta.get("new_text", pkt.text))
            finished = bool(pkt.meta.get("stream_finished", False))

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

            self.asr_received.emit(new_text, full_text, infer_ms, pkt.language, model_source, finished)


class AsrStreamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroFlow microphone -> ASR native stream realtime")
        self.setMinimumSize(640, 520)
        self.resize(760, 600)

        self._devices = self._list_input_devices()
        self._pipeline_running = False
        self._runtime_info = ""
        self._model_source_note = ""

        self._bus = TopicBus()
        self._mic_source: Optional[MicrophoneSource] = None
        self._processor: Optional[QwenStreamingAsrProcessor] = None
        self._asr_worker: Optional[StreamingAsrWorker] = None

        self._audio_bridge = AudioLevelBridge(self._bus, "audio/raw")
        self._asr_bridge = NativeStreamResultBridge(self._bus, "text/asr")
        self._audio_bridge.level_updated.connect(self._on_level)
        self._asr_bridge.asr_received.connect(self._on_asr_result)
        self._model_help_label: QLabel | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        settings_group = QGroupBox("Native Stream 설정")
        settings_layout = QVBoxLayout(settings_group)

        row_mic = QHBoxLayout()
        row_mic.addWidget(QLabel("마이크:"))
        self._combo_mic = QComboBox()
        for dev in self._devices:
            api_name = sd.query_hostapis(dev["hostapi"])["name"]
            self._combo_mic.addItem(f"{dev['index']}: {dev['name']}  ({api_name})")
        row_mic.addWidget(self._combo_mic, 1)
        settings_layout.addLayout(row_mic)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("백엔드:"))
        self._combo_backend = QComboBox()
        row_model.addWidget(self._combo_backend)

        row_model.addSpacing(14)
        row_model.addWidget(QLabel("모델:"))
        self._combo_model = QComboBox()
        self._combo_model.setEditable(True)
        for model_name in native_stream_model_names():
            self._combo_model.addItem(model_name)
        if self._combo_model.findText(ENV_STREAM_MODEL) < 0:
            self._combo_model.addItem(ENV_STREAM_MODEL)
        self._combo_model.setCurrentText(ENV_STREAM_MODEL)
        row_model.addWidget(self._combo_model)
        settings_layout.addLayout(row_model)

        row_opts = QHBoxLayout()
        row_opts.addWidget(QLabel("stream chunk(초):"))
        self._spin_chunk = QDoubleSpinBox()
        self._spin_chunk.setRange(0.5, 8.0)
        self._spin_chunk.setSingleStep(0.5)
        self._spin_chunk.setDecimals(1)
        self._spin_chunk.setValue(ENV_STREAM_CHUNK_SEC)
        row_opts.addWidget(self._spin_chunk)

        row_opts.addSpacing(12)
        row_opts.addWidget(QLabel("unfixed chunk:"))
        self._spin_unfixed_chunk = QSpinBox()
        self._spin_unfixed_chunk.setRange(0, 16)
        self._spin_unfixed_chunk.setValue(ENV_STREAM_UNFIXED_CHUNK_NUM)
        row_opts.addWidget(self._spin_unfixed_chunk)

        row_opts.addSpacing(12)
        row_opts.addWidget(QLabel("rollback token:"))
        self._spin_unfixed_token = QSpinBox()
        self._spin_unfixed_token.setRange(0, 64)
        self._spin_unfixed_token.setValue(ENV_STREAM_UNFIXED_TOKEN_NUM)
        row_opts.addWidget(self._spin_unfixed_token)
        row_opts.addStretch()
        settings_layout.addLayout(row_opts)

        row_model_path = QHBoxLayout()
        row_model_path.addWidget(QLabel("모델 경로:"))
        self._edit_model_path = QLineEdit()
        self._edit_model_path.setPlaceholderText("optional, 예: C:/models/Qwen3-ASR-0.6B")
        self._edit_model_path.setText(ENV_STREAM_MODEL_PATH)
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

        self._model_path_label = QLabel("모델 로드: -")
        self._model_path_label.setStyleSheet("color: #777; font-size: 11px;")
        self._model_path_label.setWordWrap(True)
        layout.addWidget(self._model_path_label)

        result_group = QGroupBox("누적 전사 결과")
        result_layout = QVBoxLayout(result_group)
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Consolas", 11))
        self._text_edit.setStyleSheet(
            "background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #444;"
        )
        result_layout.addWidget(self._text_edit)
        layout.addWidget(result_group, 1)

        hint = QLabel(
            "이 예제는 native streaming 가능 모델만 다룹니다. "
            "Qwen ASR transformers backend를 사용하며, chunk 예제와 달리 "
            "누적 오디오 + prefix rollback 전략으로 streaming state를 유지합니다."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(hint)

        self._combo_model.currentTextChanged.connect(self._sync_model_controls)
        self._edit_model_path.textChanged.connect(self._sync_model_controls)
        self._sync_model_controls(self._combo_model.currentText())

    @staticmethod
    def _list_input_devices() -> list[dict]:
        devices = sd.query_devices()
        result = []
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                item = dict(dev)
                item["index"] = idx
                result.append(item)
        return result

    def _current_model_name(self) -> str:
        return self._combo_model.currentText().strip()

    def _sync_model_controls(self, *_args) -> None:
        model_name = self._current_model_name()
        model_path = self._edit_model_path.text().strip() or None
        supported = infer_native_stream_backends(model_name, model_path=model_path)
        desired = default_native_stream_backend_for_model(model_name, model_path=model_path)

        self._combo_backend.blockSignals(True)
        self._combo_backend.clear()
        for backend in supported:
            self._combo_backend.addItem(backend)
        if desired is not None:
            self._combo_backend.setCurrentText(desired)
        self._combo_backend.blockSignals(False)

        if self._model_help_label is not None:
            self._model_help_label.setText(describe_native_stream_experiment(model_name, model_path=model_path))

    def _set_ui_locked(self, locked: bool) -> None:
        for widget in (
            self._combo_mic,
            self._combo_backend,
            self._combo_model,
            self._spin_chunk,
            self._spin_unfixed_chunk,
            self._spin_unfixed_token,
            self._edit_model_path,
        ):
            widget.setEnabled(not locked)
        self._btn_start.setEnabled(not locked)
        self._btn_stop.setEnabled(locked)

    def _on_start(self) -> None:
        if self._pipeline_running:
            return
        if not self._devices:
            self._status_label.setText("입력 장치가 없습니다")
            self._status_label.setStyleSheet("color: #f00;")
            return

        model_name = self._current_model_name()
        model_path = self._edit_model_path.text().strip() or None
        supported = infer_native_stream_backends(model_name, model_path=model_path)
        if not supported:
            self._status_label.setText("이 모델은 native streaming 예제에서 지원하지 않습니다")
            self._status_label.setStyleSheet("color: #f00;")
            return

        mic_idx = self._devices[self._combo_mic.currentIndex()]["index"]
        chunk_sec = self._spin_chunk.value()
        unfixed_chunk_num = self._spin_unfixed_chunk.value()
        unfixed_token_num = self._spin_unfixed_token.value()

        self._set_ui_locked(True)
        self._status_label.setText(f"native streaming 모델 로딩 중... ({model_name})")
        self._status_label.setStyleSheet("color: #fa0;")
        QApplication.processEvents()

        try:
            self._mic_source = MicrophoneSource(
                bus=self._bus,
                out_topic="audio/raw",
                samplerate=ENV_STREAM_SAMPLERATE,
                channels=1,
                blocksize=1024,
                device=mic_idx,
                source_id=f"mic_{mic_idx}",
            )
            self._processor = QwenStreamingAsrProcessor(
                model_name=model_name,
                model_path=model_path,
                language=ENV_STREAM_LANGUAGE,
                max_new_tokens=ENV_STREAM_MAX_NEW_TOKENS,
                chunk_size_sec=chunk_sec,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
            )
            self._processor.warmup()
            self._asr_worker = StreamingAsrWorker(
                bus=self._bus,
                processor=self._processor,
                in_topic="audio/raw",
                out_topic="text/asr",
                samplerate=ENV_STREAM_SAMPLERATE,
            )

            self._mic_source.start()
            self._asr_worker.start()
            self._audio_bridge.start()
            self._asr_bridge.start()
        except Exception as exc:
            self._stop_pipeline()
            self._set_ui_locked(False)
            self._status_label.setText(f"시작 실패: {exc}")
            self._status_label.setStyleSheet("color: #f00;")
            return

        self._pipeline_running = True
        self._runtime_info = (
            f"mic={mic_idx}  backend=qwen_transformers  model={model_name}  "
            f"stream_chunk={chunk_sec:.1f}s"
        )
        self._status_label.setText(f"수신 중 | {self._runtime_info}")
        self._status_label.setStyleSheet("color: #0a0;")

        if self._processor is not None:
            self._model_source_note = self._processor.get_model_source_label().strip()
            cache_root = self._processor.get_cache_root_abs().strip()
            if self._model_source_note and cache_root:
                self._model_path_label.setText(
                    f"모델 로드: {self._model_source_note}\n"
                    f"cache root(abs): {cache_root}"
                )
            elif self._model_source_note:
                self._model_path_label.setText(f"모델 로드: {self._model_source_note}")
            elif cache_root:
                self._model_path_label.setText(f"cache root(abs): {cache_root}")

    def _on_stop(self) -> None:
        self._stop_pipeline()
        self._set_ui_locked(False)
        self._status_label.setText("중지됨")
        self._status_label.setStyleSheet("color: #888;")

    def _on_level(self, level: float) -> None:
        self._gauge.set_level(level)

    def _on_asr_result(
        self,
        new_text: str,
        full_text: str,
        infer_ms: float,
        language: str,
        model_source: str,
        finished: bool,
    ) -> None:
        self._text_edit.setPlainText(full_text)
        sb = self._text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

        if model_source:
            self._model_source_note = model_source
        phase = "flush" if finished else "stream"
        delta_note = f" | delta={new_text}" if new_text else ""
        model_note = f" | {self._model_source_note}" if self._model_source_note else ""
        self._status_label.setText(
            f"수신 중 | {self._runtime_info} | {phase} [{language}] {infer_ms:.0f}ms{delta_note}{model_note}"
        )
        self._status_label.setStyleSheet("color: #0a0;")

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


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = AsrStreamWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
