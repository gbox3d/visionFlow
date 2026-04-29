"""
file : neuroflow/app/audiomi_asr_chunk_realtime.py

NeuroFlow audioMi ingress에서 오디오를 수신하여 chunk 실시간 ASR (PySide6)

Pipeline:
  AudioMiSource -> TopicBus("audio/raw") -> AccumulateAsrWorker -> TopicBus("text/asr") -> UI

누적(Accumulate) 전략:
  - step_s        : 추론 트리거 간격 (기본 1.5초)
  - max_window_s  : 버퍼 최대 길이 (기본 25초, Whisper 30초 제한 대비)
  - 이전 결과와 새 결과를 비교해 새로 추가된 suffix만 화면에 표시

Rules:
- This file NEVER performs inference.
- This file NEVER blocks on processing.
- This file ONLY renders what exists on the TopicBus.
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from typing import Optional

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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGroupBox,
)

from common.runtime.bus import TopicBus
from voiceFlow.sources.audiomi_source import AudioMiSource, CMD_LOOPBACK, CMD_MIC
from asrFlow.processors.miso_stt_asr import MisoSttAsrProcessor
from asrFlow.utils.env import (
    env_bool as _env_bool,
    env_float as _env_float,
    env_int as _env_int,
    env_lang as _env_lang,
    env_str as _env_str,
)
from asrFlow.workers.accumulate_asr_worker import AccumulateAsrWorker
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
ENV_HOST = _env_str("AUDIOMI_HOST", "127.0.0.1")
ENV_PORT = _env_int("AUDIOMI_PORT", 26070)
ENV_CHECKCODE = _env_int("AUDIOMI_CHECKCODE", 20250918)

ENV_BACKEND = _env_str("ASRFLOW_STT_BACKEND", "ct2")
ENV_MODEL = _env_str("ASRFLOW_STT_MODEL", "large-v3")
ENV_MODEL_PATH = _env_str("ASRFLOW_STT_MODEL_PATH", "")
ENV_DEVICE = _env_str("ASRFLOW_STT_DEVICE", "auto")
ENV_FP16 = _env_bool("ASRFLOW_STT_FP16", True)
ENV_LANGUAGE = _env_lang("ASRFLOW_STT_LANGUAGE", "auto")
ENV_TASK = _env_str("ASRFLOW_STT_TASK", "transcribe")
ENV_BEAM_SIZE = _env_int("ASRFLOW_STT_BEAM_SIZE", 5)
ENV_TEMPERATURE = _env_float("ASRFLOW_STT_TEMPERATURE", 0.0)
ENV_CT2_VAD_FILTER = _env_bool("ASRFLOW_STT_CT2_VAD_FILTER", False)
ENV_SAMPLERATE = _env_int("ASRFLOW_STT_SAMPLERATE", 16000)
ENV_ENABLE_LOGGING = _env_bool("ENABLE_LOGGING", False)
ENV_STT_MAX_NEW_TOKENS = _env_int("ASRFLOW_STT_MAX_NEW_TOKENS", 256)
ENV_STT_MAX_INFERENCE_BATCH_SIZE = _env_int("ASRFLOW_STT_MAX_INFERENCE_BATCH_SIZE", 8)

CMD_OPTIONS = [("loopback (0x01)", CMD_LOOPBACK), ("mic (0x02)", CMD_MIC), ("모두", None)]


class AsrResultBridge(QObject):
    asr_received = Signal(str, str, str, float, float, int, str, str)

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
                buffer_s = float(pkt.meta.get("buffer_s", 0.0))
                queue_chunks = int(pkt.meta.get("queue_chunks", 0))
                new_text = str(pkt.meta.get("new_text", pkt.text))
                full_text = str(pkt.meta.get("full_text", pkt.text))
                fallback_note = ""
                if pkt.meta.get("ct2_fallback_cpu"):
                    reason = str(pkt.meta.get("fallback_reason") or "")
                    fallback_note = f"CT2 GPU->CPU fallback ({reason})" if reason else "CT2 GPU->CPU fallback"
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
                self.asr_received.emit(
                    new_text,
                    full_text,
                    pkt.language,
                    infer_ms,
                    buffer_s,
                    queue_chunks,
                    fallback_note,
                    model_source,
                )


class WarmupStatusBridge(QObject):
    warmup_updated = Signal(float, float, float, bool, int)

    def __init__(self, bus: TopicBus, topic: str = "text/asr_status"):
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

            data = pkt if isinstance(pkt, dict) else {}
            warmup_pct = float(data.get("warmup_pct", 0.0))
            buffer_s = float(data.get("buffer_s", 0.0))
            target_s = float(data.get("target_s", 0.0))
            warmed = bool(data.get("warmed", False))
            queue_chunks = int(data.get("queue_chunks", 0))
            self.warmup_updated.emit(warmup_pct, buffer_s, target_s, warmed, queue_chunks)


class AudioMiAsrWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroFlow audioMi -> ASR chunk realtime [누적 모드]")
        self.setMinimumSize(700, 600)
        self.resize(820, 700)

        self._running = False
        self._bus = TopicBus()
        self._source: Optional[AudioMiSource] = None
        self._processor: Optional[MisoSttAsrProcessor] = None
        self._worker: Optional[AccumulateAsrWorker] = None

        self._audio_bridge = AudioLevelBridge(self._bus, "audio/raw")
        self._asr_bridge = AsrResultBridge(self._bus, "text/asr")
        self._warmup_bridge = WarmupStatusBridge(self._bus, "text/asr_status")
        self._audio_bridge.level_updated.connect(self._on_level)
        self._asr_bridge.asr_received.connect(self._on_asr_result)
        self._warmup_bridge.warmup_updated.connect(self._on_warmup_status)

        self._model_source_note: str = ""
        self._cache_root_note: str = ""
        self._runtime_info: str = ""
        self._model_help_label: QLabel | None = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        conn_group = QGroupBox("audioMi 서버 연결")
        conn_layout = QHBoxLayout(conn_group)

        conn_layout.addWidget(QLabel("Host:"))
        self._edit_host = QLineEdit(ENV_HOST)
        self._edit_host.setMaximumWidth(140)
        conn_layout.addWidget(self._edit_host)

        conn_layout.addSpacing(8)
        conn_layout.addWidget(QLabel("Port:"))
        self._spin_port = QSpinBox()
        self._spin_port.setRange(1, 65535)
        self._spin_port.setValue(ENV_PORT)
        self._spin_port.setMaximumWidth(80)
        conn_layout.addWidget(self._spin_port)

        conn_layout.addSpacing(8)
        conn_layout.addWidget(QLabel("Checkcode:"))
        self._edit_checkcode = QLineEdit(str(ENV_CHECKCODE))
        self._edit_checkcode.setMaximumWidth(100)
        conn_layout.addWidget(self._edit_checkcode)

        conn_layout.addSpacing(8)
        conn_layout.addWidget(QLabel("채널:"))
        self._combo_cmd = QComboBox()
        for label, _ in CMD_OPTIONS:
            self._combo_cmd.addItem(label)
        conn_layout.addWidget(self._combo_cmd)

        conn_layout.addStretch()
        root.addWidget(conn_group)

        asr_group = QGroupBox("ASR 설정")
        asr_layout = QVBoxLayout(asr_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("백엔드:"))
        self._combo_backend = QComboBox()
        for b in ALL_BACKENDS:
            self._combo_backend.addItem(b)
        row1.addWidget(self._combo_backend)

        row1.addSpacing(12)
        row1.addWidget(QLabel("모델:"))
        self._combo_model = QComboBox()
        self._combo_model.setEditable(True)
        for m in preset_model_names():
            self._combo_model.addItem(m)
        if self._combo_model.findText(ENV_MODEL) < 0:
            self._combo_model.addItem(ENV_MODEL)
        self._combo_model.setCurrentText(ENV_MODEL)
        row1.addWidget(self._combo_model)

        row1.addSpacing(12)
        row1.addWidget(QLabel("추론 간격(초):"))
        self._spin_step = QDoubleSpinBox()
        self._spin_step.setRange(0.5, 10.0)
        self._spin_step.setSingleStep(0.5)
        self._spin_step.setDecimals(1)
        self._spin_step.setValue(1.5)
        self._spin_step.setMaximumWidth(70)
        self._spin_step.setToolTip("step_s: 이 간격마다 누적 버퍼 전체를 추론합니다.")
        row1.addWidget(self._spin_step)

        row1.addSpacing(6)
        row1.addWidget(QLabel("최대 버퍼(초):"))
        self._spin_max_window = QDoubleSpinBox()
        self._spin_max_window.setRange(5.0, 30.0)
        self._spin_max_window.setSingleStep(1.0)
        self._spin_max_window.setDecimals(0)
        self._spin_max_window.setValue(25.0)
        self._spin_max_window.setMaximumWidth(70)
        self._spin_max_window.setToolTip("Whisper 30초 제한 대비 최대 누적 길이")
        row1.addWidget(self._spin_max_window)

        row1.addStretch()
        asr_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("모델 경로 (선택):"))
        self._edit_model_path = QLineEdit(ENV_MODEL_PATH)
        self._edit_model_path.setPlaceholderText("비워두면 자동, 또는 local model path")
        row2.addWidget(self._edit_model_path, 1)
        asr_layout.addLayout(row2)

        self._model_help_label = QLabel("")
        self._model_help_label.setWordWrap(True)
        self._model_help_label.setStyleSheet("color: #666; font-size: 11px;")
        asr_layout.addWidget(self._model_help_label)

        root.addWidget(asr_group)

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
        root.addLayout(row_ctrl)

        self._gauge = LevelGaugeWidget()
        root.addWidget(self._gauge)

        self._status_label = QLabel("대기 중")
        self._status_label.setStyleSheet("color: #888; font-size: 12px;")
        root.addWidget(self._status_label)
        self._model_path_label = QLabel("모델 로드: -")
        self._model_path_label.setStyleSheet("color: #777; font-size: 11px;")
        self._model_path_label.setWordWrap(True)
        root.addWidget(self._model_path_label)
        self._warmup_bar = QProgressBar()
        self._warmup_bar.setRange(0, 100)
        self._warmup_bar.setValue(0)
        self._warmup_bar.setFormat("워밍업 0%")
        self._warmup_bar.setTextVisible(True)
        self._warmup_bar.setMinimumHeight(18)
        self._set_warmup_bar_color(0)
        root.addWidget(self._warmup_bar)

        result_group = QGroupBox("인식 결과")
        result_layout = QVBoxLayout(result_group)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Consolas", 11))
        self._text_edit.setStyleSheet(
            "background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #444;"
        )
        result_layout.addWidget(self._text_edit)
        root.addWidget(result_group, 1)

        hint = QLabel(
            "누적 모드: 말하는 동안 버퍼를 계속 쌓아 전체 문맥을 유지합니다. "
            "최대 버퍼 길이에 맞춰 누적 길이를 관리합니다."
        )
        hint.setStyleSheet("color: #666; font-size: 10px;")
        hint.setWordWrap(True)
        root.addWidget(hint)

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
            preferred_backend=self._combo_backend.currentText().strip() or ENV_BACKEND,
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
                "temperature": ENV_TEMPERATURE,
                "vad_filter": ENV_CT2_VAD_FILTER,
            }
        if backend in {"hf_generate", "hf_pipeline"}:
            return {"temperature": ENV_TEMPERATURE}
        if backend == "qwen_transformers":
            return {
                "max_new_tokens": ENV_STT_MAX_NEW_TOKENS,
                "max_inference_batch_size": ENV_STT_MAX_INFERENCE_BATCH_SIZE,
            }
        return {}

    def _on_start(self) -> None:
        if self._running:
            return

        host = self._edit_host.text().strip() or "127.0.0.1"
        port = self._spin_port.value()
        try:
            checkcode = int(self._edit_checkcode.text().strip())
        except ValueError:
            self._status_label.setText("Checkcode가 정수가 아닙니다")
            self._status_label.setStyleSheet("color: #f00;")
            return

        cmd_filter = CMD_OPTIONS[self._combo_cmd.currentIndex()][1]
        backend = self._combo_backend.currentText().strip()
        model_size = self._current_model_name()
        if not model_size:
            self._status_label.setText("모델 이름이 비어 있습니다")
            self._status_label.setStyleSheet("color: #f00;")
            return
        model_path = self._edit_model_path.text().strip() or None
        step_s = self._spin_step.value()
        max_window_s = self._spin_max_window.value()
        backend_kwargs = self._build_backend_kwargs(backend)

        self._set_ui_locked(True)
        self._status_label.setText(f"모델 로딩 중... ({backend}/{model_size})")
        self._status_label.setStyleSheet("color: #fa0;")
        self._warmup_bar.setValue(0)
        self._warmup_bar.setFormat("워밍업 0%")
        self._set_warmup_bar_color(0)
        QApplication.processEvents()

        try:
            self._source = AudioMiSource(
                bus=self._bus,
                out_topic="audio/raw",
                host=host,
                port=port,
                checkcode=checkcode,
                cmd_filter=cmd_filter,
            )
            self._processor = MisoSttAsrProcessor(
                backend=backend,
                model_name=model_size,
                model_path=model_path,
                device=ENV_DEVICE,
                fp16=ENV_FP16,
                language=ENV_LANGUAGE,
                task=ENV_TASK,
                beam_size=ENV_BEAM_SIZE,
                backend_kwargs=backend_kwargs,
            )
            self._status_label.setText(f"모델 초기화 중... ({backend}/{model_size})")
            self._status_label.setStyleSheet("color: #fa0;")
            QApplication.processEvents()
            self._processor.warmup()

            self._worker = AccumulateAsrWorker(
                bus=self._bus,
                processor=self._processor,
                in_topic="audio/raw",
                out_topic="text/asr",
                step_s=step_s,
                max_window_s=max_window_s,
                samplerate=ENV_SAMPLERATE,
                enable_logging=ENV_ENABLE_LOGGING,
                log_dir=str(PROJECT_ROOT / "logs"),
            )

            self._warmup_bridge.start()
            self._worker.start()
            self._source.start()
            self._audio_bridge.start()
            self._asr_bridge.start()

        except Exception as e:
            self._teardown()
            self._set_ui_locked(False)
            msg = str(e).strip() or repr(e)
            self._status_label.setText(f"시작 실패: {msg}")
            self._status_label.setStyleSheet("color: #f00;")
            return

        self._running = True
        self._runtime_info = (
            f"{host}:{port}  backend={backend}  model={model_size}  "
            f"step={step_s}s  max={max_window_s:.0f}s  temp={ENV_TEMPERATURE}"
        )
        self._status_label.setText(f"수신 중 | {self._runtime_info}")
        self._status_label.setStyleSheet("color: #0a0;")

        if self._processor is not None:
            self._model_source_note = self._processor.get_model_source_label().strip()
            self._cache_root_note = self._processor.get_cache_root_abs().strip()
            if self._model_source_note and self._cache_root_note:
                self._model_path_label.setText(
                    f"모델 로드: {self._model_source_note}\n"
                    f"cache root(abs): {self._cache_root_note}"
                )
            elif self._model_source_note:
                self._model_path_label.setText(f"모델 로드: {self._model_source_note}")
            elif self._cache_root_note:
                self._model_path_label.setText(f"cache root(abs): {self._cache_root_note}")

    def _on_stop(self) -> None:
        self._teardown()
        self._set_ui_locked(False)
        self._status_label.setText("중지됨")
        self._status_label.setStyleSheet("color: #888;")

    def _on_level(self, level: float) -> None:
        self._gauge.set_level(level)

    def _on_warmup_status(
        self,
        warmup_pct: float,
        buffer_s: float,
        target_s: float,
        warmed: bool,
        queue_chunks: int,
    ) -> None:
        pct_i = int(max(0, min(100, round(warmup_pct))))
        self._warmup_bar.setValue(pct_i)
        self._set_warmup_bar_color(pct_i)
        if warmed:
            self._warmup_bar.setFormat("워밍업 완료 100%")
        else:
            self._warmup_bar.setFormat(
                f"워밍업 {pct_i}% ({buffer_s:.1f}/{target_s:.1f}s, q={queue_chunks})"
            )

    def _on_asr_result(
        self,
        new_text: str,
        full_text: str,
        language: str,
        infer_ms: float,
        buffer_s: float,
        queue_chunks: int,
        fallback_note: str,
        model_source: str,
    ) -> None:
        self._text_edit.setPlainText(full_text)

        if fallback_note:
            self._status_label.setText(f"(주의) {fallback_note}")
            self._status_label.setStyleSheet("color: #fa0;")
        else:
            self._status_label.setText(
                f"수신 중 | {self._runtime_info} | queue={queue_chunks} chunks ({buffer_s:.1f}s) | infer={infer_ms:.0f}ms"
            )
            self._status_label.setStyleSheet("color: #0a0;")

        sb = self._text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _teardown(self) -> None:
        self._warmup_bridge.stop()
        self._asr_bridge.stop()
        self._audio_bridge.stop()
        if self._worker:
            self._worker.stop()
            self._worker = None
        if self._source:
            self._source.stop()
            self._source = None
        self._processor = None
        self._gauge.set_level(0.0)
        self._model_source_note = ""
        self._cache_root_note = ""
        self._runtime_info = ""
        self._model_path_label.setText("모델 로드: -")
        self._warmup_bar.setValue(0)
        self._warmup_bar.setFormat("워밍업 0%")
        self._set_warmup_bar_color(0)
        self._text_edit.clear()
        self._running = False

    def _set_warmup_bar_color(self, pct_i: int) -> None:
        pct = max(0, min(100, int(pct_i)))
        t = pct / 100.0
        r = int(round(220 + (60 - 220) * t))
        g = int(round(60 + (180 - 60) * t))
        b = 60
        self._warmup_bar.setStyleSheet(
            "QProgressBar {"
            " border: 1px solid #444;"
            " background-color: #1e1e1e;"
            " color: #e8e8e8;"
            " text-align: center;"
            "}"
            "QProgressBar::chunk {"
            f" background-color: rgb({r}, {g}, {b});"
            "}"
        )

    def _set_ui_locked(self, locked: bool) -> None:
        for w in (
            self._edit_host, self._spin_port, self._edit_checkcode,
            self._combo_cmd, self._combo_backend, self._combo_model,
            self._spin_step, self._spin_max_window,
            self._edit_model_path,
        ):
            w.setEnabled(not locked)
        self._btn_start.setEnabled(not locked)
        self._btn_stop.setEnabled(locked)

    def closeEvent(self, event):
        self._teardown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AudioMiAsrWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
