from __future__ import annotations

"""
voiceFlow/workers/accumulate_asr_worker.py

누적(Accumulate) 기반 실시간 ASR 워커.

슬라이딩 윈도우의 문제점
-----------------------
  Whisper는 30초짜리 mel spectrogram positional encoding으로 학습됨.
  짧은 구간(3초)을 반복 추론하면 앞뒤 맥락이 없어 hallucination이 심해짐.

누적 방식 동작 원리
-------------------
  버퍼를 계속 누적하면서 step_s마다 전체 버퍼를 추론하고,
  해당 시점의 전체 텍스트를 그대로 publish한다.

  max_window_s 도달 시 오래된 앞부분을 잘라내고 최근 버퍼를 유지.

"""

from collections import deque
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
import threading
import time
from typing import Optional
import wave

import numpy as np

from visionflow.pipeline.bus import TopicBus
from voiceFlow.pipeline.packet import AsrResultPacket
from voiceFlow.workers.asr_worker import AsrProcessorProtocol


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class AccumulateAsrWorker:
    """
    누적 오디오 버퍼로 전체 문맥을 유지하며 추론하는 ASR 워커.

    Parameters
    ----------
    bus             : TopicBus
    processor       : AsrProcessorProtocol
    in_topic        : 오디오 입력 토픽
    out_topic       : ASR 결과 출력 토픽
    step_s          : 추론 트리거 간격 (초). 기본 1.5
                      → 이 간격마다 누적 버퍼 전체를 추론
    max_window_s    : 버퍼 최대 길이 (초). 기본 25.0
                      → 초과 시 앞부분을 잘라 최근 구간 유지 (Whisper 30초 제한 대비)
    samplerate      : 기본 16000
    name            : 워커 이름 (로그용)
    """

    def __init__(
        self,
        bus: TopicBus,
        processor: AsrProcessorProtocol,
        in_topic: str = "audio/raw",
        out_topic: str = "text/asr",
        status_topic: str = "text/asr_status",
        step_s: float = 1.5,
        max_window_s: float = 25.0,
        samplerate: int = 16000,
        enable_logging: bool = False,
        log_dir: str = "logs",
        name: str = "accumulate-asr-worker",
    ):
        self.bus = bus
        self.processor = processor
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.status_topic = status_topic
        self.step_s = float(step_s)
        self.max_window_s = float(max_window_s)
        self.samplerate = int(samplerate)
        self.enable_logging = bool(enable_logging)
        self.log_dir = Path(log_dir)
        self.name = name
        # 청크 트림 구조에서는 max_window - step 근처로 안정화되므로 이를 워밍업 기준으로 사용
        self._warmup_target_s = max(self.step_s, self.max_window_s - self.step_s)

        self._step_samples     = int(self.samplerate * self.step_s)
        self._max_window_samp  = int(self.samplerate * self.max_window_s)

        # 고정 청크 큐 + 잔여 tail
        self._chunk_samples = self._step_samples
        self._buffer: deque[np.ndarray] = deque()
        self._tail = np.empty(0, dtype=np.float32)
        self._buffer_samples: int = 0

        # step 카운터: 마지막 추론 이후 새로 쌓인 샘플 수
        self._new_samples_since_infer: int = 0

        # 버퍼 동기화 (ingest / infer 분리)
        self._buf_lock = threading.RLock()
        self._buf_cond = threading.Condition(self._buf_lock)
        self._last_source_id = ""

        self._running = False
        self._ingest_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None
        self._last_ver = 0
        self._seq = 0

        # 성능
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._infer_fps = 0.0
        self._buffer_warmed = False
        self._warmup_logged = False
        self._prev_accepted_text = ""
        self._last_warmup_pct: float = -1.0
        self._last_warmup_emit_ts: float = 0.0

        if self.enable_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_ver = self.bus.get_version(self.in_topic)
        self._ingest_thread = threading.Thread(target=self._ingest_loop, daemon=True)
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._ingest_thread.start()
        self._infer_thread.start()
        self._publish_warmup_status(buffer_s=0.0, queue_chunks=0, force=True)
        print(
            f"[AccumulateAsrWorker] 시작: step={self.step_s}s  "
            f"max_window={self.max_window_s}s  sr={self.samplerate}Hz"
        )

    def stop(self) -> None:
        self._running = False
        with self._buf_cond:
            self._buf_cond.notify_all()
        if self._ingest_thread:
            self._ingest_thread.join(timeout=6.0)
            self._ingest_thread = None
        if self._infer_thread:
            self._infer_thread.join(timeout=6.0)
            self._infer_thread = None
        try:
            self.processor.close()
        except Exception:
            pass
        print("[AccumulateAsrWorker] 종료")

    # ------------------------------------------------------------------ Internal

    def _reset_buffer(self) -> None:
        self._buffer = deque()
        self._tail = np.empty(0, dtype=np.float32)
        self._buffer_samples = 0
        self._new_samples_since_infer = 0

    def _append_audio_chunked(self, audio: np.ndarray) -> None:
        """입력 오디오를 고정 청크(step_s)로 큐에 적재하고 나머지는 tail로 보관."""
        if len(audio) == 0:
            return
        if len(self._tail) > 0:
            combined = np.concatenate((self._tail, audio), axis=0)
        else:
            combined = audio

        while len(combined) >= self._chunk_samples:
            self._buffer.append(combined[: self._chunk_samples].copy())
            combined = combined[self._chunk_samples :]

        self._tail = combined
        self._buffer_samples += len(audio)

    def _publish_warmup_status(self, buffer_s: float, queue_chunks: int, force: bool = False) -> None:
        target_s = self._warmup_target_s
        if target_s <= 0:
            pct = 100.0
        else:
            pct = max(0.0, min(100.0, (buffer_s / target_s) * 100.0))

        now = time.time()
        if not force:
            if int(pct) == int(self._last_warmup_pct) and (now - self._last_warmup_emit_ts) < 0.2:
                return

        self.bus.publish(
            self.status_topic,
            {
                "warmup_pct": pct,
                "buffer_s": buffer_s,
                "target_s": target_s,
                "warmed": self._buffer_warmed,
                "queue_chunks": queue_chunks,
            },
        )
        self._last_warmup_pct = pct
        self._last_warmup_emit_ts = now

    def _write_log_pair(self, audio: np.ndarray, text: str) -> None:
        if not self.enable_logging:
            return
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        wav_path = self.log_dir / f"{stem}.wav"
        txt_path = self.log_dir / f"{stem}.txt"

        audio_f32 = audio.astype(np.float32, copy=False)
        audio_i16 = np.clip(audio_f32, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767.0).astype(np.int16)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_i16.tobytes())

        with txt_path.open("w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")

    @staticmethod
    def _middle_words(text: str, edge_words: int = 3) -> str:
        words = [w for w in text.split() if w]
        if len(words) <= edge_words * 2 + 2:
            return ""
        return " ".join(words[edge_words:-edge_words]).strip()

    def _is_consistency_failure(self, prev_text: str, curr_text: str) -> tuple[bool, float]:
        prev_mid = self._middle_words(prev_text)
        curr_mid = self._middle_words(curr_text)
        if not prev_mid or not curr_mid:
            return False, 1.0
        score = SequenceMatcher(None, prev_mid, curr_mid).ratio()
        return score < 0.35, score

    def _log_consistency_failure(self, audio: np.ndarray, prev_text: str, curr_text: str, score: float) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        msg = f"[AccumulateAsrWorker] consistency FAIL score={score:.3f} | prev='{prev_text[:120]}' | curr='{curr_text[:120]}'"
        print(msg)
        if not self.enable_logging:
            return

        # 실패 이력 텍스트 로그
        fail_log = self.log_dir / "filter_failures.log"
        with fail_log.open("a", encoding="utf-8") as f:
            f.write(f"{ts} score={score:.3f}\n")
            f.write(f"PREV: {prev_text}\n")
            f.write(f"CURR: {curr_text}\n\n")

        # 실패 샘플도 쌍으로 저장
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + "_FAIL"
        wav_path = self.log_dir / f"{stem}.wav"
        txt_path = self.log_dir / f"{stem}.txt"
        audio_f32 = audio.astype(np.float32, copy=False)
        audio_i16 = np.clip(audio_f32, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_i16.tobytes())
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(curr_text.strip() + "\n")

    def _trim_buffer_to_max(self) -> None:
        """max_window 초과 시 앞쪽 청크부터 제거한다."""
        if self._buffer_samples <= self._max_window_samp:
            return
        if self._max_window_samp <= 0:
            self._reset_buffer()
            return

        trimmed = 0
        while self._buffer_samples > self._max_window_samp and self._buffer:
            old = self._buffer.popleft()
            drop = len(old)
            self._buffer_samples -= drop
            trimmed += drop

        # 청크가 없는데도 초과한 경우(초기 구간) tail 앞부분을 잘라 정렬
        if self._buffer_samples > self._max_window_samp and len(self._tail) > 0:
            drop = min(self._buffer_samples - self._max_window_samp, len(self._tail))
            self._tail = self._tail[drop:]
            self._buffer_samples -= drop
            trimmed += drop

        if trimmed > 0:
            print(
                f"[AccumulateAsrWorker] max_window 초과 → 앞부분 청크 트림 "
                f"({trimmed / self.samplerate:.2f}s 제거, {self._buffer_samples / self.samplerate:.2f}s 유지)"
            )

    def _ingest_loop(self) -> None:
        while self._running:
            pkt, v = self.bus.wait_latest(self.in_topic, self._last_ver, timeout=0.2)
            if pkt is None:
                continue
            self._last_ver = v
            source_id = pkt.source_id

            audio: np.ndarray = pkt.audio
            if audio.ndim > 1:
                audio = audio[:, 0]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # 추론과 분리된 ingest 경로: 추론 중에도 누적 지속
            with self._buf_cond:
                self._append_audio_chunked(audio)
                self._new_samples_since_infer += len(audio)
                self._last_source_id = source_id

                # max_window 초과 → 앞부분만 잘라 최근 구간 유지
                if self._buffer_samples > self._max_window_samp:
                    self._trim_buffer_to_max()

                buffer_s = self._buffer_samples / self.samplerate
                queue_chunks = len(self._buffer)
                if self._new_samples_since_infer >= self._step_samples:
                    self._buf_cond.notify()
            self._publish_warmup_status(buffer_s=buffer_s, queue_chunks=queue_chunks)

    def _infer_loop(self) -> None:
        while self._running:
            with self._buf_cond:
                if self._new_samples_since_infer < self._step_samples:
                    self._buf_cond.wait(timeout=0.2)
                if not self._running:
                    break
                if self._new_samples_since_infer < self._step_samples:
                    continue

                self._new_samples_since_infer = 0
                parts = list(self._buffer)
                if len(self._tail) > 0:
                    parts.append(self._tail)
                if not parts:
                    continue
                full_audio = np.concatenate(parts, axis=0)
                source_id = self._last_source_id
                buffer_s = self._buffer_samples / self.samplerate
                queue_chunks = len(self._buffer)
                queue_tail_s = len(self._tail) / self.samplerate

            self._seq += 1
            t0 = time.time()
            try:
                result = self.processor.process(
                    audio_chunk=full_audio,
                    samplerate=self.samplerate,
                    seq=self._seq,
                    source_id=source_id,
                )
            except Exception as e:
                print(f"[AccumulateAsrWorker] 추론 실패(seq={self._seq}): {e}")
                continue
            infer_ms = (time.time() - t0) * 1000.0

            # infer fps
            self._fps_n += 1
            now = time.time()
            d = now - self._fps_t0
            if d >= 1.0:
                self._infer_fps = self._fps_n / d
                self._fps_n = 0
                self._fps_t0 = now

            if result is None:
                continue

            curr_text = result.text.strip()
            if not curr_text:
                continue

            # 버퍼가 max_window까지 채워지기 전에는 안정화 단계로 보고 결과를 내보내지 않는다.
            if not self._buffer_warmed:
                if buffer_s < self._warmup_target_s:
                    if not self._warmup_logged:
                        print(
                            f"[AccumulateAsrWorker] 버퍼 워밍업 중... "
                            f"({buffer_s:.1f}/{self._warmup_target_s:.1f}s)"
                        )
                        self._warmup_logged = True
                    continue
                self._buffer_warmed = True
                self._prev_accepted_text = curr_text
                print("[AccumulateAsrWorker] 버퍼 워밍업 완료 -> 필터 활성화")
                self._publish_warmup_status(buffer_s=buffer_s, queue_chunks=queue_chunks, force=True)

            # 이전 결과와 중간부 일관성 비교. 크게 다르면 실패 처리한다.
            if self._prev_accepted_text:
                failed, score = self._is_consistency_failure(self._prev_accepted_text, curr_text)
                if failed:
                    self._log_consistency_failure(full_audio, self._prev_accepted_text, curr_text, score)
                    continue

            self._prev_accepted_text = curr_text
            self._write_log_pair(full_audio, curr_text)

            enriched = AsrResultPacket(
                text=curr_text,
                segments=result.segments,
                language=result.language,
                ts_ms=result.ts_ms,
                seq=result.seq,
                source_id=result.source_id,
                meta={
                    **result.meta,
                    "infer_fps": self._infer_fps,
                    "infer_ms": infer_ms,
                    "step_s": self.step_s,
                    "buffer_s": buffer_s,
                    "queue_chunks": queue_chunks,
                    "queue_tail_s": queue_tail_s,
                    "full_text": curr_text,      # 전체 누적 텍스트
                    "new_text": curr_text,        # 현재 전체 텍스트
                    "accumulate_mode": True,
                },
            )
            self.bus.publish(self.out_topic, enriched)
