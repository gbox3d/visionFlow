from __future__ import annotations

from collections import deque
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from common.contracts.packets import AsrResultPacket
from asrFlow.vendors.whisper.core.config import HF_CACHE_DIR


class QwenStreamingAsrProcessor:
    """Qwen ASR streaming processor using transformers backend.

    Replicates the streaming logic from qwen_asr's vLLM-only streaming API
    (buffer -> accumulate -> prefix rollback -> decode) using the transformers
    backend so that it runs on Windows without vLLM.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        model_path: str | None = None,
        language: str | None = None,
        max_new_tokens: int = 512,
        chunk_size_sec: float = 2.0,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        max_accum_samples: int = 2000,
    ):
        self.backend = "qwen_transformers"
        self.model_name = model_name
        self.model_path = model_path
        self.language = language
        self.max_new_tokens = int(max_new_tokens)
        self.chunk_size_sec = float(chunk_size_sec)
        self.unfixed_chunk_num = int(unfixed_chunk_num)
        self.unfixed_token_num = int(unfixed_token_num)
        self.max_accum_samples = max(1, int(max_accum_samples))

        self._qwen_model: Any | None = None

        # Streaming state
        self._chunk_size_samples: int = 0
        self._chunk_id: int = 0
        self._buffer: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._audio_accum_queue: deque[np.ndarray] = deque()
        self._audio_accum_samples: int = 0
        self._prompt_raw: str = ""
        self._force_language: str | None = None
        self._raw_decoded: str = ""
        self._text: str = ""
        self._language_detected: str = ""

        self.load_source = "explicit_model_path" if model_path else "hf_hub_or_cache"
        self.model_id = str(Path(model_path).expanduser()) if model_path else model_name
        self.resolved_local_path = str(Path(model_path).expanduser()) if model_path else None
        self.effective_model_path = self.resolved_local_path
        self.cache_dir = str(HF_CACHE_DIR)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._qwen_model is not None:
            return

        from qwen_asr import Qwen3ASRModel
        from qwen_asr.inference.utils import (
            SAMPLE_RATE,
            normalize_language_name,
            validate_language,
        )
        from asrFlow.vendors.qwen_asr.transcriber import _normalize_language

        token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
        if token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token

        os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "hub"))

        checkpoint = str(Path(self.model_path).expanduser()) if self.model_path else self.model_name

        print(
            "[QwenStreamingAsrProcessor] 모델 로딩: "
            f"backend={self.backend}, model={self.model_name}, "
            f"model_path={self.model_path or '-'}, chunk={self.chunk_size_sec}s"
        )
        t0 = time.time()

        self._qwen_model = Qwen3ASRModel.from_pretrained(
            checkpoint,
            max_new_tokens=self.max_new_tokens,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

        # Resolve forced language (e.g. "ko" -> "Korean", "auto" -> None)
        lang = _normalize_language(self.language)
        if lang is not None and lang.strip():
            ln = normalize_language_name(lang)
            validate_language(ln)
            self._force_language = ln

        self._prompt_raw = self._qwen_model._build_text_prompt(
            context="", force_language=self._force_language
        )
        self._chunk_size_samples = max(1, int(round(self.chunk_size_sec * SAMPLE_RATE)))

        dt = time.time() - t0
        print(f"[QwenStreamingAsrProcessor] 모델 로딩 완료 ({dt:.1f}s)")

    # ------------------------------------------------------------------
    # Streaming inference helpers
    # ------------------------------------------------------------------

    def _infer_chunk(self, audio: np.ndarray, prompt: str) -> str:
        """Run single-sample inference using transformers generate()."""
        m = self._qwen_model
        inputs = m.processor(
            text=[prompt],
            audio=[audio],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(m.device).to(m.dtype)

        with torch.no_grad():
            output = m.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        seqs = output.sequences if hasattr(output, "sequences") else output
        input_len = inputs["input_ids"].shape[1]
        decoded = m.processor.batch_decode(
            seqs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0]

    def _build_prefix(self) -> str:
        """Build prefix text from previous output with rollback strategy."""
        if self._chunk_id < self.unfixed_chunk_num:
            return ""

        tokenizer = self._qwen_model.processor.tokenizer
        cur_ids = tokenizer.encode(self._raw_decoded)
        k = self.unfixed_token_num
        while True:
            end_idx = max(0, len(cur_ids) - k)
            prefix = tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
            if "\ufffd" not in prefix:
                break
            if end_idx == 0:
                prefix = ""
                break
            k += 1
        return prefix

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Accumulate audio, run inference, update streaming state."""
        from qwen_asr.inference.utils import parse_asr_output

        self._append_audio_accum(chunk)
        audio_accum = self._get_audio_accum()

        prefix = self._build_prefix()
        prompt = self._prompt_raw + prefix

        gen_text = self._infer_chunk(audio_accum, prompt)
        self._raw_decoded = (prefix + gen_text) if prefix else gen_text

        lang, txt = parse_asr_output(self._raw_decoded, user_language=self._force_language)
        self._language_detected = lang
        self._text = txt
        self._chunk_id += 1

    def _append_audio_accum(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return

        c = np.asarray(chunk, dtype=np.float32)
        if c.size >= self.max_accum_samples:
            tail = c[-self.max_accum_samples :].copy()
            self._audio_accum_queue.clear()
            self._audio_accum_queue.append(tail)
            self._audio_accum_samples = int(tail.size)
            return

        self._audio_accum_queue.append(c.copy())
        self._audio_accum_samples += int(c.size)

        while self._audio_accum_samples > self.max_accum_samples and self._audio_accum_queue:
            overflow = self._audio_accum_samples - self.max_accum_samples
            head = self._audio_accum_queue[0]
            if overflow >= head.size:
                self._audio_accum_queue.popleft()
                self._audio_accum_samples -= int(head.size)
            else:
                self._audio_accum_queue[0] = head[overflow:].copy()
                self._audio_accum_samples -= int(overflow)
                break

    def _get_audio_accum(self) -> np.ndarray:
        if not self._audio_accum_queue:
            return np.zeros((0,), dtype=np.float32)
        if len(self._audio_accum_queue) == 1:
            return self._audio_accum_queue[0]
        return np.concatenate(list(self._audio_accum_queue))

    # ------------------------------------------------------------------
    # Audio normalization / text delta
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_audio(audio_chunk: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio_chunk)
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        if audio.dtype != np.float32:
            return audio.astype(np.float32)
        return audio

    @staticmethod
    def _delta_text(prev_text: str, curr_text: str) -> str:
        prev = prev_text.strip()
        curr = curr_text.strip()
        if not prev:
            return curr
        if curr.startswith(prev):
            return curr[len(prev):].strip()
        return curr

    # ------------------------------------------------------------------
    # Packet construction
    # ------------------------------------------------------------------

    def _build_packet(
        self,
        prev_text: str,
        *,
        seq: int,
        source_id: str,
        finished: bool,
    ) -> Optional[AsrResultPacket]:
        text = self._text.strip()
        if not text or text == prev_text.strip():
            return None

        language = self._language_detected or self.language or "auto"
        new_text = self._delta_text(prev_text, text)

        return AsrResultPacket(
            text=text,
            segments=[],
            language=language,
            ts_ms=int(time.time() * 1000),
            seq=seq,
            source_id=source_id,
            meta={
                "backend": self.backend,
                "model_name": self.model_name,
                "model_path": self.model_path,
                "model_id": self.model_id,
                "load_source": self.load_source,
                "resolved_local_path": self.resolved_local_path,
                "effective_model_path": self.effective_model_path,
                "resolved_local_path_abs": (
                    str(Path(self.resolved_local_path).resolve()) if self.resolved_local_path else None
                ),
                "effective_model_path_abs": (
                    str(Path(self.effective_model_path).resolve()) if self.effective_model_path else None
                ),
                "cache_dir": self.cache_dir,
                "cache_root_abs": str(Path(self.cache_dir).resolve()),
                "native_streaming": True,
                "stream_finished": finished,
                "stream_chunk_sec": self.chunk_size_sec,
                "max_accum_samples": self.max_accum_samples,
                "unfixed_chunk_num": self.unfixed_chunk_num,
                "unfixed_token_num": self.unfixed_token_num,
                "full_text": text,
                "new_text": new_text,
            },
        )

    # ------------------------------------------------------------------
    # Public API (same interface as before)
    # ------------------------------------------------------------------

    def feed_audio(
        self,
        audio_chunk: np.ndarray,
        *,
        samplerate: int = 16000,
        seq: int = 0,
        source_id: str = "",
    ) -> Optional[AsrResultPacket]:
        if int(samplerate) != 16000:
            raise ValueError(f"Qwen native streaming은 16kHz PCM만 지원합니다. got={samplerate}")

        self._ensure_model()

        audio = self._normalize_audio(audio_chunk)
        if audio.size == 0:
            return None

        prev_text = self._text
        self._buffer = np.concatenate([self._buffer, audio])

        while self._buffer.shape[0] >= self._chunk_size_samples:
            chunk = self._buffer[: self._chunk_size_samples]
            self._buffer = self._buffer[self._chunk_size_samples:]
            self._process_chunk(chunk)

        return self._build_packet(prev_text, seq=seq, source_id=source_id, finished=False)

    def flush(self, *, seq: int = 0, source_id: str = "") -> Optional[AsrResultPacket]:
        self._ensure_model()

        if self._buffer.size == 0:
            return None

        prev_text = self._text
        tail = self._buffer
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._process_chunk(tail)

        return self._build_packet(prev_text, seq=seq, source_id=source_id, finished=True)

    def warmup(self) -> None:
        self._ensure_model()

    def get_model_source_label(self) -> str:
        self._ensure_model()
        return self.model_id

    def get_cache_root_abs(self) -> str:
        return str(Path(self.cache_dir).resolve())

    def reset(self) -> None:
        """Clear streaming state without unloading the model."""
        self._chunk_id = 0
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._audio_accum_queue.clear()
        self._audio_accum_samples = 0
        self._raw_decoded = ""
        self._text = ""
        self._language_detected = ""

    def close(self) -> None:
        self.reset()
        self._qwen_model = None
        print("[QwenStreamingAsrProcessor] 프로세서 종료")
