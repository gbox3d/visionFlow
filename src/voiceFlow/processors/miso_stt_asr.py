from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

from voiceFlow.pipeline.packet import AsrResultPacket
from voiceFlow.vendors.miso_stt.transcriber import WhisperTranscriber

BackendName = Literal["ct2", "hf_generate", "hf_pipeline"]


class MisoSttAsrProcessor:
    def __init__(
        self,
        backend: BackendName = "ct2",
        model_name: str = "large-v3",
        model_path: str | None = None,
        device: str = "auto",
        fp16: bool = True,
        language: str | None = None,
        task: str = "transcribe",
        beam_size: int = 5,
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.fp16 = fp16
        self.language = language
        self.task = task
        self.beam_size = int(beam_size)
        self.backend_kwargs = dict(backend_kwargs) if backend_kwargs else {}

        self._transcriber: WhisperTranscriber | None = None
        self._ct2_fallback_cpu = False
        self._ct2_fallback_reason: str | None = None

    def _ensure_transcriber(self) -> None:
        if self._transcriber is not None:
            return

        kwargs = dict(self.backend_kwargs)
        if self.backend == "ct2":
            kwargs.setdefault("beam_size", self.beam_size)

        print(
            "[MisoSttAsrProcessor] 모델 로딩: "
            f"backend={self.backend}, model={self.model_name}, "
            f"model_path={self.model_path or '-'}, device={self.device}, fp16={self.fp16}"
        )
        t0 = time.time()
        try:
            self._transcriber = WhisperTranscriber(
                backend=self.backend,
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device,
                fp16=self.fp16,
                **kwargs,
            )
        except Exception as e:
            msg = str(e).lower()
            is_ct2_cuda_load_issue = (
                self.backend == "ct2"
                and self.device in ("auto", "cuda")
                and any(k in msg for k in ("cuda", "cublas", "cudnn", "ctranslate2", "cublas64_12", "cudnn64_9"))
            )
            if not is_ct2_cuda_load_issue:
                raise

            self._ct2_fallback_cpu = True
            self._ct2_fallback_reason = str(e).strip() or repr(e)
            print(
                "[MisoSttAsrProcessor] 경고: CT2 GPU 초기화 실패 -> CPU fallback 적용 | "
                f"reason={self._ct2_fallback_reason}"
            )
            self._transcriber = WhisperTranscriber(
                backend="ct2",
                model_name=self.model_name,
                model_path=self.model_path,
                device="cpu",
                fp16=False,
                **kwargs,
            )
        dt = time.time() - t0
        print(f"[MisoSttAsrProcessor] 모델 로딩 완료 ({dt:.1f}s)")
        if self._transcriber is not None:
            backend_obj = self._transcriber.backend
            effective_model_path = getattr(backend_obj, "effective_model_path", None)
            cache_root = getattr(backend_obj, "download_root", None) or getattr(backend_obj, "cache_dir", None)
            model_id = getattr(backend_obj, "model_id", None)
            if effective_model_path:
                print(f"[MisoSttAsrProcessor] model path(abs): {Path(str(effective_model_path)).resolve()}")
            if cache_root:
                print(f"[MisoSttAsrProcessor] cache root(abs): {Path(str(cache_root)).resolve()}")
            if model_id:
                print(f"[MisoSttAsrProcessor] model id: {model_id}")

    def process(
        self,
        audio_chunk: np.ndarray,
        samplerate: int = 16000,
        seq: int = 0,
        source_id: str = "",
    ) -> Optional[AsrResultPacket]:
        self._ensure_transcriber()
        assert self._transcriber is not None

        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        rms = float(np.sqrt(np.mean(audio_chunk**2)))

        text, segments = self._transcriber.transcribe_full_timeline(
            waveform=audio_chunk,
            sample_rate=samplerate,
            language=self.language,
            task=self.task,
        )
        text = (text or "").strip()
        if not text:
            return None

        normalized_segments = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in segments
        ]

        backend_obj = self._transcriber.backend
        language = self.language or "auto"
        info_language = getattr(getattr(backend_obj, "model", None), "language", None)
        if info_language and self.language is None:
            language = str(info_language)

        meta: Dict[str, Any] = {
            "backend": self.backend,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "rms": rms,
            "task": self.task,
            "beam_size": self.beam_size if self.backend == "ct2" else None,
            "ct2_fallback_cpu": self._ct2_fallback_cpu,
            "fallback_reason": self._ct2_fallback_reason,
        }
        for key in (
            "load_source",
            "effective_model_path",
            "resolved_local_path",
            "compute_type",
            "device",
            "model_id",
            "download_root",
            "cache_dir",
        ):
            value = self._backend_attr(backend_obj, key)
            if value is None:
                continue
            if hasattr(value, "type"):
                value = value.type
            meta[key] = value
        effective_model_path = meta.get("effective_model_path")
        if effective_model_path:
            meta["effective_model_path_abs"] = str(Path(str(effective_model_path)).resolve())
        resolved_local_path = meta.get("resolved_local_path")
        if resolved_local_path:
            meta["resolved_local_path_abs"] = str(Path(str(resolved_local_path)).resolve())
        cache_root = meta.get("download_root") or meta.get("cache_dir")
        if cache_root:
            meta["cache_root_abs"] = str(Path(str(cache_root)).resolve())

        return AsrResultPacket(
            text=text,
            segments=normalized_segments,
            language=language,
            ts_ms=int(time.time() * 1000),
            seq=seq,
            source_id=source_id,
            meta=meta,
        )

    def warmup(self) -> None:
        """모델을 미리 로딩해 스트림 수신 전에 초기화를 끝낸다."""
        self._ensure_transcriber()

    @staticmethod
    def _backend_attr(backend_obj: Any, name: str) -> Any:
        if hasattr(backend_obj, name):
            return getattr(backend_obj, name)
        nested = getattr(backend_obj, "backend", None)
        if nested is not None and hasattr(nested, name):
            return getattr(nested, name)
        return None

    def get_model_source_label(self) -> str:
        """현재 로딩된 모델명을 UI 표시에 적합한 문자열로 반환."""
        self._ensure_transcriber()
        assert self._transcriber is not None
        backend_obj = self._transcriber.backend

        model_id = str(self._backend_attr(backend_obj, "model_id") or "").strip()
        return model_id or self.model_name

    def get_cache_root_abs(self) -> str:
        """현재 백엔드의 캐시 루트 절대 경로를 반환한다."""
        self._ensure_transcriber()
        assert self._transcriber is not None
        backend_obj = self._transcriber.backend
        cache_root = self._backend_attr(backend_obj, "download_root") or self._backend_attr(backend_obj, "cache_dir")
        if not cache_root:
            return ""
        return str(Path(str(cache_root)).resolve())

    def close(self) -> None:
        self._transcriber = None
        print("[MisoSttAsrProcessor] 프로세서 종료")
