from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from asrFlow.utils.env import env_bool_any, env_int_any, env_lang_any, env_str_any, env_value


@dataclass(frozen=True)
class RuntimeEnvLoadResult:
    project_root: Path
    requested_path: Path | None
    loaded_paths: tuple[Path, ...]
    missing_paths: tuple[Path, ...]

    @property
    def base_dir(self) -> Path:
        if self.loaded_paths:
            return self.loaded_paths[0].parent
        if self.requested_path is not None:
            return self.requested_path.parent
        return self.project_root


def neuroflow_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_env_path(
    env_path: str | os.PathLike[str] | None,
    *,
    base_dir: Path | None = None,
) -> Path | None:
    if env_path is None:
        return None

    raw = str(env_path).strip()
    if not raw:
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ((base_dir or Path.cwd()) / path).resolve()
    return path


def _resolve_runtime_path(raw: str, *, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_model_path(*, config_base_dir: Path | None = None) -> str | None:
    raw = env_value("ASRFLOW_STT_MODEL_PATH", "MODEL_DIR")
    if raw is None:
        return None

    base_dir = config_base_dir or neuroflow_project_root()
    return str(_resolve_runtime_path(raw, base_dir=base_dir))


def prepare_runtime_env(*, config_base_dir: Path | None = None) -> None:
    base_dir = config_base_dir or neuroflow_project_root()

    whisper_model_dir = env_value("WHISPER_MODEL_DIR")
    if whisper_model_dir is not None:
        os.environ["WHISPER_MODEL_DIR"] = str(_resolve_runtime_path(whisper_model_dir, base_dir=base_dir))
    else:
        model_path = _resolve_model_path(config_base_dir=base_dir)
        if model_path is not None:
            os.environ["WHISPER_MODEL_DIR"] = str(Path(model_path).parent)

    token = env_value("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")
    if token and env_value("HUGGINGFACE_HUB_TOKEN") is None:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token


def load_runtime_env(
    env_path: str | os.PathLike[str] | None = None,
    *,
    base_dir: Path | None = None,
    include_project_env: bool = True,
) -> RuntimeEnvLoadResult:
    project_root = neuroflow_project_root()
    project_env_path = project_root / ".env"
    requested_path = resolve_env_path(env_path, base_dir=base_dir)

    candidates: list[Path] = []
    if requested_path is not None:
        candidates.append(requested_path)
    elif include_project_env:
        candidates.append(project_env_path)

    if include_project_env and project_env_path not in candidates:
        candidates.append(project_env_path)

    loaded_paths: list[Path] = []
    missing_paths: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)

        if resolved.is_file():
            load_dotenv(dotenv_path=resolved, override=False)
            loaded_paths.append(resolved)
        else:
            missing_paths.append(resolved)

    result = RuntimeEnvLoadResult(
        project_root=project_root,
        requested_path=requested_path,
        loaded_paths=tuple(loaded_paths),
        missing_paths=tuple(missing_paths),
    )
    prepare_runtime_env(config_base_dir=result.base_dir)
    return result


def build_processor_from_env(*, config_base_dir: Path | None = None):
    prepare_runtime_env(config_base_dir=config_base_dir)

    from asrFlow.processors.miso_stt_asr import MisoSttAsrProcessor

    backend_kwargs: dict[str, object] = {}

    max_new_tokens = env_value("ASRFLOW_STT_MAX_NEW_TOKENS")
    if max_new_tokens is not None:
        backend_kwargs["max_new_tokens"] = int(max_new_tokens)

    max_inference_batch_size = env_value("ASRFLOW_STT_MAX_INFERENCE_BATCH_SIZE")
    if max_inference_batch_size is not None:
        backend_kwargs["max_inference_batch_size"] = int(max_inference_batch_size)

    forced_aligner = env_value("ASRFLOW_STT_FORCED_ALIGNER")
    if forced_aligner is not None:
        backend_kwargs["forced_aligner"] = forced_aligner

    if env_value("ASRFLOW_STT_RETURN_TIMESTAMPS", "RETURN_TIMESTAMPS") is not None:
        backend_kwargs["return_time_stamps"] = env_bool_any(
            ("ASRFLOW_STT_RETURN_TIMESTAMPS", "RETURN_TIMESTAMPS"),
            False,
        )

    return MisoSttAsrProcessor(
        backend=env_str_any(("ASRFLOW_STT_BACKEND",), "qwen_transformers"),
        model_name=env_str_any(("ASRFLOW_STT_MODEL", "MODEL_ID"), "Qwen/Qwen3-ASR-0.6B"),
        model_path=_resolve_model_path(config_base_dir=config_base_dir),
        device=env_str_any(("ASRFLOW_STT_DEVICE", "ASR_DEVICE"), "auto"),
        fp16=env_bool_any(("ASRFLOW_STT_FP16", "ASR_FP16"), True),
        language=env_lang_any(("ASRFLOW_STT_LANGUAGE", "ASR_LANGUAGE"), "auto"),
        task=env_str_any(("ASRFLOW_STT_TASK",), "transcribe"),
        beam_size=env_int_any(("ASRFLOW_STT_BEAM_SIZE",), 5),
        backend_kwargs=backend_kwargs or None,
    )


__all__ = [
    "RuntimeEnvLoadResult",
    "build_processor_from_env",
    "load_runtime_env",
    "neuroflow_project_root",
    "prepare_runtime_env",
    "resolve_env_path",
]
