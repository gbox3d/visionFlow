from __future__ import annotations

from pathlib import Path


def require_model_file(model_path: str) -> str:
    path = Path(model_path)
    if path.is_file():
        return str(path)

    raise FileNotFoundError(
        f"Missing MediaPipe model: {model_path}. "
        "Run `uv run nf-vision-models-download` from the repo root, "
        "or set the corresponding *_MODEL_PATH value to a valid local file."
    )
