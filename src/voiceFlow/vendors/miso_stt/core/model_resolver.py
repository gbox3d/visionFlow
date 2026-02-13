from __future__ import annotations

from pathlib import Path


def _resolve_local_hf_model(path: Path) -> Path | None:
    if not path.exists():
        return None

    if (path / "config.json").exists():
        return path

    snapshots = path / "snapshots"
    if snapshots.is_dir():
        for snap in sorted(snapshots.iterdir(), reverse=True):
            if (snap / "config.json").exists():
                return snap

    for model_dir in sorted(path.glob("models--*"), reverse=True):
        model_snapshots = model_dir / "snapshots"
        if model_snapshots.is_dir():
            for snap in sorted(model_snapshots.iterdir(), reverse=True):
                if (snap / "config.json").exists():
                    return snap

    return None


def _resolve_local_ct2_model(path: Path) -> Path | None:
    if not path.exists():
        return None

    if (path / "model.bin").exists():
        return path

    snapshots = path / "snapshots"
    if snapshots.is_dir():
        for snap in sorted(snapshots.iterdir(), reverse=True):
            if (snap / "model.bin").exists():
                return snap

    for model_dir in sorted(path.glob("models--*"), reverse=True):
        model_snapshots = model_dir / "snapshots"
        if model_snapshots.is_dir():
            for snap in sorted(model_snapshots.iterdir(), reverse=True):
                if (snap / "model.bin").exists():
                    return snap

    return None


_HF_ADAPTER_MARKERS = (
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
)


def describe_hf_model_path_issue(path: Path) -> str:
    if not path.exists():
        return "Path does not exist."
    if not path.is_dir():
        return "Path is not a directory. Please pass a model directory."

    has_adapter = any((path / marker).exists() for marker in _HF_ADAPTER_MARKERS)
    if has_adapter and not (path / "config.json").exists():
        return (
            "Detected LoRA/PEFT adapter-only directory. "
            "Unmerged adapters are not supported. Merge adapter into base Whisper model "
            "and pass the merged checkpoint directory containing config.json."
        )

    return (
        "Could not find Whisper model files (config.json or snapshot with config.json). "
        "Pass a merged HF Whisper checkpoint directory."
    )


def describe_ct2_model_path_issue(path: Path) -> str:
    if not path.exists():
        return "Path does not exist."
    if not path.is_dir():
        return "Path is not a directory. Please pass a CT2 model directory."

    return (
        "Could not find CT2 model files (model.bin or snapshot with model.bin). "
        "Pass a valid faster-whisper converted model directory."
    )

