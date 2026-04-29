from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TtsModelAsset:
    key: str
    filename: str
    url: str
    description: str


MODEL_ASSETS: tuple[TtsModelAsset, ...] = (
    TtsModelAsset(
        key="piper-kss-ko-model",
        filename="piper-kss-korean.onnx",
        url="https://huggingface.co/neurlang/piper-onnx-kss-korean/resolve/main/piper-kss-korean.onnx",
        description="Piper ONNX Korean KSS model",
    ),
    TtsModelAsset(
        key="piper-kss-ko-config",
        filename="piper-kss-korean.onnx.json",
        url="https://huggingface.co/neurlang/piper-onnx-kss-korean/raw/main/piper-kss-korean.onnx.json",
        description="Piper ONNX Korean KSS config",
    ),
)


def _default_output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / "pyproject.toml").exists():
        return repo_root / "models"
    return Path.cwd() / "models"


def _iter_selected_assets(keys: list[str] | None) -> list[TtsModelAsset]:
    if not keys:
        return list(MODEL_ASSETS)
    if "piper-kss-ko" in keys:
        keys = ["piper-kss-ko-model", "piper-kss-ko-config"]
    wanted = set(keys)
    return [asset for asset in MODEL_ASSETS if asset.key in wanted]


def _download_file(url: str, target: Path) -> None:
    tmp_target = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp_target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_target.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download default NeuroFlow TTS models into ./models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="target directory for downloaded model files (default: repo-root/models)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["piper-kss-ko", *(asset.key for asset in MODEL_ASSETS)],
        help="download only selected assets",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--list", action="store_true", help="print available asset keys and exit")
    args = parser.parse_args()

    if args.list:
        print(f"{'piper-kss-ko':24s} -> model + config  (Korean Piper KSS bundle)")
        for asset in MODEL_ASSETS:
            print(f"{asset.key:24s} -> {asset.filename}  ({asset.description})")
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_assets = _iter_selected_assets(args.only)
    if not selected_assets:
        print("No TTS model assets selected.", file=sys.stderr)
        raise SystemExit(2)

    print(f"[nf-tts-models-download] target_dir={output_dir}")
    for asset in selected_assets:
        target = output_dir / asset.filename
        if target.exists() and not args.force:
            print(f"[skip] {asset.filename} already exists")
            continue

        print(f"[download] {asset.filename}")
        try:
            _download_file(asset.url, target)
        except Exception as exc:
            try:
                broken = target.with_suffix(target.suffix + ".part")
                if broken.exists():
                    broken.unlink()
            except Exception:
                pass
            raise RuntimeError(f"failed to download {asset.filename}: {exc}") from exc

        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"[ok] {asset.filename} ({size_mb:.1f} MiB)")

    print("[nf-tts-models-download] done")
