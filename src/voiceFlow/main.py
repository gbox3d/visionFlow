"""voiceFlow sample launcher for source/device tools."""

from __future__ import annotations

import importlib
import sys

SAMPLES = [
    {
        "category": "Device",
        "items": [
            ("list_microphones", "voiceFlow.sample.list_microphones", "오디오 입력 장치 목록"),
        ],
    },
    {
        "category": "Source",
        "items": [
            ("simple_mic_test", "voiceFlow.sample.simple_mic_test", "마이크 source smoke test"),
            ("mic_level_monitor", "voiceFlow.sample.mic_level_monitor", "실시간 입력 레벨 모니터"),
        ],
    },
    {
        "category": "App",
        "items": [
            ("asr_chunk_realtime", "neuroflow.app.asr_chunk_realtime", "로컬 마이크 -> ASR chunk 실험 UI"),
            ("audiomi_asr_chunk_realtime", "neuroflow.app.audiomi_asr_chunk_realtime", "audioMi -> ASR chunk 실험 UI"),
            ("asr_stream_realtime", "neuroflow.app.asr_stream_realtime", "로컬 마이크 -> native stream ASR UI"),
        ],
    },
    {
        "category": "Network",
        "items": [
            ("microphone_client", "voiceFlow.sample.microphone_client", "마이크 녹음 후 NFCP 서버로 요청"),
        ],
    },
]


def _print_menu() -> dict[int, tuple[str, str]]:
    print()
    print("=" * 48)
    print("  voiceFlow Launcher")
    print("=" * 48)

    idx = 1
    index_map: dict[int, tuple[str, str]] = {}
    for group in SAMPLES:
        print(f"\n  [{group['category']}]")
        for name, module, desc in group["items"]:
            print(f"    {idx}) {name:<24s} - {desc}")
            index_map[idx] = (name, module)
            idx += 1

    print("\n    0) 종료")
    print()
    return index_map


def main() -> None:
    original_argv = sys.argv[:]

    while True:
        index_map = _print_menu()

        try:
            choice = input("  선택: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "0" or choice.lower() == "q":
            break

        try:
            num = int(choice)
        except ValueError:
            print("  [!] 올바른 번호를 입력하세요.")
            continue

        if num not in index_map:
            print(f"  [!] 1~{len(index_map)} 또는 0을 입력하세요.")
            continue

        name, module_path = index_map[num]
        print(f"\n  >>> {name} 실행 중...\n")

        try:
            sys.argv = [name]
            mod = importlib.import_module(module_path)
            entry = getattr(mod, "main", None)
            if not callable(entry):
                raise RuntimeError(f"{module_path} does not expose main()")
            entry()
        except KeyboardInterrupt:
            print(f"\n  [{name}] 중단됨")
        except Exception as exc:
            print(f"\n  [{name}] 오류: {exc}")
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    main()
