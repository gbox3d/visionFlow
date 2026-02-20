# voiceFlow

`voiceFlow`는 `TopicBus` 기반 실시간 음성 파이프라인입니다.

## Pipeline

`audioMi TCP` -> `AudioMiSource` -> `TopicBus("audio/raw")` -> `AccumulateAsrWorker` -> `TopicBus("text/asr")` -> `PySide6 UI`

## 핵심 구성

- `sources/audiomi_source.py`
  - audioMi 서버에서 PCM16 mono 16kHz 수신
  - `audio/raw` 토픽으로 publish
- `processors/miso_stt_asr.py`
  - `miso_stt` 백엔드(`ct2`, `hf_generate`, `hf_pipeline`) 래핑
  - 모델 로딩/웜업/추론 및 메타 정보 제공
- `workers/accumulate_asr_worker.py`
  - ingest/infer 스레드 분리
  - 고정 청크(`step_s`) 큐 누적 후 전체 버퍼 추론
  - `max_window_s` 초과 시 앞 청크 트림
  - 워밍업 상태(`text/asr_status`) publish
  - 선택적 로그 저장(`logs/*.wav`, `logs/*.txt`)
- `sample/audiomi_asr_realtime.py`
  - audioMi 실시간 누적 ASR UI
  - 모델명/캐시 루트/워밍업 진행률/queue/infer 시간 표시
- `utils/env.py`
  - `.env` 파싱 공용 유틸 (`env_str`, `env_int`, `env_bool`, `env_lang`, `env_*_any`)
- `utils/audio_device.py`
  - 마이크 디바이스 파싱/카메라 path 기반 마이크 매칭 유틸
- `utils/text.py`
  - UI 텍스트 줄바꿈 공용 유틸 (`wrap_text`)

## 실행

```bash
uv run python -m voiceFlow.sample.audiomi_asr_realtime
```

기존 단일 실시간 샘플:

```bash
uv run python -m voiceFlow.sample.asr_realtime
```

두 샘플(`asr_realtime.py`, `audiomi_asr_realtime.py`)은 공통으로 `voiceFlow.utils.env`를 사용합니다.

## 환경 변수 (.env)

필수/주요 항목:

- `AUDIOMI_HOST` (기본 `127.0.0.1`)
- `AUDIOMI_PORT` (기본 `26070`)
- `AUDIOMI_CHECKCODE` (기본 `20250918`)
- `VOICEFLOW_STT_BACKEND` (`ct2` | `hf_generate` | `hf_pipeline`)
- `VOICEFLOW_STT_MODEL` (예: `large-v3`)
- `VOICEFLOW_STT_MODEL_PATH` (비우면 자동 해석)
- `VOICEFLOW_STT_DEVICE` (기본 `auto`)
- `VOICEFLOW_STT_FP16` (기본 `true`)
- `VOICEFLOW_STT_LANGUAGE` (`auto` 또는 언어 코드)
- `VOICEFLOW_STT_TASK` (기본 `transcribe`)
- `VOICEFLOW_STT_BEAM_SIZE` (CT2 전용)
- `VOICEFLOW_STT_TEMPERATURE` (기본 `0.0`)
- `VOICEFLOW_STT_CT2_VAD_FILTER` (기본 `false`)
- `VOICEFLOW_STT_SAMPLERATE` (기본 `16000`)
- `ENABLE_LOGGING` (`true`일 때만 `logs/` 저장)

## 모델 로드 우선순위

1. UI/ENV의 `VOICEFLOW_STT_MODEL_PATH`
2. 모델명(`VOICEFLOW_STT_MODEL`) 기반 로컬/HF 캐시 해석

UI에는 시작 후 아래 정보가 표시됩니다.

- 모델명(또는 model id)
- `cache root(abs)`

## 누적 모드 동작

- 입력 오디오는 무음 포함 전체 수신/누적
- `step_s`마다 누적 버퍼 전체를 재추론
- 워밍업(`max_window_s - step_s`) 이전 결과는 publish하지 않음
- 워밍업 완료 후 중간 문장 일관성 필터로 급격한 변화를 차단

## 로그

`ENABLE_LOGGING=true`일 때:

- 성공 추론: `logs/<timestamp>.wav` + `logs/<timestamp>.txt`
- 필터 실패: `logs/<timestamp>_FAIL.wav` + `logs/<timestamp>_FAIL.txt`
- 실패 이력: `logs/filter_failures.log`

## 디바이스 리스팅 (배포/개발 공용)

마이크/카메라 디바이스 확인용 도구:

- 콘솔: `device_lister.py`
- UI: `device_lister_ui.py`

개발 환경 실행:

```bash
uv run python device_lister.py
uv run python device_lister_ui.py
```

배포 빌드:

```powershell
.\scripts\build_device_lister.ps1 -Clean
.\scripts\build_device_lister_ui.ps1 -Clean
```

생성 결과:

- `dist/device_lister/device_lister.exe`
- `dist/device_lister_ui/device_lister_ui.exe`

UI 버전에서는 카메라 path를 선택 후 `Copy Camera Path`로 클립보드에 복사할 수 있습니다.
