# Migration Map

## 목적

이 문서는 현재 repo 안의 자산을 앞으로 어떤 구조로 옮길지, 그리고 무엇이 이미 되었고 무엇이 아직 안 되었는지 상태 중심으로 정리한 매핑표다.

상태 표기:

- `Done`
- `Bridge`
- `Pending`
- `External`
- `Keep`


## 1. 최종 목표 구조

```text
src/
  common/
  asrFlow/
  llmFlow/
  ttsFlow/
  visionflow/
  backend/
```


## 2. 현재 repo 내부 매핑

### 2.1 Common

| 현재 자산 | 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `src/common/contracts/packets.py` | 유지 | `Done` | 공통 packet 정의 시작됨 |
| `src/common/contracts/job.py` | 유지 | `Done` | request/result 상태 모델 있음 |
| `src/common/protocols/common_protocol.md` | 유지 | `Done` | 공통 프로토콜 문서 초안 |
| `src/common/protocols/nfcp.py` | 유지 | `Done` | 문서 대응 구현 있음 |
| `src/voiceFlow/utils/env.py` | `src/common/utils/env.py` | `Pending` | env 유틸 공통 승격 후보 |
| `src/visionflow/pipeline/bus.py` | `src/common/bus/topic_bus.py` 또는 현 위치 유지 | `Pending` | 공통 bus로 올릴지 결정 필요 |

### 2.2 ASR

| 현재 자산 | 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `src/asrFlow/gateways/tcp_asr_server.py` | 유지 | `Done` | NFCP 기반 ASR 서버 |
| `src/asrFlow/sample/microphone_client.py` | 유지 | `Done` | 마이크 녹음 후 서버 요청 |
| `src/asrFlow/processors/miso_stt_asr.py` | 유지 | `Bridge` | 실제 구현은 아직 `voiceFlow` 참조 |
| `src/voiceFlow/processors/miso_stt_asr.py` | `src/asrFlow/processors/miso_stt_asr.py` | `Pending` | 핵심 processor 추출 필요 |
| `src/voiceFlow/workers/asr_worker.py` | `src/asrFlow/workers/asr_worker.py` | `Pending` | worker 계층 이관 |
| `src/voiceFlow/workers/accumulate_asr_worker.py` | `src/asrFlow/workers/accumulate_asr_worker.py` | `Pending` | 실시간 누적 worker |
| `src/voiceFlow/sources/audiomi_source.py` | `src/asrFlow/sources/audiomi_source.py` | `Pending` | 외부 PCM 입력 |
| `src/voiceFlow/sources/microphone_source.py` | `src/asrFlow/sources/microphone_source.py` | `Pending` | 로컬 마이크 입력 |
| `src/voiceFlow/vendors/miso_stt/*` | `src/asrFlow/vendors/miso_stt/*` | `Pending` | 실제 backend/core 구현 |
| `src/voiceFlow/utils/audio_device.py` | `src/asrFlow/utils/audio_device.py` | `Pending` | 개발/디바이스 보조 유틸 |
| `src/voiceFlow/sample/simple_mic_test.py` | `src/asrFlow/sample/simple_mic_test.py` | `Pending` | 간단한 입력 테스트 |
| `src/voiceFlow/sample/list_microphones.py` | `src/asrFlow/sample/list_microphones.py` | `Pending` | 디바이스 점검 샘플 |

### 2.3 Vision

| 현재 자산 | 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `src/visionflow/**/*` | 현 위치 유지 | `Keep` | 1차 이동 대상이 아님 |
| `src/visionflow/pipeline/bus.py` | 후속 연동 포인트 정의 | `Keep` | 패턴 재사용 가능성 높음 |


## 3. 아직 repo 밖에 있는 후보 자산

### 3.1 LLM

| 외부 자산 | 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `miso_kiosk/llm/chat_memory.py` | `src/llmFlow/memory/chat_memory.py` | `External` | 메모리 핵심 후보 |
| `miso_kiosk/llm/summarizer.py` | `src/llmFlow/memory/summarizer.py` | `External` | provider 분리 필요 |
| `miso_kiosk/llm/ollama_chat.py` | `src/llmFlow/providers/ollama_chat.py` | `External` | LLM provider 씨앗 |
| `miso_kiosk/llm/*.txt` | `src/llmFlow/prompts/legacy/*` | `External` | 프롬프트 자산 |

### 3.2 TTS

| 외부 자산 | 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `voiceAI/TTS/server.py` | `src/ttsFlow/gateways/legacy_tcp_tts_server.py` | `External` | gateway/engine 분리 필요 |
| `voiceAI/TTS/server_fastapi.py` | `src/ttsFlow/gateways/http_tts_server.py` | `External` | HTTP 진입점 후보 |
| `voiceAI/TTS/protocol.md` | `src/common/protocols/tts_tcp_legacy.md` | `External` | legacy 문서 보존 대상 |

### 3.3 Legacy STT

| 외부 자산 | 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `voiceAI/STT/server.py` | `src/asrFlow/gateways/legacy_tcp_asr_server.py` | `External` | 호환성 레이어 후보 |
| `voiceAI/STT/asr_protocol.md` | `src/common/protocols/asr_tcp_legacy.md` | `External` | legacy 문서 보존 대상 |


## 4. 신규 생성이 필요한 모듈

| 신규 경로 | 상태 | 목적 |
| --- | --- | --- |
| `src/llmFlow/` | `Pending` | 세션, 메모리, provider, gateway |
| `src/ttsFlow/` | `Pending` | TTS processor, vendor, gateway |
| `src/backend/` | `Pending` | flow 조합 orchestration |
| `src/common/utils/` | `Pending` | env 등 공통 유틸 |
| `src/common/errors/` | `Pending` | 상태/에러 코드 정리 |


## 5. 우선순위 제안

### Step 1. `asrFlow` 브리지 제거

- `voiceFlow` 의존이 큰 순서대로 `processor -> vendors -> workers -> sources` 이동

### Step 2. `llmFlow` 최소 골격 생성

- `generate` 한 가지 요청만 우선 지원

### Step 3. `ttsFlow` 최소 골격 생성

- `synthesize` 한 가지 요청만 우선 지원

### Step 4. `backend` 생성

- `pipeline_run` 한 가지 조합 진입점부터 시작


## 6. 현재 결론

- 지금 이관의 실질적 핵심은 `voiceFlow`를 `asrFlow`로 천천히 분해하는 것이다.
- `visionflow`는 이동보다 유지가 맞다.
- `llmFlow`, `ttsFlow`, `backend`는 아직 설계 문서상 목표이지 코드상 현실은 아니다.
