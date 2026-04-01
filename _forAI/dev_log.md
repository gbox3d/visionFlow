# Dev Log

## 2026-04-01

### ASR chunk / stream 분리 + 문서 정리

- realtime 예제를 `chunk`와 `stream`으로 분리했다.
- 로컬 마이크 chunk UI는 `neuroflow.app.asr_chunk_realtime`로 정리했다.
- audioMi chunk UI는 `neuroflow.app.audiomi_asr_chunk_realtime`로 정리했다.
- native streaming UI는 `neuroflow.app.asr_stream_realtime`로 추가했다.
- Qwen native streaming용 `asrFlow.processors.qwen_streaming_asr`와 `asrFlow.workers.streaming_asr_worker`를 추가했다.
- model/backend 분류를 `neuroflow.app.asr_model_catalog`에서 chunk/native-stream 기준으로 나눴다.
- public script를 `nf-asr-chunk-realtime`, `nf-audiomi-asr-chunk-realtime`, `nf-asr-stream-realtime` 기준으로 정리했다.
- stream 경로는 `qwen_asr` transformers backend 기준으로 기본 dependency 안에서 동작하게 맞췄다.
- `_forAI/architecture.md`를 새로 만들고 Mermaid 구조 도표를 넣었다.
- `_forAI/readme.md`, `_forAI/inventory.md`, `_forAI/memo.md`를 현재 canonical 구조 기준으로 다시 썼다.
- 루트 README, `src/asrFlow/README.md`, `src/voiceFlow/readme.md`, `sample.env`를 현재 구현 기준으로 다시 정리했다.
- 패키지 버전을 `0.2.0`으로 올리고 NFCP `PING`/`DESCRIBE` 응답 메타에 버전을 노출하도록 맞췄다.

### 이번에 확인한 검증 결과

- `forai-scaffold` 재실행 결과 기존 `_forAI` 표준 문서 세트는 그대로 유지됐다.
- `python -m py_compile` 기준 아래 경로 문법 검사를 통과했다.
  - `src/asrFlow/processors/qwen_streaming_asr.py`
  - `src/asrFlow/workers/streaming_asr_worker.py`
  - `src/neuroflow/app/asr_chunk_realtime.py`
  - `src/neuroflow/app/audiomi_asr_chunk_realtime.py`
  - `src/neuroflow/app/asr_stream_realtime.py`
  - `src/neuroflow/app/asr_model_catalog.py`
  - `src/neuroflow/app/asr_ui_common.py`
  - `src/voiceFlow/main.py`
- import smoke 통과:
  - `neuroflow.app.asr_chunk_realtime`
  - `neuroflow.app.audiomi_asr_chunk_realtime`
  - `neuroflow.app.asr_stream_realtime`
  - `asrFlow.processors.qwen_streaming_asr`
  - `asrFlow.workers.streaming_asr_worker`
  - `voiceFlow.main`
- entry point 등록 확인:
  - `nf-asr-server`
  - `nf-asr-chunk-realtime`
  - `nf-audiomi-asr-chunk-realtime`
  - `nf-asr-stream-realtime`
  - `nf-voice`
- `uv lock`와 기본 `uv sync`는 정상 동작했다.

## 2026-03-31

### `voiceFlow` / `asrFlow` 경계 재정리

- `asrFlow`는 네트워크 edge나 UI sample을 소유하지 않는 pure ASR core로 다시 정리했다.
- `src/voiceFlow/sample/asr_realtime.py`, `src/voiceFlow/sample/audiomi_asr_realtime.py`를 voice-side canonical sample app으로 이동했다.
- `src/asrFlow/sample/asr_realtime.py`, `src/asrFlow/sample/audiomi_asr_realtime.py`는 `voiceFlow` 경로를 가리키는 compatibility shim으로 바꿨다.
- `voiceFlow.main` 런처에 realtime ASR app 2개를 추가했다.
- `pyproject.toml` public script를 `nf-voice-asr-realtime`, `nf-voice-audiomi-asr-realtime` 기준으로 정리하고 기존 `nf-asr-*` 표면은 제거했다.
- `sample.env`의 STT canonical 키를 `ASRFLOW_STT_*` 기준으로 갱신했다.
- moved sample app은 transition 동안 legacy `VOICEFLOW_STT_*`도 fallback으로 읽도록 보강했다.
- 루트 `README.md`, `src/voiceFlow/readme.md`, `src/asrFlow/README.md`, `_forAI` 핵심 문서를 현재 경계 기준으로 다시 정리했다.

### 이번에 확인한 검증 결과

- `python -m py_compile` 기준 아래 경로의 문법 검사를 통과했다.
  - `src/voiceFlow/sample/asr_realtime.py`
  - `src/voiceFlow/sample/audiomi_asr_realtime.py`
  - `src/asrFlow/sample/asr_realtime.py`
  - `src/asrFlow/sample/audiomi_asr_realtime.py`
  - `src/voiceFlow/main.py`
  - `src/asrFlow/processors/miso_stt_asr.py`
- import smoke 통과:
  - `voiceFlow.sample.asr_realtime`
  - `voiceFlow.sample.audiomi_asr_realtime`
  - `asrFlow.sample.asr_realtime`
  - `asrFlow.sample.audiomi_asr_realtime`
  - `voiceFlow.main`
- `.venv\\Scripts\\python.exe -m voiceFlow.main`에 `0`을 입력해 런처 메뉴 출력/종료를 확인했다.
- `uv run` 기반 재설치는 당시 실행 중이던 `nf-voice.exe` 프로세스가 스크립트 파일을 잡고 있어서 Windows 파일 교체 단계에서 막혔다.

## 2026-03-25

### `_forAI` 폴더 정리

- `_forAI` 문서를 실제 repo 상태 기준으로 다시 정리했다.
- 비어 있던 `memo.md`, `dev_log.md`를 채웠다.
- `readme.md`를 `_forAI` 인덱스 역할로 바꿨다.
- `inventory.md`는 외부 repo 가정 중심 문서에서 현재 `NeuroFlow` 상태 인벤토리로 바꿨다.
- `plan.md`는 장황한 초기 구상 문서에서 현재 상태와 목표 구조를 함께 보는 실행용 문서로 줄였다.
- `migration_map.md`는 미래 구조만 나열하는 방식에서 `Done / Bridge / Pending / External / Keep` 상태 중심 문서로 바꿨다.

### 이번에 확인한 핵심 사실

- `visionflow`는 가장 안정된 축이다.
- `voiceFlow`는 실제 ASR core 자산 저장소다.
- `asrFlow`는 이미 서버/클라이언트가 있지만 아직 `voiceFlow` 브리지 의존이 남아 있다.
- `llmFlow`, `ttsFlow`, `backend`는 아직 repo 안에 없다.
- 루트 문서와 패키지 설명은 아직 전체 비전보다 `VisionFlow` 중심에 가깝다.

### 후속 추천

- 다음 코드 작업은 `asrFlow` 독립도 높이기에 집중하는 것이 가장 효율적이다.

## 2026-03-30

### `forai-scaffold` 확인

- `forai-scaffold` 스크립트로 `_forAI` 기본 문서 세트 동기화를 확인했다.
- 기존 `_forAI` 문서는 모두 이미 존재했고, 생성/덮어쓰기는 발생하지 않았다.

### 이번에 다시 확인한 핵심 사실

- 루트 `main.py`가 현재 가장 직접적인 통합 데모 진입점이다.
- 초기 점검 시점에는 공용 runtime 승격 이전 상태로 보고 있었고, 음성 코어 소유권도 `voiceFlow` 편중으로 파악했다.
- 루트 `README.md`가 설명하는 `vf-*` 실행 경로와 `pyproject.toml`의 실제 `project.scripts`는 현재 일치하지 않는다.
- 자동 테스트 스위트는 보이지 않고 샘플/수동 실행 중심으로 검증되는 상태다.

### 이번 문서 정리에서 한 일

- `_forAI/readme.md`를 현재 코드 구조 중심으로 다시 요약했다.
- `_forAI/inventory.md`에 실제 실행 표면과 모듈 간 의존 관계를 반영했다.
- `_forAI/memo.md`에 현재 병목과 우선 질문을 갱신했다.
- 사용자 수정 중인 `_forAI/plan.md`는 건드리지 않았다.

### 마이그레이션 점검 + 문서 정리

- `common.runtime.bus.TopicBus`가 canonical bus 경로임을 문서에 반영했다.
- `common.contracts.packets`가 canonical packet 경로임을 문서에 반영했다.
- `asrFlow`가 processor/worker/vendor canonical 소유자이고, `voiceFlow`는 source/util/compatibility shell 성격이라는 점을 문서에 반영했다.
- `_temp_plan.md`를 미래 제안 문서가 아니라 `구현 점검 + 남은 TODO` 문서로 교체했다.
- `src/asrFlow/README.md`, `src/voiceFlow/readme.md`를 현재 코드 기준으로 다시 정리했다.
- `_forAI/migration_map.md`도 상태 매핑 기준에 맞게 함께 갱신했다.

### 이번에 확인한 검증 결과

- `.venv` 기준 import smoke:
  - `common.runtime.bus`
  - `common.contracts.packets`
  - `asrFlow.processors.miso_stt_asr`
  - `asrFlow.workers.asr_worker`
  - `asrFlow.workers.accumulate_asr_worker`
  - `voiceFlow.processors.miso_stt_asr`
  - `voiceFlow.workers.asr_worker`
  - `voiceFlow.workers.accumulate_asr_worker`
  - `visionflow.pipeline.bus`
- `.venv` 기준 `main.py`, `deviceMngUI.py` import smoke 통과
- `python -m py_compile` 기준 `main.py`, `deviceMngUI.py`, `src/asrFlow/gateways/tcp_asr_server.py` 문법 검사 통과

### 남은 일

- `main.py`의 processor/worker import canonicalization
- `asrFlow/contracts`, `asrFlow/sources` 실구현 또는 역할 재정의
- `Qwen3-ASR` 및 streaming session 후속 작업
- 루트 `README.md`와 패키지 메타데이터 정리

### 추가 구조 정리

- root `README.md`를 현재 public entrypoint 기준으로 다시 정리했다.
- vision public surface는 `nf-vision` 런처 + `nf-vision-models-download` utility 중심으로 축소했다.
- `voiceFlow/main.py`를 추가하고 source/device/network 샘플 런처로 정리했다.
- `voiceFlow.sample`에서는 ASR UI 샘플을 제거하고 입력/edge 샘플만 남겼다.
- `voiceFlow.vendors.miso_stt.*`는 대부분 `asrFlow.vendors.whisper.*` thin compatibility re-export로 정리했다.
- `common.runtime.audio_codec`를 추가해 transport용 오디오 encode/decode canonical 경로를 만들었다.
- `asrFlow.utils.audio`는 `common.runtime.audio_codec` compatibility shim으로 바꿨다.
- microphone network edge의 canonical ownership을 `asrFlow`에서 `voiceFlow`로 옮겼다.
- `voiceFlow.gateways.microphone_server`와 `voiceFlow.sample.microphone_client`를 새 canonical 경로로 추가했다.
- `asrFlow.gateways.tcp_asr_server`와 `asrFlow.sample.microphone_client`는 `voiceFlow` 경로를 가리키는 compatibility shim으로 바꿨다.
- `pyproject.toml` public script를 `nf-voice`, `nf-voice-mic-server`, `nf-voice-mic-client`, `nf-asr-realtime`, `nf-asr-audiomi-realtime` 기준으로 정리했다.

### 이번에 확인한 검증 결과

- `forai-scaffold` 재실행 결과 `_forAI` 표준 문서 세트는 이미 존재했고 추가 생성은 없었다.
- `python -m py_compile` 기준 아래 경로의 문법 검사를 통과했다.
  - `src/common/runtime/audio_codec.py`
  - `src/voiceFlow/gateways/microphone_server.py`
  - `src/voiceFlow/sample/microphone_client.py`
  - `src/asrFlow/gateways/tcp_asr_server.py`
  - `src/asrFlow/sample/microphone_client.py`
  - `src/asrFlow/utils/audio.py`
- import smoke 통과:
  - `common.runtime.audio_codec`
  - `voiceFlow.gateways.microphone_server`
  - `voiceFlow.sample.microphone_client`
  - `asrFlow.gateways.tcp_asr_server`
  - `asrFlow.sample.microphone_client`
  - `asrFlow.utils.audio`
- console script smoke 통과:
  - `uv run nf-voice-mic-server --help`
  - `uv run nf-voice-mic-client --help`
  - `uv run nf-voice`
