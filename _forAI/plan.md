# NeuroFlow Plan

> Note
> 이 문서는 과거 확장 계획과 마이그레이션 메모가 섞여 있다.
> 현재 구조 판단은 `_forAI/architecture.md`, `_forAI/readme.md`, `_forAI/inventory.md`를 우선 기준으로 본다.
> 본문 중 `vLLM`/`stream group` 언급은 역사적 메모이며, 현재 canonical stream 구현은 `Qwen ASR + qwen-asr transformers backend`다.
> `ttsFlow`는 현재 이미 구현되어 있으며, 기본 TTS는 `speecht5-ko` CPU 품질 경로다.

## 목차

1. [1. 미션](#1-미션)
2. [2. 현재 실제 상태](#2-현재-실제-상태)
3. [3. 프로젝트를 어떻게 이해하면 좋은가](#3-프로젝트를-어떻게-이해하면-좋은가)
4. [4. 목표 구조](#4-목표-구조)
5. [5. 모듈 역할](#5-모듈-역할)
6. [6. 설계 원칙](#6-설계-원칙)
7. [7. 핵심 판단](#7-핵심-판단)
8. [8. 추천 이행 순서](#8-추천-이행-순서)
9. [9. 지금 당장 보이는 갭](#9-지금-당장-보이는-갭)
10. [10. 바로 다음 액션 추천](#10-바로-다음-액션-추천)
11. [11. 당분간 범위 밖](#11-당분간-범위-밖)
12. [12. 2026-03-27 추가 계획: `Streaming ASR` + `Qwen3-ASR` + `MiracleASRServer`](#12-2026-03-27-추가-계획-streaming-asr--qwen3-asr--miracleasrserver)

## 1. 미션

`NeuroFlow`의 목표는 단일 앱을 만드는 것이 아니라, 아래 파이프라인을 쉽게 조합할 수 있는 멀티모달 라이브러리와 런타임을 만드는 것이다.

- `Vision`
- `ASR/STT`
- `LLM`
- `TTS`

핵심은 모델 하나를 잘 돌리는 것보다, 입력과 출력, 서비스 경계를 명확히 나누고 파이프라인을 쉽게 갈아끼울 수 있게 만드는 데 있다.


## 2. 현재 실제 상태

2026-03-31 기준으로 이 repo는 아래처럼 보는 것이 가장 정확하다.

### 이미 잘 돌아가는 축

- `src/visionflow`
  - 카메라, 얼굴/포즈 추론, 렌더링 샘플이 이미 정리되어 있음
  - 현재는 `common.runtime.bus.TopicBus`를 기준으로 동작하고, `visionflow.pipeline.bus`는 compatibility re-export 경로임
- `src/voiceFlow`
  - 현재 canonical 역할은 audio ingress/source + device utility + microphone network edge + voice-side app 쪽이다
  - `sources`, `gateways/microphone_server.py`, `sample/microphone_client.py`, `sample/asr_realtime.py`, `sample/audiomi_asr_realtime.py`, `main.py`는 핵심 자산이다
  - `processors` / `workers` / `pipeline.packet` / 일부 util 경로는 compatibility shim으로 남아 있음

### 새 구조로 옮기기 시작한 축

- `src/common`
  - `NFCP`, `JobRequest/JobResult`, 공통 packet 정의가 있음
  - 여기에 `runtime.bus`, 공용 packet contracts, 공용 audio codec이 이미 승격됨
- `src/asrFlow`
  - `processors`, `workers`, `vendors/whisper`의 canonical 소유자가 되었음
  - 예전 NFCP server/client/UI sample 경로는 현재 `voiceFlow` canonical 경로를 가리키는 compatibility shim임
  - 다만 `contracts/`, `sources/`는 아직 뼈대만 있는 상태임

### 아직 비어 있는 축

- `src/llmFlow`
- `src/ttsFlow`
- `src/backend`


## 3. 프로젝트를 어떻게 이해하면 좋은가

겉으로는 `VisionFlow + voiceFlow`처럼 보이지만, 실제 방향은 아래가 더 정확하다.

```text
sources -> flow modules -> gateway/orchestrator -> app/client/avatar
```

조금 더 구체적으로 쓰면:

```text
microphone/audio stream -> voiceFlow(edge/source) -> asrFlow -> llmFlow -> ttsFlow
camera/frame stream     -> visionflow -----------------------^
```

즉:

- `visionflow`는 병렬 입력 축
- `asrFlow/llmFlow/ttsFlow`는 대화형 음성 파이프라인 축
- `backend`는 여러 flow를 한 요청 단위로 묶는 조합 축


## 4. 목표 구조

```text
NeuroFlow/
  src/
    common/
      contracts/
      protocols/
      utils/
      errors/
    asrFlow/
      contracts/
      processors/
      workers/
      vendors/
      utils/
    llmFlow/
      session/
      memory/
      prompts/
      providers/
      gateways/
      sample/
    ttsFlow/
      processors/
      workers/
      gateways/
      vendors/
      sample/
    visionflow/
      pipeline/
      sources/
      processors/
      workers/
      sample/
    backend/
      orchestrators/
      pipelines/
      gateways/
      sample/
```


## 5. 모듈 역할

- `common`
  - 서비스 간 공통 계약
  - protocol, packet, job, audio codec, error code
- `voiceFlow`
  - 오디오 입력, 디바이스 유틸, microphone client/server edge, voice-side sample app 계층
- `asrFlow`
  - 오디오를 텍스트로 바꾸는 모델/추론 코어 계층
- `llmFlow`
  - 세션, 메모리, 프롬프트, 응답 생성 계층
- `ttsFlow`
  - 텍스트를 음성으로 바꾸는 계층
- `visionflow`
  - 시각 입력 처리 계층
- `backend`
  - 여러 flow를 한 요청 단위로 묶는 orchestration 계층


## 6. 설계 원칙

### 6.1 조합 가능성이 우선

- 각 flow는 독립 실행 가능해야 한다.
- 각 flow는 교체 가능한 gateway를 가져야 한다.
- 내부 구현은 달라도 외부 계약은 최대한 통일한다.

### 6.2 공통 계약을 먼저 고정

- 서비스 간 통신은 `NFCP`를 기준으로 맞춘다.
- 최소 공통 명령은 `PING`, `DESCRIBE`, 실제 작업 명령으로 시작한다.
- 요청 추적은 `request_id`, 대화 맥락은 `session_id`로 통일한다.

### 6.3 legacy 자산은 버리지 않고 분해

- `voiceFlow`는 지금 당장 폐기할 코드가 아니다.
- 오히려 ingress/source/app 경계를 맡는 canonical 모듈로 계속 유지하는 편이 맞다.
- `visionflow`도 억지로 옮기기보다 유지 후 연동이 낫다.

### 6.4 UI는 코어에서 분리

- 샘플 UI와 디바이스 관리 도구는 남겨도 된다.
- 다만 processor, worker, protocol은 UI 없이도 사용할 수 있어야 한다.


## 7. 핵심 판단

### `ASR`와 `STT`

외부 설명에서는 거의 같은 뜻으로 써도 되지만, 모듈 이름은 `asrFlow`로 통일하는 편이 깔끔하다.

### `voiceFlow`의 위치

현재 `voiceFlow`는 더 이상 canonical ASR core 저장소가 아니다. source/device 축과 microphone network edge, transition shim을 담당하는 보조 축으로 보는 편이 정확하다.

### `visionflow`의 위치

`visionflow`는 `backend` 안으로 억지로 흡수하기보다, 독립 flow로 유지하면서 후속 단계에서 이벤트 계약만 연결하는 편이 안전하다.


## 8. 추천 이행 순서

### Phase 1. `common + asrFlow` 안정화

- `NFCP` 문서와 구현을 현재 voice ingress 서버/클라이언트 기준으로 굳힌다.
- `common.runtime` / `common.contracts`를 기준으로 공용 경로를 굳힌다.
- `voiceFlow` sample/app이 `asrFlow` core를 직접 쓰도록 canonical import 전환을 진행한다.
- `asrFlow/contracts`의 실제 역할을 확정하고, `asrFlow/sources`는 비우거나 제거할지 결정한다.

### Phase 2. `llmFlow` 최소 골격 생성

- provider interface
- session/memory
- 단건 `generate` gateway
- 샘플 prompt 자산 정리

### Phase 3. `ttsFlow` 최소 골격 생성

현재 완료된 축이다.

- `ttsFlow` engine/service/gateway/sample client 분리
- NFCP `TTS_SYNTHESIZE(3001)` 서버 추가
- 키오스크 연동용 REST gateway 추가
- 기본 TTS는 `speecht5-ko` CPU 품질 경로, Piper ONNX는 fallback

### Phase 4. `backend` MVP

- `pipeline_run` 요청 1개로 `ASR -> LLM -> TTS` 체인 실행
- `session_id` 유지
- 결과 및 오류 표준화

### Phase 5. `visionflow` 연동

- vision 결과를 직접 `LLM` 컨텍스트에 넣을지
- 아니면 `backend` 이벤트 컨텍스트로만 유지할지 결정


## 9. 지금 당장 보이는 갭

- 루트 `README.md`와 `pyproject.toml`은 현재 public entrypoint 기준으로 많이 정리됐지만, 배포 spec과 일부 legacy 문서는 아직 따라오지 못했다.
- `asrFlow`는 canonical core가 되었지만 `contracts`, `sources`는 아직 비어 있다.
- 제품 경로 일부와 배포 자산은 아직 compatibility shim에 기대고 있다.
- `llmFlow`, `backend`는 목표 문서 대비 실제 코드가 없다.
- `ttsFlow`는 구현됐지만 pipeline orchestration과 LLM 연동은 아직 없다.
- 네이밍이 `visionflow`, `voiceFlow`, `asrFlow`로 섞여 있어 장기적으로는 정리가 필요하다.


## 10. 바로 다음 액션 추천

1. `asrFlow/contracts`와 `asrFlow/sources`의 방향을 확정한다.
2. 배포 spec과 legacy 문서가 새 canonical 경로를 직접 쓰게 정리한다.
3. compatibility shim 제거 기준을 제품 경로 중심으로 고정한다.
4. 그 다음에 `Qwen3-ASR`와 streaming session 설계를 재개한다.
5. 마지막에 `llmFlow`, `backend` 골격과 `ASR -> LLM -> TTS` orchestration으로 넘어간다.


## 11. 당분간 범위 밖

- 키오스크 UI 전체 설계
- Unity/Unreal 전용 adapter 세부 구현
- 인증, TLS, 운영 인프라 고도화
- 대규모 streaming 최적화와 분산 처리

지금 우선순위는 어디까지나 멀티모달 파이프라인 라이브러리의 중심 구조를 먼저 세우는 것이다.


## 12. 2026-03-27 추가 계획: `Streaming ASR` + `Qwen3-ASR` + `MiracleASRServer`

이 섹션은 기존 멀티모달 전체 계획 위에, 지금 바로 필요한 `ASR` 확장만 별도 작업선으로 정리한 추가 계획이다.


### 12.1 왜 별도 계획이 필요한가

- `MiracleASRServer`는 이미 형제 repo의 `NeuroFlow/src`를 black-box처럼 참조하는 구조다.
- 하지만 현재 `NeuroFlow`의 ASR core는 여전히 사실상 `Whisper` 전용 경로에 가깝다.
- canonical processor와 worker는 `asrFlow`로 이동했지만, `AsrEngine` / `StreamingSession` 계약은 아직 없다.
- `voiceFlow.vendors.miso_stt.*`는 현재 대부분 `asrFlow.vendors.whisper.*`를 가리키는 thin compatibility 경로로 정리됐다.
- `MiracleASRServer`의 streaming도 현재는 "청크 누적 -> 전체 버퍼 재추론" 방식이라, 모델 native streaming과는 다르다.

즉, 이번 요구는 단순히 `ASRFLOW_STT_MODEL=Qwen/Qwen3-ASR-1.7B`처럼 모델명만 바꾸는 일이 아니라:

- `Whisper 전용 구현`을 `ASR 엔진 추상화`로 분리하고
- `Qwen3-ASR` vendor를 새로 붙이고
- streaming 책임을 `NeuroFlow` 쪽으로 당겨와
- `MiracleASRServer`는 프로토콜 gateway 역할만 하게 만드는 작업이다.


### 12.2 현재 코드 기준 진단

#### `NeuroFlow`

- `src/asrFlow/processors/miso_stt_asr.py`
  - 현재 canonical `MisoSttAsrProcessor` 구현을 가진다.
- `src/voiceFlow/processors/miso_stt_asr.py`
  - 현재는 `asrFlow.processors.miso_stt_asr`를 가리키는 compatibility re-export다.
- `src/asrFlow/workers/*`
  - 현재 canonical worker 구현을 가진다.
- `src/asrFlow/vendors/whisper/*`
  - Whisper canonical vendor 경로가 이미 분리됐다.
- `src/voiceFlow/vendors/miso_stt/transcriber.py`
  - `asrFlow.vendors.whisper.transcriber`를 가리키는 legacy compatibility 경로다.
- `src/voiceFlow/vendors/miso_stt/backends/hf_generate.py`
  - 여전히 `WhisperProcessor`, `WhisperForConditionalGeneration`에 직접 결합된 legacy 경로다.

#### `MiracleASRServer`

- `neuroflow_adapter.py`
  - 현재는 `NeuroFlow`를 한 파일에서 black-box처럼 묶는 방향이 맞다.
- `server.py`
  - legacy TCP framing과 streaming 세션 관리는 이미 있다.
  - 다만 streaming 핵심은 서버가 직접 버퍼를 누적하고 일정 step마다 같은 processor를 다시 호출하는 구조다.

결론:

- `MiracleASRServer` 방향은 맞다.
- 진짜 바꿔야 하는 축은 `NeuroFlow` 내부 ASR 엔진 경계다.


### 12.3 `Qwen3-ASR` 도입 기준

2026-01-29 공개된 공식 `Qwen3-ASR` 라인업과 2026-01-30 기준 `qwen-asr` 패키지 기준으로 보면:

- 우선 공식 공개 ASR 모델은 `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B` 두 가지다.
- 별도 timestamp 용으로 `Qwen/Qwen3-ForcedAligner-0.6B`가 있다.
- 공식 패키지는 `transformers` backend와 `vLLM` backend를 함께 제공한다.
- 문서상 streaming은 지원되지만, 빠른 inference와 streaming 쪽은 `vLLM` backend 중심으로 보는 편이 맞다.

따라서 모델 선택 정책은 아래처럼 잡는다.

- 기본값: `Qwen/Qwen3-ASR-1.7B`
- 1차 공식 지원 선택지: `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B`
- 확장 선택지: 임의의 `Qwen/Qwen3-ASR-*` HF ID 또는 로컬 checkpoint 경로
- optional 부가 모델: `Qwen/Qwen3-ForcedAligner-0.6B`

즉, UI/설정에서는 `xx` 계열 선택을 열어 두되, 처음부터 검증 대상으로 삼는 카탈로그는 위 세 모델로 제한하는 편이 안전하다.


### 12.4 목표 구조

```text
NeuroFlow/
  src/
    asrFlow/
      contracts/
        asr_engine.py
        streaming_session.py
      registry/
        model_catalog.py
      processors/
        asr_engine_processor.py
      vendors/
        whisper/
          ...
        qwen_asr/
          runtime.py
          streaming.py
          config.py
      gateways/   # compatibility shell 또는 thin bridge
        tcp_asr_server.py
      sample/
        ...

voiceFlow/
  gateways/
    microphone_server.py

MiracleASRServer/
  neuroflow_adapter.py
  server.py
```

핵심은 아래 한 줄이다.

- `MiracleASRServer`는 protocol gateway
- `NeuroFlow`는 model/runtime/streaming engine


### 12.5 책임 분리 원칙

#### `NeuroFlow`가 가져갈 책임

- 모델 family 선택
- 모델 로딩/워밍업
- single-shot transcription
- streaming session 상태 관리
- partial/final 결과 정책
- resample 이후 추론 정책
- `Whisper`와 `Qwen3-ASR` 차이를 숨기는 공통 인터페이스

#### `MiracleASRServer`가 유지할 책임

- little-endian TCP framing
- request code 처리
- 연결/timeout/backpressure
- legacy 프로토콜 호환
- `NeuroFlow` black-box 호출

#### 금지할 것

- `MiracleASRServer` 안에서 `Qwen` vendor import
- `MiracleASRServer` 안에서 모델별 조건 분기 추가
- `MiracleASRServer` 안에서 streaming 추론 정책을 별도 진화시키는 것


### 12.6 설정 체계 재정리

현재 `ASRFLOW_STT_BACKEND`, `ASRFLOW_STT_MODEL` 식 설정은 `Whisper`에는 맞지만, `Qwen3-ASR`까지 포함하면 의미가 흐려진다.

추가 계획 기준의 canonical 설정은 아래처럼 재정리하는 편이 낫다.

```env
ASRFLOW_VENDOR=qwen_asr
ASRFLOW_RUNTIME=transformers
ASRFLOW_MODEL_PROFILE=qwen3_asr_1_7b
ASRFLOW_MODEL_ID=Qwen/Qwen3-ASR-1.7B
ASRFLOW_MODEL_PATH=
ASRFLOW_STREAMING_MODE=accumulate_window
ASRFLOW_ALIGNER_MODEL_ID=
```

호환성 원칙:

- 기존 `ASRFLOW_STT_*` 환경 변수는 1차에서는 backward-compatible alias로 유지
- `Whisper` 경로는 `vendor=whisper`로 정리
- `Qwen3-ASR` 경로는 `vendor=qwen_asr`로 분리


### 12.7 단계별 추가 이행 계획

#### Phase A. `Whisper 전용 구조`를 `ASR 엔진 구조`로 분리

- `MisoSttAsrProcessor` 중심 구조를 범용 `AsrEngine` 인터페이스로 분리
- 현재 `Whisper` 구현은 `vendors/whisper` 아래로 명시적 이동
- `asrFlow/processors`는 vendor 이름을 모르게 정리
- `voiceFlow -> asrFlow` bridge 제거 범위를 이번 작업과 연결

완료 기준:

- processor가 더 이상 `WhisperTranscriber`라는 이름에 종속되지 않는다.

#### Phase B. `Qwen3-ASR` vendor 1차 도입

- `qwen-asr` 패키지 기반 `transformers` runtime 먼저 붙인다.
- single-shot transcription부터 먼저 성공시킨다.
- 모델 카탈로그에 아래 프로파일을 추가한다.
  - `qwen3_asr_1_7b`
  - `qwen3_asr_0_6b`
  - `qwen3_forced_aligner_0_6b`
- freeform HF ID / local path override도 허용한다.

완료 기준:

- `NeuroFlow` 단독으로 `Qwen/Qwen3-ASR-1.7B` 단건 추론이 동작한다.

#### Phase C. streaming session을 `NeuroFlow` 내부로 이동

- 현재 `MiracleASRServer`에 있는 누적 버퍼/step/warmup/max_window 로직을 `NeuroFlow` session 객체로 옮긴다.
- `MiracleASRServer`는 `start -> push_chunk -> end`만 호출하게 만든다.
- partial semantics는 지금처럼 "delta가 아니라 현재까지의 전체 누적 텍스트"를 유지한다.

1차 streaming 모드는 두 층으로 나눈다.

- `accumulate_window`
  - 현재 구조를 engine 내부로 옮긴 호환 모드
  - `transformers` runtime에서도 바로 사용 가능
- `native`
  - 모델 또는 runtime이 제공하는 true streaming 모드
  - `Qwen3-ASR + vLLM` 실험이 들어가는 2차 목표

완료 기준:

- `MiracleASRServer`는 더 이상 자체 streaming 추론 정책을 직접 갖지 않는다.

#### Phase D. `MiracleASRServer` black-box 통합 안정화

- `neuroflow_adapter.py`를 유일한 binding point로 유지한다.
- adapter가 `build_processor_from_env()` 수준을 넘어 `build_engine_from_env()`, `build_streaming_session()`까지 제공하게 한다.
- `server.py`는 protocol code와 세션 lifecycle만 담당한다.
- legacy `TRANSCRIBE`와 streaming request code는 깨지지 않게 유지한다.

완료 기준:

- `MiracleASRServer` 쪽 변경 없이도 `NeuroFlow` 내부 vendor를 바꾸면 모델 계열을 교체할 수 있다.

#### Phase E. `NFCP`와 legacy Miracle streaming을 내부적으로 수렴

- `common.protocols.nfcp`의 `ASR_TRANSCRIBE_STREAM`를 실제 구현으로 연결한다.
- `voiceFlow/gateways/microphone_server.py`와 `MiracleASRServer/server.py`가 같은 engine/session 코어를 재사용하게 만든다.
- 내부 코어는 하나, 외부 gateway만 둘로 유지한다.

완료 기준:

- `NFCP voice ingress server`와 `MiracleASRServer`가 모델/streaming core를 공유한다.


### 12.8 바로 다음 액션

1. `NeuroFlow` 안에 `AsrEngine` / `StreamingSession` 계약 초안을 만든다.
2. 현재 `Whisper` 구현을 vendor 단위로 분리할 리팩터링 지점을 확정한다.
3. `Qwen3-ASR`용 model catalog와 env naming을 먼저 문서화한다.
4. `MiracleASRServer`의 accumulate streaming 로직을 `NeuroFlow` session으로 옮길 이동 단위를 정한다.
5. 그 다음에 `qwen-asr` transformers backend로 `Qwen/Qwen3-ASR-1.7B` 단건 추론을 먼저 붙인다.
6. 마지막에 `vLLM` 기반 native streaming을 2차 실험으로 붙인다.


### 12.9 1차 범위 밖

- forced aligner timestamp API 외부 노출
- 한 프로세스 내 다중 모델 동시 상주시켜 hot-swap
- 분산 inference / multi-worker scheduler
- `MiracleASRServer` 프로토콜 자체 변경

이번 추가 계획의 핵심은 `MiracleASRServer`를 더 똑똑하게 만드는 것이 아니라, `NeuroFlow`를 진짜 black-box ASR 라이브러리로 만드는 것이다.
