# NeuroFlow Plan

## 1. 미션

`NeuroFlow`의 목표는 단일 앱을 만드는 것이 아니라, 아래 파이프라인을 쉽게 조합할 수 있는 멀티모달 라이브러리와 런타임을 만드는 것이다.

- `Vision`
- `ASR/STT`
- `LLM`
- `TTS`

핵심은 모델 하나를 잘 돌리는 것보다, 입력과 출력, 서비스 경계를 명확히 나누고 파이프라인을 쉽게 갈아끼울 수 있게 만드는 데 있다.


## 2. 현재 실제 상태

2026-03-25 기준으로 이 repo는 아래처럼 보는 것이 가장 정확하다.

### 이미 잘 돌아가는 축

- `src/visionflow`
  - 카메라, 얼굴/포즈 추론, 렌더링 샘플이 이미 정리되어 있음
  - `TopicBus` 기반 실시간 파이프라인 패턴이 검증되어 있음
- `src/voiceFlow`
  - 실질적인 ASR/STT 핵심 자산이 가장 많이 모여 있음
  - `miso_stt`, `audioMi`, 마이크 입력, 누적 ASR 워커, 샘플 UI가 있음

### 새 구조로 옮기기 시작한 축

- `src/common`
  - `NFCP`, `JobRequest/JobResult`, 공통 packet 정의가 있음
- `src/asrFlow`
  - NFCP 기반 TCP ASR 서버와 마이크 샘플 클라이언트가 있음
  - 다만 processor core는 아직 `voiceFlow` 브리지 의존이 남아 있음

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
microphone/audio stream -> asrFlow -> llmFlow -> ttsFlow
camera/frame stream     -> visionflow -----------^
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
      sources/
      processors/
      workers/
      gateways/
      vendors/
      sample/
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
  - protocol, packet, job, error code
- `asrFlow`
  - 오디오 입력을 텍스트로 바꾸는 계층
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
- 오히려 `asrFlow`로 추출해야 할 핵심 자산 저장소에 가깝다.
- `visionflow`도 억지로 옮기기보다 유지 후 연동이 낫다.

### 6.4 UI는 코어에서 분리

- 샘플 UI와 디바이스 관리 도구는 남겨도 된다.
- 다만 processor, worker, protocol은 UI 없이도 사용할 수 있어야 한다.


## 7. 핵심 판단

### `ASR`와 `STT`

외부 설명에서는 거의 같은 뜻으로 써도 되지만, 모듈 이름은 `asrFlow`로 통일하는 편이 깔끔하다.

### `voiceFlow`의 위치

현재 `voiceFlow`는 사실상 ASR core 저장소다. 장기적으로는 기능을 더 키우기보다 `asrFlow`로 옮기는 편이 맞다.

### `visionflow`의 위치

`visionflow`는 `backend` 안으로 억지로 흡수하기보다, 독립 flow로 유지하면서 후속 단계에서 이벤트 계약만 연결하는 편이 안전하다.


## 8. 추천 이행 순서

### Phase 1. `common + asrFlow` 안정화

- `NFCP` 문서와 구현을 현재 서버/클라이언트 기준으로 굳힌다.
- `asrFlow`의 `voiceFlow` 브리지 의존성을 줄인다.
- `sources`, `workers`, `vendors`를 `asrFlow`로 옮길 계획을 확정한다.

### Phase 2. `llmFlow` 최소 골격 생성

- provider interface
- session/memory
- 단건 `generate` gateway
- 샘플 prompt 자산 정리

### Phase 3. `ttsFlow` 최소 골격 생성

- processor와 gateway 분리
- 엔진 초기화 코드 캡슐화
- 단건 `synthesize` 요청부터 시작

### Phase 4. `backend` MVP

- `pipeline_run` 요청 1개로 `ASR -> LLM -> TTS` 체인 실행
- `session_id` 유지
- 결과 및 오류 표준화

### Phase 5. `visionflow` 연동

- vision 결과를 직접 `LLM` 컨텍스트에 넣을지
- 아니면 `backend` 이벤트 컨텍스트로만 유지할지 결정


## 9. 지금 당장 보이는 갭

- 루트 `README.md`와 `pyproject.toml` 설명은 아직 비전 중심이다.
- `asrFlow`는 생겼지만 실제 core 상당수는 여전히 `voiceFlow`에 있다.
- `llmFlow`, `ttsFlow`, `backend`는 목표 문서 대비 실제 코드가 없다.
- 네이밍이 `visionflow`, `voiceFlow`, `asrFlow`로 섞여 있어 장기적으로는 정리가 필요하다.


## 10. 바로 다음 액션 추천

1. `voiceFlow -> asrFlow` 실제 이동 목록을 파일 단위로 확정한다.
2. `llmFlow`의 최소 인터페이스와 디렉터리 골격을 만든다.
3. `ttsFlow`도 같은 방식으로 최소 골격을 만든다.
4. 마지막으로 `backend`에서 `pipeline_run` 조합 진입점을 정의한다.
5. 루트 문서와 패키지 메타데이터는 그 다음에 현재 비전에 맞게 맞춘다.


## 11. 당분간 범위 밖

- 키오스크 UI 전체 설계
- Unity/Unreal 전용 adapter 세부 구현
- 인증, TLS, 운영 인프라 고도화
- 대규모 streaming 최적화와 분산 처리

지금 우선순위는 어디까지나 멀티모달 파이프라인 라이브러리의 중심 구조를 먼저 세우는 것이다.
