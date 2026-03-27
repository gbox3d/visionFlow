# NeuroFlow Inventory

## 목적

이 문서는 지금 이 repo 안에 실제로 무엇이 있는지, 그리고 어떤 자산이 다음 구조로 넘어갈 핵심 후보인지 빠르게 확인하기 위한 현재 상태 인벤토리다.

과거 문서처럼 외부 repo 전체를 가정해서 쓰기보다, 우선 `NeuroFlow` 현재 작업 트리 기준으로 정리한다.


## 1. 현재 repo 핵심 자산

| 경로 | 상태 | 역할 |
| --- | --- | --- |
| `src/visionflow` | `Stable` | 카메라, 얼굴/포즈 추론, 렌더링 샘플, `TopicBus` 기반 실시간 파이프라인 |
| `src/voiceFlow` | `Reusable Legacy Core` | 현재 가장 성숙한 ASR/STT 자산 보관소 |
| `src/asrFlow` | `In Progress` | 새 공통 프로토콜 기반 ASR 서버 시작점 |
| `src/common` | `In Progress` | 공통 계약, protocol, packet, job 정의 |
| `scripts/`, `*.spec`, 디바이스 도구 | `Support` | 개발/배포/디바이스 점검 보조 자산 |
| `_forAI` | `Planning` | 문서 기반 구조 정리와 의사결정 기록 |


## 2. 모듈별 관찰

### `src/visionflow`

강점:

- 구조가 가장 정돈되어 있다.
- `pipeline -> source -> worker -> sample` 흐름이 명확하다.
- `TopicBus`가 이미 검증된 패턴으로 보인다.

주의:

- 현재는 시각 파이프라인 중심이다.
- 아직 `backend`나 `common`과 직접 연결되는 계약은 없다.

### `src/voiceFlow`

강점:

- `miso_stt` 기반 ASR processor와 vendor 코드가 있다.
- `audioMi` 입력, 마이크 입력, 누적형 worker, 샘플 UI가 있다.
- 실전적으로 쓸 수 있는 자산이 가장 많다.

주의:

- 이름은 `voiceFlow`지만 실제로는 ASR/STT 쪽 비중이 훨씬 크다.
- UI/샘플/유틸이 섞여 있어 그대로 새 표준 구조로 삼기에는 경계가 흐리다.

### `src/asrFlow`

강점:

- 새 `NFCP` 기반 서버 진입점이 이미 있다.
- 마이크 샘플 클라이언트가 있어 end-to-end 확인이 가능하다.

주의:

- `processors/miso_stt_asr.py`는 아직 `voiceFlow` 브리지다.
- `sources`, `workers`, `vendors`는 아직 본격 이동되지 않았다.

### `src/common`

강점:

- 새 서비스 계약의 씨앗이 이미 생겼다.
- `common_protocol.md`와 `nfcp.py`가 함께 있어 문서와 코드가 연결된다.

주의:

- `utils`, `errors`, 공통 bus 계층은 아직 비어 있거나 미정이다.


## 3. 지금 바로 가치가 큰 파일

| 파일 | 분류 | 메모 |
| --- | --- | --- |
| `src/visionflow/pipeline/bus.py` | `Adopt or Promote Later` | 공통 event bus 후보 |
| `src/voiceFlow/processors/miso_stt_asr.py` | `Adopt` | 실제 ASR core 진입점 |
| `src/voiceFlow/workers/accumulate_asr_worker.py` | `Adopt` | 실시간 누적 ASR 동작 핵심 |
| `src/voiceFlow/sources/audiomi_source.py` | `Adopt` | 외부 PCM 입력 연동 자산 |
| `src/voiceFlow/vendors/miso_stt/*` | `Adopt` | ASR backend 실제 구현 |
| `src/common/contracts/packets.py` | `Keep` | 공통 packet 정의 시작점 |
| `src/common/contracts/job.py` | `Keep` | job 요청/결과 상태 모델 |
| `src/common/protocols/common_protocol.md` | `Keep` | 서비스 간 공통 통신 계약 초안 |
| `src/common/protocols/nfcp.py` | `Keep` | 공통 프로토콜 구현 |
| `src/asrFlow/gateways/tcp_asr_server.py` | `Keep and Expand` | 새 구조의 첫 gateway |
| `src/asrFlow/sample/microphone_client.py` | `Keep` | 프로토콜 smoke test 용도 |


## 4. 아직 비어 있는 축

현재 repo 안에는 아래 디렉터리가 없다.

- `src/llmFlow`
- `src/ttsFlow`
- `src/backend`

즉, 프로젝트 비전은 멀티모달 전체이지만 실제 구현 진척은 아직 `vision + ASR/common` 쪽에 몰려 있다.


## 5. 외부 참고 자산

현재 repo 밖에 있지만 향후 씨앗이 될 가능성이 높은 외부 자산은 아래와 같다.

- `voiceAI/STT`
  - legacy TCP STT 서버와 프로토콜
- `voiceAI/TTS`
  - legacy TCP TTS 서버와 엔진 실험 코드
- `miso_kiosk/llm`
  - 대화 메모리, 프롬프트, LLM provider 후보 자산

이 자산들은 지금 인벤토리의 핵심이 아니라, 필요 시 가져올 후보군으로 보는 편이 맞다.


## 6. 지금 시점 분류

### `Adopt`

- `visionflow`의 파이프라인 패턴
- `voiceFlow`의 ASR processor, vendor, source, worker
- `common`의 NFCP/job/packet 정의
- `asrFlow`의 gateway/client 뼈대

### `Adapt`

- `voiceFlow`의 env/util 구조
- `voiceFlow` 샘플 코드 중 UI 의존이 적은 부분
- `TopicBus`의 공통 계층 승격 여부

### `Reference`

- 루트 `README.md`
- 디바이스 관리 UI와 각종 배포 스크립트
- 외부 repo의 legacy STT/TTS/LLM 자산

### `Exclude for Core`

- PyInstaller spec
- UI 전용 편의 도구
- 대용량 테스트 미디어
- 앱 조립 중심 코드


## 7. 현재 결론

- 이 프로젝트의 진짜 출발점은 `visionflow`와 `voiceFlow`다.
- 새 표준 구조의 출발점은 `common`과 `asrFlow`다.
- 앞으로 가장 먼저 정리해야 할 것은 `voiceFlow -> asrFlow` 실제 이관 범위 확정이다.
