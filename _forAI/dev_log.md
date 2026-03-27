# Dev Log

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
