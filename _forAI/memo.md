# NeuroFlow Memo

## 현재 이 프로젝트를 보는 프레임

- `visionflow`는 이미 검증된 시각 파이프라인이다.
- `voiceFlow`는 사실상 ASR/STT 코어 저장소다.
- `common`과 `asrFlow`는 새 구조의 시작점이다.
- `llmFlow`, `ttsFlow`, `backend`는 다음 단계에서 실제로 만들어야 한다.


## 핵심 판단

- 이 프로젝트는 앱보다 라이브러리와 서비스 조합 계층에 가깝다.
- `ASR`와 `STT`는 외부 설명에서는 함께 써도 되지만 모듈 이름은 `asrFlow`로 통일하는 편이 낫다.
- `voiceFlow`는 장기 표준 이름이라기보다 이전 자산 저장소로 다루는 편이 맞다.
- `visionflow`는 억지로 합치지 말고 독립 축으로 유지하는 편이 안전하다.


## 열린 질문

- `TopicBus`를 `common`으로 승격할지, 아니면 `visionflow` 내부 패턴으로만 남길지?
- `backend` 1차 버전에서 streaming partial event까지 넣을지, 단건 request/result부터 갈지?
- `llmFlow`의 시작 provider를 로컬 모델로 둘지 API 모델로 둘지?
- 루트 `README.md`와 `pyproject.toml` 설명을 언제 현재 비전에 맞게 갱신할지?


## 다음 작업 추천

- `voiceFlow -> asrFlow` 실제 이동 목록을 파일 단위로 확정
- `llmFlow` 최소 디렉터리와 인터페이스 초안 생성
- 루트 문서와 패키지 설명의 방향 불일치 해소
