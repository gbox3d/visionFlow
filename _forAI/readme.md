# _forAI Guide

## 한 줄 요약

`NeuroFlow`는 현재 `visionflow`와 `voiceFlow`에서 검증된 자산을 바탕으로, `Vision + ASR/STT + LLM + TTS`를 느슨하게 조합할 수 있는 멀티모달 파이프라인 라이브러리로 재편되는 중이다.

즉, 이 프로젝트는 단일 앱보다는 각 파이프라인을 쉽게 갈아끼우고 묶을 수 있게 만드는 공용 라이브러리/런타임 쪽에 가깝다.


## 이 폴더의 목적

- 현재 코드 상태를 빠르게 파악하기 위한 작업 문서
- 미래 구조와 현재 구현 상태를 분리해서 정리
- 다음 리팩터링 순서를 합의하기 위한 기준점 제공


## 현재 프로젝트 상태 요약

- `src/visionflow`
  - 가장 완성도가 높은 축
  - 카메라, 얼굴/포즈 추론, `TopicBus` 기반 실시간 파이프라인이 이미 동작함
- `src/voiceFlow`
  - 실질적인 ASR/STT 핵심 자산 보관소
  - `miso_stt` 벤더, 입력 소스, 워커, 샘플이 모여 있음
- `src/common`
  - 새 공통 계약 계층이 생김
  - `NFCP`, `JobRequest/JobResult`, 공통 packet 정의가 들어가 있음
- `src/asrFlow`
  - 새 구조로 옮겨가는 첫 단계
  - NFCP 기반 TCP ASR 서버와 마이크 클라이언트가 있음
  - 다만 processor core는 아직 `voiceFlow`를 브리지로 재사용 중
- `src/llmFlow`, `src/ttsFlow`, `src/backend`
  - 목표 구조에는 포함되지만 아직 이 repo 안에서는 본격 구현 전 단계


## 문서 읽기 순서

1. `readme.md`
2. `inventory.md`
3. `plan.md`
4. `migration_map.md`
5. `memo.md`
6. `dev_log.md`


## 각 문서 역할

- `inventory.md`
  - 현재 repo 안에 실제로 무엇이 있는지 정리
- `plan.md`
  - 이 프로젝트를 어떤 구조로 키울지 정리
- `migration_map.md`
  - 현재 자산을 목표 구조로 어떻게 옮길지 정리
- `memo.md`
  - 열린 질문, 판단 기준, 짧은 메모
- `dev_log.md`
  - `_forAI` 정리 작업 로그


## 참고

- 루트 `README.md`는 아직 전체 프로젝트 소개라기보다 `VisionFlow` 중심 문서에 가깝다.
- 따라서 현재 제품 비전은 루트 문서보다 이 `_forAI` 문서 묶음이 더 잘 설명한다.
