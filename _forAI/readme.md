# _forAI Guide

## 목차

1. [한 줄 요약](#한-줄-요약)
2. [먼저 볼 문서](#먼저-볼-문서)
3. [현재 canonical 구조](#현재-canonical-구조)
4. [현재 public entry points](#현재-public-entry-points)
   - [visionFlow](#visionflow)
   - [voiceFlow](#voiceflow)
   - [asrFlow](#asrflow)
5. [지금 핵심 경계](#지금-핵심-경계)
6. [문서별 역할](#문서별-역할)
7. [유지 규칙](#유지-규칙)
8. [주의](#주의)

---

## 한 줄 요약

현재 `NeuroFlow`는 아래 5축으로 보는 것이 가장 정확하다.

- `common`
  - 공용 runtime, contracts, protocol
- `voiceFlow`
  - 오디오 ingress, source, client
- `asrFlow`
  - bootstrap, processor, worker, vendor, ASR service
- `neuroflow.app`
  - composition root, app entry point
- `visionflow`
  - vision runtime, camhub, sample

## 먼저 볼 문서

1. [architecture.md](./architecture.md)
2. [inventory.md](./inventory.md)
3. [memo.md](./memo.md)
4. [dev_log.md](./dev_log.md)

## 현재 canonical 구조

```text
common
  -> runtime / contracts / protocols

voiceFlow
  -> sources / gateways / sample client

asrFlow
  -> bootstrap / processors / workers / services / vendors

neuroflow.app
  -> asr_server / asr_chunk_realtime / audiomi_asr_chunk_realtime / asr_stream_realtime

visionflow
  -> camera / mediapipe / camhub / samples
```

## 현재 public entry points

### visionFlow

| 커맨드 | 역할 |
| --- | --- |
| `uv run nf-vision` | vision sample launcher |
| `uv run nf-vision-models-download` | MediaPipe 기본 모델 다운로드 |
| `uv run nf-vision-camhub` | 카메라 이미지 중계 허브 서버 (FastAPI) |
| `uv run nf-vision-camhub-client` | 로컬 카메라 → camhub 전송 클라이언트 |

### voiceFlow

| 커맨드 | 역할 |
| --- | --- |
| `uv run nf-voice` | voice sample launcher |
| `uv run nf-voice-mic-client` | 마이크 녹음 후 NFCP ASR ingress 요청 |

### asrFlow

| 커맨드 | 역할 |
| --- | --- |
| `uv run nf-asr-server` | canonical NFCP ASR 서버 |
| `uv run nf-asr-chunk-realtime` | 로컬 마이크 chunk ASR UI |
| `uv run nf-audiomi-asr-chunk-realtime` | audioMi accumulate chunk ASR UI |
| `uv run nf-asr-stream-realtime` | Qwen native streaming UI |

## 지금 핵심 경계

- `voiceFlow`는 텍스트를 만드는 코어가 아니라 ingress/edge를 소유한다.
- `asrFlow`는 source/client/server를 소유하지 않고 ASR 실행만 맡는다.
- `neuroflow.app`만 `voiceFlow`와 `asrFlow`를 조립한다.
- `chunk` 예제와 `stream` 예제는 분리됐다.
- 현재 native stream 경로는 `Qwen ASR + qwen-asr transformers backend`다.

## 문서별 역할

| 문서 | 역할 |
| --- | --- |
| [architecture.md](./architecture.md) | 구조 도표와 실행 흐름 |
| [inventory.md](./inventory.md) | 현재 repo의 실제 표면과 canonical 파일 인벤토리 |
| [memo.md](./memo.md) | 남은 결정 포인트 |
| [dev_log.md](./dev_log.md) | 실제 정리 작업 기록 |
| [plan.md](./plan.md) | 과거 확장 계획과 장기 메모 |
| [migration_map.md](./migration_map.md) | 과거 migration 기록 |

## 유지 규칙

- 새 entry point가 추가되면 해당 flow 섹션의 테이블에 행을 추가한다.
- 계획이 아닌 참고 정보는 `plan.md`가 아니라 `memo.md`에 둔다.
- 저장소 구조나 실행 명령이 바뀌면 `inventory.md`를 먼저 갱신한다.
- 작업 이력은 날짜를 붙여 `dev_log.md`에만 남긴다.

## 주의

- `plan.md`, `migration_map.md`는 역사적 메모 성격이 강하다.
- 현재 구조 판단은 `architecture.md`와 `inventory.md`를 기준으로 하는 편이 빠르다.
