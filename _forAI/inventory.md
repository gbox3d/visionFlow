# NeuroFlow Inventory

## 목차

1. [목적](#목적)
2. [최근 변경 구현사항](#최근-변경-구현사항)
3. [1. 실제 실행 표면](#1-실제-실행-표면)
4. [2. Public Commands](#2-public-commands)
5. [3. App Entry Point Map](#3-app-entry-point-map)
6. [4. Canonical Files](#4-canonical-files)
7. [5. Model Experiment Split](#5-model-experiment-split)
8. [6. Live Compatibility Surface](#6-live-compatibility-surface)
9. [7. 설치 메모](#7-설치-메모)
10. [8. 문서 인벤토리](#8-문서-인벤토리)

## 목적

현재 repo에서 실제로 쓰는 entry point, canonical 파일, 남아 있는 compatibility 표면을 빠르게 확인하기 위한 인벤토리다.

## 최근 변경 구현사항

| 구현사항 | 관련 파일 | 이전 대비 차이 |
| --- | --- | --- |
| NFCP server metadata 조회 | `src/common/protocols/nfcp.py`, `src/common/protocols/common_protocol.md`, `src/voiceFlow/gateways/asr_ingress_server.py` | `DESCRIBE` 응답 의존에서 `SERVER_INFO(102)` 별도 조회로 분리 |
| streaming buffer clear command | `src/common/protocols/nfcp.py`, `src/voiceFlow/gateways/asr_ingress_server.py`, `README.md` | 세션 종료 외 수동 초기화 부재에서 `ASR_CLEAR_BUFFER(1003)` 제공으로 변경 |
| bounded audio accumulation | `src/asrFlow/processors/qwen_streaming_asr.py`, `src/neuroflow/app/asr_server.py`, `sample.env` | 전체 배열 누적에서 `ASRFLOW_STREAM_MAX_ACCUM_SAMPLES` 기반 bounded queue로 변경 |
| integrated TTS server | `src/ttsFlow/*`, `src/neuroflow/app/tts_server.py`, `src/neuroflow/app/tts_rest_server.py`, `src/common/contracts/tts_gateway.py` | 별도 TTS 프로젝트에서 본체 `uv`/NFCP/REST 통합으로 변경 |
| version/env/docs sync | `pyproject.toml`, `src/neuroflow/__init__.py`, `sample.env`, `README.md` | `0.2.0`/구 설정 문서에서 `0.2.1`/최신 옵션 반영으로 갱신 |

## 1. 실제 실행 표면

| 경로 | 상태 | 역할 |
| --- | --- | --- |
| `src/common` | `Canonical` | runtime bus, packet contracts, audio codec, NFCP |
| `src/voiceFlow` | `Canonical` | 오디오 source, ingress server, client, voice launcher |
| `src/asrFlow` | `Canonical` | bootstrap, processor, worker, ASR service, vendor |
| `src/ttsFlow` | `Canonical` | 한국어 TTS engine, NFCP/REST TTS gateway, sample client |
| `src/neuroflow/app` | `Canonical` | app 조립과 실행 진입점 |
| `src/visionflow` | `Canonical` | vision runtime과 sample |
| `sample.env` | `Reference` | canonical env 예시 |
| `_forAI` | `Docs` | 구조 메모와 작업 기록 |
| `_forAI/developer_promise_system.md` | `Docs` | Unity/C# 연동 문서 인덱스 |
| `_forAI/unity_nfcp_tcp_guide.md` | `Docs` | ASR/TTS NFCP TCP Unity 예제 |
| `_forAI/unity_tts_rest_guide.md` | `Docs` | TTS REST Unity 예제 |

## 2. Public Commands

| 커맨드 | 상태 | 메모 |
| --- | --- | --- |
| `uv run nf-vision` | `Keep` | vision sample launcher |
| `uv run nf-vision-models-download` | `Keep` | MediaPipe 기본 모델 다운로드 |
| `uv run nf-voice` | `Keep` | voice sample launcher |
| `uv run nf-voice-mic-client` | `Keep` | 마이크 녹음 후 NFCP ASR ingress 요청 |
| `uv run nf-asr-server` | `Keep` | canonical NFCP ASR 서버 |
| `uv run nf-asr-chunk-realtime` | `Keep` | 로컬 마이크 chunk ASR UI |
| `uv run nf-audiomi-asr-chunk-realtime` | `Keep` | audioMi accumulate chunk ASR UI |
| `uv run nf-asr-stream-realtime` | `Keep` | Qwen native streaming UI |
| `uv run nf-vision-camhub` | `Keep` | 카메라 이미지 중계 허브 서버 (NFCP TCP) |
| `uv run nf-vision-camhub-client` | `Keep` | camhub 전송 카메라 클라이언트 (NFCP TCP) |
| `uv run nf-tts-models-download` | `Keep` | Piper fallback용 한국어 ONNX TTS 모델 다운로드 |
| `uv run nf-tts-server` | `Keep` | canonical NFCP TTS 서버 |
| `uv run nf-tts-rest-server` | `Keep` | 외부 앱/키오스크용 TTS REST API 서버 |
| `uv run nf-tts-client "안녕하세요"` | `Keep` | TTS_SYNTHESIZE 테스트 클라이언트 |

## 3. App Entry Point Map

| 모듈 | 역할 | 비고 |
| --- | --- | --- |
| `neuroflow.app.asr_server` | server composition root | `voiceFlow` + `asrFlow` 조립 |
| `neuroflow.app.tts_server` | server composition root | `ttsFlow` 조립 |
| `neuroflow.app.tts_rest_server` | REST composition root | `ttsFlow` REST gateway 조립 |
| `neuroflow.app.asr_chunk_realtime` | 로컬 마이크 chunk UI | `AsrWorker` 사용 |
| `neuroflow.app.audiomi_asr_chunk_realtime` | audioMi accumulate UI | `AccumulateAsrWorker` 사용 |
| `neuroflow.app.asr_stream_realtime` | native stream UI | `StreamingAsrWorker` 사용 |
| `voiceFlow.main` | voice launcher | 위 app과 source sample 호출 |

## 4. Canonical Files

### 공용 기반

| 파일 | 역할 |
| --- | --- |
| `src/common/runtime/bus.py` | canonical `TopicBus` |
| `src/common/contracts/packets.py` | `AudioPacket`, `AsrResultPacket` |
| `src/common/contracts/asr_gateway.py` | ASR request/response contract |
| `src/common/contracts/tts_gateway.py` | TTS request/response contract |
| `src/common/runtime/audio_codec.py` | transport 오디오 encode/decode |
| `src/common/protocols/nfcp.py` | NFCP protocol 구현 |

### voice ingress

| 파일 | 역할 |
| --- | --- |
| `src/voiceFlow/gateways/asr_ingress_server.py` | canonical NFCP ingress server |
| `src/voiceFlow/sources/microphone_source.py` | microphone source |
| `src/voiceFlow/sources/audiomi_source.py` | audioMi source |
| `src/voiceFlow/sample/microphone_client.py` | NFCP batch client |
| `src/voiceFlow/main.py` | voice launcher |

### ASR core

| 파일 | 역할 |
| --- | --- |
| `src/asrFlow/bootstrap.py` | env load / processor bootstrap |
| `src/asrFlow/processors/miso_stt_asr.py` | chunk / batch canonical processor |
| `src/asrFlow/processors/qwen_streaming_asr.py` | Qwen native streaming processor |
| `src/asrFlow/workers/asr_worker.py` | chunk worker |
| `src/asrFlow/workers/accumulate_asr_worker.py` | accumulate worker |
| `src/asrFlow/workers/streaming_asr_worker.py` | native streaming worker |
| `src/asrFlow/services/nfcp_asr_handler.py` | decoded request -> processor 호출 |
| `src/asrFlow/vendors/whisper/*` | Whisper runtime |
| `src/asrFlow/vendors/qwen_asr/*` | Qwen transformers runtime |

### app layer

| 파일 | 역할 |
| --- | --- |
| `src/neuroflow/app/asr_server.py` | server composition |
| `src/neuroflow/app/tts_server.py` | NFCP TTS server composition |
| `src/neuroflow/app/tts_rest_server.py` | REST TTS server composition |
| `src/neuroflow/app/asr_chunk_realtime.py` | chunk UI |
| `src/neuroflow/app/audiomi_asr_chunk_realtime.py` | accumulate UI |
| `src/neuroflow/app/asr_stream_realtime.py` | stream UI |
| `src/neuroflow/app/asr_model_catalog.py` | model/backend 분류 |
| `src/neuroflow/app/asr_ui_common.py` | UI 공용 gauge/bridge |

### vision camhub

| 파일 | 역할 |
| --- | --- |
| `src/visionflow/camhub/hub.py` | `FrameHub` 인메모리 프레임 저장소 |
| `src/visionflow/camhub/server.py` | NFCP TCP 카메라 허브 서버 |
| `src/visionflow/camhub/main.py` | `nf-vision-camhub` entry point |
| `src/visionflow/camhub/camera_client.py` | 로컬 카메라 → hub 전송 클라이언트 |

### TTS core

| 파일 | 역할 |
| --- | --- |
| `src/ttsFlow/engines/piper_ko.py` | Korean Piper ONNX engine |
| `src/ttsFlow/engines/speecht5_ko.py` | Korean SpeechT5 quality engine |
| `src/ttsFlow/engines/stub.py` | smoke test tone engine |
| `src/ttsFlow/services/nfcp_tts_handler.py` | TTS request -> engine 호출 |
| `src/ttsFlow/gateways/tts_server.py` | canonical NFCP TTS server |
| `src/ttsFlow/gateways/rest_tts_server.py` | REST TTS API gateway |
| `src/ttsFlow/sample/nfcp_tts_client.py` | NFCP TTS smoke client |
| `src/common/tools/download_tts_models.py` | TTS model download utility |

## 5. Model Experiment Split

| 구분 | 모델/백엔드 |
| --- | --- |
| `chunk` | Whisper `ct2`, `hf_generate`, `hf_pipeline` |
| `chunk` | Qwen ASR `qwen_transformers` |
| `stream` | Qwen ASR `qwen_transformers` |
| `tts` | SpeechT5 Korean CPU quality path + Piper ONNX fallback |

## 6. Live Compatibility Surface

현재 소스 기준으로 문서화할 가치가 남아 있는 compatibility surface는 거의 없다.

| 파일 | 상태 | 설명 |
| --- | --- | --- |
| `src/visionflow/pipeline/bus.py` | `Bridge` | `common.runtime.bus` re-export |
| `src/asrFlow/utils/audio.py` | `Bridge` | `common.runtime.audio_codec` re-export |

나머지 예전 `main.py`, device UI/spec, `voiceFlow` 내부 STT 구현물, legacy gateway/sample shim은 소스 파일 기준으로 제거된 상태다.

## 7. 설치 메모

| 용도 | 명령 |
| --- | --- |
| 기본 개발 및 stream 예제 포함 | `uv sync` |

메모:

- 현재 native stream 경로는 기본 dependency에 포함돼 있다.
- stream 예제와 NFCP streaming path는 모두 `Qwen ASR + qwen_transformers` 기준이다.
- TTS 경로는 `NF_TTS_ENGINE=speecht5-ko`, `NF_TTS_DEVICE=cpu` 기본값이다. Piper fallback 모델 파일만 `uv run nf-tts-models-download`로 준비한다.
- REST TTS gateway는 기본 `NF_TTS_REST_PORT=26121`을 사용한다.

## 8. 문서 인벤토리

| 문서 | 역할 |
| --- | --- |
| `README.md` | 사용자/운영자용 실행 안내 |
| `_forAI/readme.md` | AI와 개발자가 먼저 보는 문서 인덱스 |
| `_forAI/developer_promise_system.md` | Unity/C# 연동 문서 인덱스 |
| `_forAI/unity_nfcp_tcp_guide.md` | ASR/TTS NFCP TCP Unity 예제 |
| `_forAI/unity_tts_rest_guide.md` | TTS REST Unity 예제 |
| `_forAI/architecture.md` | 모듈 경계와 런타임 흐름 |
| `_forAI/inventory.md` | 현재 실행 표면과 canonical 파일 |
| `_forAI/memo.md` | 고정 판단, 열린 질문, 다음 작업 |
| `_forAI/dev_log.md` | 실제 작업 기록 |
| `_forAI/plan.md` | 과거 확장 계획과 장기 방향 |
| `_forAI/migration_map.md` | migration 기록과 compatibility 표면 |
| `_forAI/ASR_models_research.md` | ASR 모델 서베이 요약 |
