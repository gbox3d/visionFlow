# NeuroFlow Inventory

## 목적

현재 repo에서 실제로 쓰는 entry point, canonical 파일, 남아 있는 compatibility 표면을 빠르게 확인하기 위한 인벤토리다.

## 1. 실제 실행 표면

| 경로 | 상태 | 역할 |
| --- | --- | --- |
| `src/common` | `Canonical` | runtime bus, packet contracts, audio codec, NFCP |
| `src/voiceFlow` | `Canonical` | 오디오 source, ingress server, client, voice launcher |
| `src/asrFlow` | `Canonical` | bootstrap, processor, worker, ASR service, vendor |
| `src/neuroflow/app` | `Canonical` | app 조립과 실행 진입점 |
| `src/visionflow` | `Canonical` | vision runtime과 sample |
| `sample.env` | `Reference` | canonical env 예시 |
| `_forAI` | `Docs` | 구조 메모와 작업 기록 |

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

## 3. App Entry Point Map

| 모듈 | 역할 | 비고 |
| --- | --- | --- |
| `neuroflow.app.asr_server` | server composition root | `voiceFlow` + `asrFlow` 조립 |
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
| `src/neuroflow/app/asr_chunk_realtime.py` | chunk UI |
| `src/neuroflow/app/audiomi_asr_chunk_realtime.py` | accumulate UI |
| `src/neuroflow/app/asr_stream_realtime.py` | stream UI |
| `src/neuroflow/app/asr_model_catalog.py` | model/backend 분류 |
| `src/neuroflow/app/asr_ui_common.py` | UI 공용 gauge/bridge |

## 5. Model Experiment Split

| 구분 | 모델/백엔드 |
| --- | --- |
| `chunk` | Whisper `ct2`, `hf_generate`, `hf_pipeline` |
| `chunk` | Qwen ASR `qwen_transformers` |
| `stream` | Qwen ASR `qwen_transformers` |

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
