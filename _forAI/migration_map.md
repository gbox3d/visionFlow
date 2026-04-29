# Migration Map

> Note
> 이 문서는 migration 기록 성격이 강하다.
> 최신 canonical 경로와 public 표면은 `_forAI/architecture.md`와 `_forAI/inventory.md`를 우선 기준으로 본다.

## 목차

1. [목적](#목적)
2. [1. 현재 기준 구조](#1-현재-기준-구조)
3. [2. 현재 repo 내부 매핑](#2-현재-repo-내부-매핑)
4. [3. 제거된 legacy 표면](#3-제거된-legacy-표면)

## 목적

현재 repo 안의 자산이 어떤 canonical 경로로 정리되었는지, 무엇이 compatibility shim인지, 무엇이 제거되었는지 상태 중심으로 정리한 매핑표다.

상태 표기:

- `Done`
- `Bridge`
- `Pending`
- `Removed`
- `Keep`

## 1. 현재 기준 구조

```text
src/
  common/
    runtime/
    contracts/
    protocols/
    tools/
  visionflow/
  voiceFlow/
    sources/
    gateways/
    sample/
    utils/
  asrFlow/
    bootstrap.py
    processors/
    workers/
    vendors/
    services/
    utils/
    contracts/   # skeleton
    sources/     # skeleton
  ttsFlow/
    engines/
    services/
    gateways/
    sample/
  neuroflow/
    app/
```

## 2. 현재 repo 내부 매핑

### 2.1 Common

| 현재 자산 | canonical / 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `src/common/contracts/packets.py` | 유지 | `Done` | `AudioPacket`, `AsrResultPacket`, `Packet` 공용 계약 |
| `src/common/contracts/asr_gateway.py` | 유지 | `Done` | ASR request/response contract |
| `src/common/contracts/job.py` | 유지 | `Done` | request/result 상태 모델 |
| `src/common/protocols/common_protocol.md` | 유지 | `Done` | 공통 프로토콜 문서 초안 |
| `src/common/protocols/nfcp.py` | 유지 | `Done` | NFCP 구현 |
| `src/common/runtime/bus.py` | 유지 | `Done` | canonical `TopicBus` |
| `src/common/runtime/audio_codec.py` | 유지 | `Done` | canonical transport audio codec |
| `src/visionflow/pipeline/bus.py` | `src/common/runtime/bus.py` | `Bridge` | compatibility re-export |
| `src/asrFlow/utils/audio.py` | `src/common/runtime/audio_codec.py` | `Bridge` | compatibility re-export |

### 2.2 ASR Core

| 현재 자산 | canonical / 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `src/asrFlow/bootstrap.py` | 유지 | `Done` | env load / processor bootstrap |
| `src/asrFlow/processors/miso_stt_asr.py` | 유지 | `Done` | canonical batch/chunk processor |
| `src/asrFlow/processors/qwen_streaming_asr.py` | 유지 | `Done` | canonical native streaming processor |
| `src/asrFlow/workers/asr_worker.py` | 유지 | `Done` | canonical chunk worker |
| `src/asrFlow/workers/accumulate_asr_worker.py` | 유지 | `Done` | canonical accumulate worker |
| `src/asrFlow/workers/streaming_asr_worker.py` | 유지 | `Done` | canonical stream worker |
| `src/asrFlow/services/nfcp_asr_handler.py` | 유지 | `Done` | ingress request -> processor 호출 |
| `src/asrFlow/vendors/whisper/*` | 유지 | `Done` | Whisper canonical vendor 경로 |
| `src/asrFlow/vendors/qwen_asr/*` | 유지 | `Done` | Qwen ASR canonical vendor 경로 |
| `src/asrFlow/contracts/` | 유지 | `Pending` | 뼈대만 있고 실계약 없음 |
| `src/asrFlow/sources/` | 유지 | `Pending` | 뼈대만 있고 실소유 자산 없음 |

### 2.3 Voice Ingress

| 현재 자산 | canonical / 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `src/voiceFlow/main.py` | 유지 | `Done` | voice sample launcher |
| `src/voiceFlow/sources/microphone_source.py` | 유지 | `Done` | canonical 로컬 마이크 source |
| `src/voiceFlow/sources/audiomi_source.py` | 유지 | `Done` | canonical 외부 PCM/TCP ingress source |
| `src/voiceFlow/gateways/asr_ingress_server.py` | 유지 | `Done` | canonical NFCP ingress server |
| `src/voiceFlow/sample/microphone_client.py` | 유지 | `Done` | canonical microphone request client |
| `src/voiceFlow/sample/simple_mic_test.py` | 유지 | `Keep` | 간단한 입력 테스트 경로 |
| `src/voiceFlow/sample/list_microphones.py` | 유지 | `Keep` | 디바이스 점검 보조 sample |
| `src/voiceFlow/sample/mic_level_monitor.py` | 유지 | `Keep` | 입력 레벨 모니터링 UI |
| `src/voiceFlow/utils/audio_device.py` | 유지 | `Done` | canonical 디바이스 유틸 |

### 2.4 App Layer

| 현재 자산 | canonical / 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `src/neuroflow/app/asr_server.py` | 유지 | `Done` | server composition root |
| `src/neuroflow/app/tts_server.py` | 유지 | `Done` | NFCP TTS server composition root |
| `src/neuroflow/app/tts_rest_server.py` | 유지 | `Done` | 외부 앱/키오스크용 TTS REST composition root |
| `src/neuroflow/app/asr_chunk_realtime.py` | 유지 | `Done` | 로컬 마이크 chunk UI |
| `src/neuroflow/app/audiomi_asr_chunk_realtime.py` | 유지 | `Done` | audioMi accumulate UI |
| `src/neuroflow/app/asr_stream_realtime.py` | 유지 | `Done` | native stream UI |
| `src/neuroflow/app/asr_model_catalog.py` | 유지 | `Done` | model/backend 분류 |
| `src/neuroflow/app/asr_ui_common.py` | 유지 | `Done` | UI 공용 bridge/gauge |

### 2.5 Vision

| 현재 자산 | canonical / 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `src/visionflow/**/*` | 현 위치 유지 | `Keep` | 1차 이동 대상이 아님 |
| `src/common/tools/download_mediapipe_models.py` | 유지 | `Done` | MediaPipe 기본 모델 bootstrap 도구 |

### 2.6 TTS Core

| 현재 자산 | canonical / 목표 경로 | 상태 | 메모 |
| --- | --- | --- | --- |
| `src/ttsFlow/engines/speecht5_ko.py` | 유지 | `Done` | 기본 한국어 품질 TTS engine |
| `src/ttsFlow/engines/piper_ko.py` | 유지 | `Done` | 빠른 fallback용 Piper ONNX engine |
| `src/ttsFlow/engines/stub.py` | 유지 | `Done` | smoke test tone engine |
| `src/ttsFlow/services/nfcp_tts_handler.py` | 유지 | `Done` | TTS request -> engine 호출 |
| `src/ttsFlow/gateways/tts_server.py` | 유지 | `Done` | canonical NFCP TTS server |
| `src/ttsFlow/gateways/rest_tts_server.py` | 유지 | `Done` | 외부 앱/키오스크용 REST gateway |
| `src/ttsFlow/sample/nfcp_tts_client.py` | 유지 | `Done` | NFCP TTS smoke client |
| `src/common/tools/download_tts_models.py` | 유지 | `Done` | Piper fallback 모델 다운로드 도구 |

## 3. 제거된 legacy 표면

| 과거 자산 | 현재 상태 | 메모 |
| --- | --- | --- |
| 루트 `main.py` | `Removed` | 저장소 공용 진입점 역할 종료 |
| `deviceMngUI.py` 및 관련 `.spec` | `Removed` | 현재 public surface에서 제외 |
| `src/asrFlow/gateways/tcp_asr_server.py` | `Removed` | canonical ownership을 `voiceFlow.gateways.asr_ingress_server`로 정리 |
| `src/asrFlow/sample/microphone_client.py` | `Removed` | canonical ownership을 `voiceFlow.sample.microphone_client`로 정리 |
| `src/voiceFlow` 내부 STT processor/worker/vendor | `Removed` | canonical ownership을 `asrFlow`로 정리 |
| `src/voiceFlow/sample/asr_realtime.py` | `Removed` | `neuroflow.app.asr_chunk_realtime`로 통합 |
| `src/voiceFlow/sample/audiomi_asr_realtime.py` | `Removed` | `neuroflow.app.audiomi_asr_chunk_realtime`로 통합 |
