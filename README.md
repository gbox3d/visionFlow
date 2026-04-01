# NeuroFlow

`NeuroFlow`는 vision runtime, audio ingress, ASR core를 한 저장소에서 운영하는 멀티모달 런타임이다.

현재 canonical 경계는 아래 5축으로 보는 것이 가장 정확하다.

- `common`
  - 공용 contract, protocol, runtime
- `neuroflow.app`
  - composition root와 app-level entry point
- `voiceFlow`
  - source, client, ingress server
- `asrFlow`
  - env/bootstrap, processor, worker, vendor, ASR handler
- `visionflow`
  - 카메라/MediaPipe sample과 vision launcher

즉 NFCP ASR 서버는 `voiceFlow` transport와 `asrFlow` core를 `neuroflow.app.asr_server`에서 조립해 띄운다.

## Install

```bash
uv sync
uv run nf-vision-models-download
```

Python `>=3.11` 기준이다.

`.env`가 필요하면 `sample.env`를 기준으로 프로젝트 루트에 배치하면 된다.

## Public Entry Points

```bash
uv run nf-vision
uv run nf-vision-models-download

uv run nf-voice
uv run nf-voice-mic-client --host 127.0.0.1 --port 26100 --duration 4

uv run nf-asr-server
uv run nf-asr-chunk-realtime
uv run nf-audiomi-asr-chunk-realtime
uv run nf-asr-stream-realtime
```

- `nf-vision`
  - vision sample menu launcher
- `nf-vision-models-download`
  - MediaPipe 기본 모델 bootstrap
- `nf-voice`
  - voice source / device / sample app launcher
- `nf-voice-mic-client`
  - 로컬 마이크를 녹음해 NFCP ASR ingress 서버로 보내는 batch client
- `nf-asr-server`
  - canonical NFCP ASR 서버
  - `ASR_TRANSCRIBE` batch 요청을 처리한다
  - `NF_ASR_STREAM_ENABLED=true`면 `ASR_TRANSCRIBE_STREAM`도 함께 노출한다
- `nf-asr-chunk-realtime`
  - 로컬 마이크 기반 chunk 실험 UI
  - Whisper(`ct2`, `hf_generate`, `hf_pipeline`)와 Qwen ASR(`qwen_transformers`)를 시험한다
- `nf-audiomi-asr-chunk-realtime`
  - audioMi 입력 기반 accumulate chunk 실험 UI
  - 누적 buffer 전체를 다시 추론하면서 suffix만 UI에 반영한다
- `nf-asr-stream-realtime`
  - 로컬 마이크 기반 native streaming 실험 UI
  - 현재 canonical stream 경로는 `Qwen ASR + qwen-asr transformers backend`다
  - Windows/Linux 모두 같은 경로로 실행 가능하게 맞춰져 있다

## ASR Runtime Split

### 1. Batch / Chunk

```text
MicrophoneSource or AudioMiSource
  -> TopicBus("audio/raw")
  -> AsrWorker / AccumulateAsrWorker
  -> MisoSttAsrProcessor
  -> TopicBus("text/asr")
  -> UI or NFCP response
```

- 설정 키는 `ASRFLOW_STT_*`
- 대표 backend는 `ct2`, `hf_generate`, `hf_pipeline`, `qwen_transformers`
- `ASRFLOW_STT_MODEL_PATH`를 주면 로컬 모델 경로를 우선 사용한다

### 2. Native Stream

```text
MicrophoneSource
  -> TopicBus("audio/raw")
  -> StreamingAsrWorker
  -> QwenStreamingAsrProcessor
  -> TopicBus("text/asr")
  -> UI or NFCP stream response
```

- 설정 키는 `ASRFLOW_STREAM_*`
- 현재 canonical model family는 `Qwen/Qwen3-ASR-0.6B`
- `qwen_asr`의 streaming 로직을 transformers backend로 재현해 `vLLM` 없이 동작한다

## NFCP ASR Server

`uv run nf-asr-server`는 아래 두 경로를 한 프로세스에서 조립한다.

```text
voiceFlow.gateways.asr_ingress_server
  -> common.protocols.nfcp
  -> common.runtime.audio_codec
  -> asrFlow.services.nfcp_asr_handler
  -> asrFlow.processors.miso_stt_asr

optional stream path
  -> asrFlow.processors.qwen_streaming_asr
```

자주 쓰는 점검 순서:

```bash
uv run nf-asr-server
uv run nf-voice-mic-client --host 127.0.0.1 --port 26100 --duration 4
```

## Vision Models

vision sample은 `models/` 아래 기본 MediaPipe 자산을 요구한다.

- `models/blaze_face_short_range.tflite`
- `models/face_landmarker.task`
- `models/pose_landmarker.task`
- `models/pose_landmarker_lite.task`

```bash
uv run nf-vision-models-download
uv run nf-vision-models-download --list
uv run nf-vision-models-download --force
uv run nf-vision-models-download --only face-detector face-landmarker
uv run nf-vision-models-download --only pose-full pose-lite
```

## Direct Module Runs

vision sample:

```bash
uv run python -m visionflow.sample.camera.list_cameras
uv run python -m visionflow.sample.camera.simple --camera-id 0 --width 1280 --height 720
uv run python -m visionflow.sample.face_detection.simple --running-mode LIVE_STREAM --min-score 0.6
uv run python -m visionflow.sample.pose.simple --running-mode LIVE_STREAM
uv run python -m visionflow.sample.detect_test
```

voice / ASR sample:

```bash
uv run python -m voiceFlow.sample.microphone_client --host 127.0.0.1 --port 26100 --duration 4
uv run python -m neuroflow.app.asr_chunk_realtime
uv run python -m neuroflow.app.audiomi_asr_chunk_realtime
uv run python -m neuroflow.app.asr_stream_realtime
```

## Environment

`sample.env`는 현재 canonical 키 기준으로 정리돼 있다.

chunk / batch 계열:

```env
ASRFLOW_STT_BACKEND=qwen_transformers
ASRFLOW_STT_MODEL=Qwen/Qwen3-ASR-0.6B
ASRFLOW_STT_MODEL_PATH=
ASRFLOW_STT_DEVICE=auto
ASRFLOW_STT_FP16=true
ASRFLOW_STT_LANGUAGE=auto
ASRFLOW_STT_TASK=transcribe
ASRFLOW_STT_BEAM_SIZE=5
ASRFLOW_STT_MAX_NEW_TOKENS=256
ASRFLOW_STT_MAX_INFERENCE_BATCH_SIZE=8
ASRFLOW_STT_CHUNK_SEC=3.0
ASRFLOW_STT_SAMPLERATE=16000
```

native stream 계열:

```env
ASRFLOW_STREAM_MODEL=Qwen/Qwen3-ASR-0.6B
ASRFLOW_STREAM_MODEL_PATH=
ASRFLOW_STREAM_LANGUAGE=auto
ASRFLOW_STREAM_CHUNK_SEC=2.0
ASRFLOW_STREAM_SAMPLERATE=16000
ASRFLOW_STREAM_MAX_NEW_TOKENS=512
ASRFLOW_STREAM_UNFIXED_CHUNK_NUM=2
ASRFLOW_STREAM_UNFIXED_TOKEN_NUM=5
NF_ASR_STREAM_ENABLED=true
```

ingress / network 계열:

```env
NF_ASR_SERVER_HOST=0.0.0.0
NF_ASR_SERVER_PORT=26100
AUDIOMI_HOST=127.0.0.1
AUDIOMI_PORT=26070
AUDIOMI_CHECKCODE=20250918
```

## Layout

```text
src/
  common/
    contracts/
    protocols/
    runtime/
    tools/
  neuroflow/
    app/
  visionflow/
    main.py
    processors/
    sample/
    sources/
    workers/
  voiceFlow/
    gateways/
    main.py
    sample/
    sources/
    utils/
  asrFlow/
    bootstrap.py
    processors/
    services/
    workers/
    vendors/
    utils/
```

상세 구조는 [`_forAI/architecture.md`](_forAI/architecture.md), 현재 인벤토리는 [`_forAI/inventory.md`](_forAI/inventory.md)를 보면 된다.
