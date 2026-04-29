# voiceFlow

`voiceFlow`는 `NeuroFlow`의 audio ingress / edge 계층이다.

canonical 범위:

- microphone / external audio source
- device utility
- NFCP ingress server
- microphone client

`voiceFlow`는 텍스트를 직접 만드는 코어가 아니라, 오디오 입력과 네트워크 edge를 소유하고 `asrFlow`와 연결하는 역할만 맡는다.

## 현재 소유 범위

- `main.py`
  - voice sample launcher
- `sources/microphone_source.py`
  - 로컬 마이크 source
- `sources/audiomi_source.py`
  - audioMi TCP ingress source
- `gateways/asr_ingress_server.py`
  - canonical NFCP ingress server
- `sample/microphone_client.py`
  - batch transcribe client
- `sample/list_microphones.py`
  - 입력 장치 목록
- `sample/simple_mic_test.py`
  - microphone source smoke test
- `sample/mic_level_monitor.py`
  - 입력 레벨 모니터
- `utils/audio_device.py`
  - 디바이스 탐색/매칭 보조 유틸

연결되는 app entry point:

- `neuroflow.app.asr_server`
- `neuroflow.app.asr_chunk_realtime`
- `neuroflow.app.audiomi_asr_chunk_realtime`
- `neuroflow.app.asr_stream_realtime`

## Runtime Spine

```text
voiceFlow.sources / voiceFlow.sample
  -> common.runtime.bus.TopicBus("audio/raw")
  -> asrFlow.workers / asrFlow.processors
  -> common.runtime.bus.TopicBus("text/asr")
  -> UI

voiceFlow.gateways.asr_ingress_server
  -> common.protocols.nfcp
  -> common.runtime.audio_codec
  -> asrFlow.services.nfcp_asr_handler
  -> asrFlow.processors.miso_stt_asr
```

즉 `voiceFlow`는 입력과 transport를 맡고, 실제 STT 추론은 `asrFlow`가 맡는다.

## NFCP Server Surface

`voiceFlow.gateways.asr_ingress_server`는 아래 command를 처리한다.

- `PING`
- `DESCRIBE`
- `ASR_TRANSCRIBE`
- `ASR_TRANSCRIBE_STREAM`
  - `streaming_processor`가 구성된 경우에만 노출
  - 현재 `neuroflow.app.asr_server`에서는 `NF_ASR_STREAM_ENABLED=true`일 때 활성화된다

## 실행

```bash
uv run nf-voice
uv run nf-asr-server
uv run nf-voice-mic-client --host 127.0.0.1 --port 26100 --duration 4
uv run nf-asr-chunk-realtime
uv run nf-audiomi-asr-chunk-realtime
uv run nf-asr-stream-realtime
```

직접 실행:

```bash
uv run python -m voiceFlow.sample.list_microphones
uv run python -m voiceFlow.sample.simple_mic_test
uv run python -m voiceFlow.sample.mic_level_monitor
uv run python -m voiceFlow.sample.microphone_client --host 127.0.0.1 --port 26100 --duration 4
uv run python -m neuroflow.app.asr_chunk_realtime
uv run python -m neuroflow.app.audiomi_asr_chunk_realtime
uv run python -m neuroflow.app.asr_stream_realtime
```

## Canonical Path 기준

- bus
  - `common.runtime.bus.TopicBus`
- packet contracts
  - `common.contracts.packets`
- transport audio codec
  - `common.runtime.audio_codec`
- ASR processor / worker
  - `asrFlow.processors.*`
  - `asrFlow.workers.*`
- app composition
  - `neuroflow.app.asr_server`

## Environment

`voiceFlow`에서 자주 쓰는 ingress/network 값:

- `NF_ASR_SERVER_HOST`
- `NF_ASR_SERVER_PORT`
- `NF_ASR_STREAM_ENABLED`
- `AUDIOMI_HOST`
- `AUDIOMI_PORT`
- `AUDIOMI_CHECKCODE`
- `DEVICE_PATH`
- `CAMERA_DEVICE_PATH`
- `DEMO_DEVICE_PATH`

STT 모델/추론 설정은 `voiceFlow` 소유가 아니라 `asrFlow` 설정이다.

```env
ASRFLOW_STT_BACKEND=qwen_transformers
ASRFLOW_STT_MODEL=Qwen/Qwen3-ASR-0.6B
ASRFLOW_STT_MODEL_PATH=
ASRFLOW_STT_DEVICE=auto
ASRFLOW_STT_FP16=true
ASRFLOW_STT_LANGUAGE=auto
ASRFLOW_STT_TASK=transcribe
```

native stream 예제는 별도 키를 본다.

```env
ASRFLOW_STREAM_MODEL=Qwen/Qwen3-ASR-0.6B
ASRFLOW_STREAM_MODEL_PATH=
ASRFLOW_STREAM_LANGUAGE=auto
ASRFLOW_STREAM_CHUNK_SEC=2.0
ASRFLOW_STREAM_SAMPLERATE=16000
```

현재 native stream 경로는 기본 의존성 안에서 동작한다.
