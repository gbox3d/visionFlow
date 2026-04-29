# asrFlow

`asrFlow`는 `NeuroFlow`의 canonical ASR core 모듈이다.

이 패키지는 입력 장치, network ingress, client/server, UI를 소유하지 않는다. 받은 오디오를 모델에 넣어 텍스트 결과를 만드는 경계만 맡는다.

## 현재 소유 범위

- `bootstrap.py`
  - runtime env 로드와 batch processor bootstrap
- `processors/miso_stt_asr.py`
  - chunk / batch 계열 canonical processor
- `processors/qwen_streaming_asr.py`
  - native streaming용 Qwen processor
- `workers/asr_worker.py`
  - 고정 chunk worker
- `workers/accumulate_asr_worker.py`
  - 누적 buffer worker
- `workers/streaming_asr_worker.py`
  - native streaming worker
- `services/nfcp_asr_handler.py`
  - decoded request를 processor 호출로 바꾸는 handler
- `vendors/whisper/`
  - Whisper runtime
- `vendors/qwen_asr/`
  - Qwen ASR transformers runtime
- `utils/env.py`
  - ASR 설정 파싱 유틸
- `utils/text.py`
  - 텍스트 후처리 / UI 보조 유틸
- `utils/audio.py`
  - `common.runtime.audio_codec` compatibility re-export

## asrFlow가 소유하지 않는 것

- microphone source
- audioMi source
- NFCP ingress server
- microphone client
- realtime UI app

이 경계는 `voiceFlow`와 `neuroflow.app`가 맡는다.

## Runtime Split

### Chunk / Batch

```text
voiceFlow source / gateway
  -> common.runtime.bus.TopicBus("audio/raw")
  -> asrFlow.workers.asr_worker or accumulate_asr_worker
  -> asrFlow.processors.miso_stt_asr
  -> common.contracts.packets.AsrResultPacket
```

- 지원 backend: `ct2`, `hf_generate`, `hf_pipeline`, `qwen_transformers`
- 모델 경로 override는 `ASRFLOW_STT_MODEL_PATH`
- batch NFCP 서버도 같은 processor를 사용한다

### Native Stream

```text
MicrophoneSource or NFCP stream
  -> StreamingAsrWorker
  -> QwenStreamingAsrProcessor
  -> AsrResultPacket(meta.native_streaming=true)
```

- 현재 canonical stream backend는 `qwen_transformers`
- `QwenStreamingAsrProcessor`는 `qwen_asr` streaming 로직을 transformers backend로 재현한다
- 별도 `vLLM` dependency group은 현재 필요 없다

## Direct Usage

network edge가 필요하면 `asrFlow` 단독이 아니라 `neuroflow.app.asr_server` 또는 `voiceFlow` 쪽을 사용한다.

```bash
uv run nf-asr-server
uv run nf-voice-mic-client --host 127.0.0.1 --port 26100 --duration 4
uv run nf-asr-chunk-realtime
uv run nf-audiomi-asr-chunk-realtime
uv run nf-asr-stream-realtime
```

## Environment

batch / chunk 계열:

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
```

## 현재 구조

- `bootstrap.py`
  - env 로드와 runtime path 정규화
- `processors/miso_stt_asr.py`
  - canonical batch/chunk processor
- `processors/qwen_streaming_asr.py`
  - Qwen native streaming processor
- `workers/asr_worker.py`
  - 단건 chunk worker
- `workers/accumulate_asr_worker.py`
  - 누적형 실시간 worker
- `workers/streaming_asr_worker.py`
  - native streaming worker
- `services/nfcp_asr_handler.py`
  - ingress request -> processor 호출 변환
- `vendors/whisper/`
  - Whisper runtime
- `vendors/qwen_asr/`
  - Qwen ASR runtime

## 현재 제약

- chunk 예제는 Whisper 계열과 Qwen ASR transformers 계열을 다룬다
- native stream 예제는 현재 Qwen ASR 계열만 canonical 지원한다
- `QwenStreamingAsrProcessor`는 `16kHz PCM mono` 기준으로 동작한다
