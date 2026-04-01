# NeuroFlow Architecture

## 1. Layer Ownership

```mermaid
flowchart TD
    A[common<br/>runtime / contracts / protocols] --> B[voiceFlow<br/>sources / ingress / client]
    A --> C[asrFlow<br/>bootstrap / processors / workers / vendors / services]
    A --> D[visionflow<br/>sources / workers / samples]
    B --> E[neuroflow.app<br/>composition root / app entry points]
    C --> E

    classDef core fill:#e7f1ff,stroke:#2458a6,color:#102542;
    classDef edge fill:#eef8e8,stroke:#4f8a10,color:#214d12;
    classDef app fill:#fff3d9,stroke:#a36d00,color:#5b3b00;

    class A core;
    class C,D core;
    class B edge;
    class E app;
```

핵심 해석:

- `common`은 공용 기반층이다.
- `voiceFlow`는 오디오 ingress와 edge를 소유한다.
- `asrFlow`는 순수 ASR 코어를 소유한다.
- `neuroflow.app`은 조립과 실행 진입점만 맡는다.

## 2. Canonical Runtime Paths

```mermaid
flowchart LR
    Mic[MicrophoneSource] --> Bus1["TopicBus<br/>audio/raw"]
    AudioMi[AudioMiSource] --> Bus1
    Bus1 --> Chunk["AsrWorker<br/>chunk"]
    Bus1 --> Acc["AccumulateAsrWorker<br/>accumulate"]
    Bus1 --> Stream["StreamingAsrWorker<br/>native stream"]
    Chunk --> Bus2["TopicBus<br/>text/asr"]
    Acc --> Bus2
    Stream --> Bus2
    Bus2 --> UI1["asr_chunk_realtime"]
    Bus2 --> UI2["audiomi_asr_chunk_realtime"]
    Bus2 --> UI3["asr_stream_realtime"]
```

## 3. Network ASR Server Flow

```mermaid
flowchart LR
    Client["NFCP client"] --> Ingress["voiceFlow.gateways.asr_ingress_server"]
    Ingress --> Protocol["common.protocols.nfcp"]
    Ingress --> Codec["common.runtime.audio_codec"]
    Codec --> Handler["asrFlow.services.nfcp_asr_handler"]
    Handler --> BatchProc["asrFlow.processors.miso_stt_asr"]
    Ingress --> StreamProc["asrFlow.processors.qwen_streaming_asr<br/>(optional)"]
    BatchProc --> BatchVendor["Whisper / Qwen vendor"]
    StreamProc --> StreamVendor["Qwen ASR transformers"]
```

## 4. Public Entry Points

```mermaid
flowchart TD
    NFASR["uv run nf-asr-server"] --> AppServer["neuroflow.app.asr_server"]
    AppServer --> Ingress["voiceFlow.gateways.asr_ingress_server"]
    AppServer --> Handler["asrFlow.services.nfcp_asr_handler"]

    NFChunk["uv run nf-asr-chunk-realtime"] --> ChunkApp["neuroflow.app.asr_chunk_realtime"]
    NFAudioMi["uv run nf-audiomi-asr-chunk-realtime"] --> AudioMiApp["neuroflow.app.audiomi_asr_chunk_realtime"]
    NFStream["uv run nf-asr-stream-realtime"] --> StreamApp["neuroflow.app.asr_stream_realtime"]
    NFVoice["uv run nf-voice"] --> VoiceMain["voiceFlow.main"]
```

## 5. Model Experiment Split

```mermaid
flowchart LR
    ChunkModels["Chunk examples"] --> Whisper["Whisper<br/>ct2 / hf_generate / hf_pipeline"]
    ChunkModels --> QwenChunk["Qwen ASR<br/>qwen_transformers"]

    StreamModels["Stream examples"] --> QwenStream["Qwen ASR<br/>qwen_transformers / native streaming"]
```

현재 기준:

- `chunk` 예제는 일반 실험 UI다.
- `stream` 예제는 native streaming 지원 모델만 다룬다.
- 현재 canonical native streaming 경로는 `Qwen ASR + qwen-asr transformers backend`다.

## 6. Live Compatibility Surface

```mermaid
flowchart LR
    Legacy1["visionflow.pipeline.bus"] --> Canon1["common.runtime.bus"]
    Legacy2["asrFlow.utils.audio"] --> Canon2["common.runtime.audio_codec"]
```

현재 소스 기준으로 문서화할 가치가 남아 있는 compatibility surface는 위 두 경로 정도다.
