# asrFlow

`asrFlow`는 `NeuroFlow Common Protocol v1` 기반 ASR 서버 시작점이다.

현재 1차 구현 범위:

- `PING`
- `DESCRIBE`
- `ASR_TRANSCRIBE`
- 마이크 입력 샘플 클라이언트


## 실행

프로젝트 루트에서:

```bash
uv run asrflow-server --env .env
```

다른 터미널에서:

```bash
uv run asrflow-mic-client --host 127.0.0.1 --port 26100 --duration 4
```


## 환경 변수

```env
ASRFLOW_HOST=0.0.0.0
ASRFLOW_PORT=26100

ASRFLOW_STT_BACKEND=ct2
ASRFLOW_STT_MODEL=large-v3
ASRFLOW_STT_MODEL_PATH=
ASRFLOW_STT_DEVICE=auto
ASRFLOW_STT_FP16=true
ASRFLOW_STT_LANGUAGE=ko
ASRFLOW_STT_TASK=transcribe
ASRFLOW_STT_BEAM_SIZE=5
```


## 현재 구조

- `gateways/tcp_asr_server.py`
  - NFCP TCP 서버
- `processors/miso_stt_asr.py`
  - 기존 `voiceFlow` processor 브리지
- `utils/audio.py`
  - 요청 body 오디오 디코딩
- `sample/microphone_client.py`
  - 마이크 녹음 후 서버로 요청 전송


## 현재 제약

- 1차 구현은 단건 `ASR_TRANSCRIBE` 중심
- streaming request/partial event는 후속 확장 예정
- processor core는 아직 `voiceFlow` 자산을 재사용
