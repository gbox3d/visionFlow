# NeuroFlow Memo

## 현재 보는 프레임

- `common`은 공용 기반층이다.
- `voiceFlow`는 ingress/edge 계층이다.
- `asrFlow`는 ASR 코어다.
- `neuroflow.app`은 composition root다.
- `chunk` 예제와 `stream` 예제는 분리됐다.

## 프로토콜 표준: NFCP TCP

**새 서버를 만들 때는 반드시 NFCP TCP 방식을 사용한다.** REST(FastAPI, Flask 등)는 쓰지 않는다.

- 기준 패턴: `voiceFlow.gateways.asr_ingress_server.AsrIngressServer`
- 프로토콜: `common.protocols.nfcp` (`read_frame` / `write_frame` / `build_*_frame`)
- 모든 서버는 `PING`, `DESCRIBE` 핸들러를 기본 포함한다
- dispatch loop → handle_client → run 구조를 따른다
- 추가 의존성(FastAPI, uvicorn 등) 없이 asyncio + NFCP만으로 구현한다
- 참고 구현: `CamHubServer` (`visionflow.camhub.server`)

## 현재 고정된 판단

- `ASRFLOW_STT_*`는 chunk / batch 계열 설정이다.
- `ASRFLOW_STREAM_*`는 native stream 계열 설정이다.
- `nf-asr-stream-realtime`는 현재 `Qwen ASR + qwen_transformers` 전용 경로다.
- `nf-asr-server`는 `voiceFlow.gateways.asr_ingress_server`를 canonical server로 쓴다.
- `NF_ASR_STREAM_ENABLED=true`면 NFCP server가 streaming command도 함께 노출한다.
- native stream 경로는 현재 기본 dependency 안에서 동작한다.

## 열린 질문

- `asrFlow/contracts`를 언제 실제 `StreamingSession` / `AsrEngine` 계약으로 채울지?
- `qwen_streaming_asr.py`를 stream vendor/service 계층으로 더 아래로 내릴지?
- stream 예제를 앞으로 Qwen 외 다른 모델 family까지 넓힐지?
- server streaming path에 partial/final 외 운영 메타를 얼마나 더 실을지?

## 다음 작업 추천

- `stream` 쪽에 실제 session contract를 도입
- `qwen_streaming_asr.py`의 계층 위치를 다시 판단
- README와 `_forAI`를 현재 canonical 구조 기준으로 계속 짧게 유지
