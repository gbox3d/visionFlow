# NeuroFlow Memo

## 목차

1. [현재 보는 프레임](#현재-보는-프레임)
2. [프로토콜 표준: NFCP TCP](#프로토콜-표준-nfcp-tcp)
3. [Unity 연동 문서 메모](#unity-연동-문서-메모)
4. [현재 고정된 판단](#현재-고정된-판단)
5. [열린 질문](#열린-질문)
6. [다음 작업 추천](#다음-작업-추천)
7. [2026-04-28 프로젝트 파악 메모](#2026-04-28-프로젝트-파악-메모)
8. [TTS 통합 방향](#tts-통합-방향)

## 현재 보는 프레임

- `common`은 공용 기반층이다.
- `voiceFlow`는 ingress/edge 계층이다.
- `asrFlow`는 ASR 코어다.
- `ttsFlow`는 TTS 코어와 NFCP/REST TTS gateway 표면이다.
- `neuroflow.app`은 composition root다.
- `chunk` 예제와 `stream` 예제는 분리됐다.

## 프로토콜 표준: NFCP TCP

**내부 표준 서버는 NFCP TCP 방식을 우선 사용한다.** REST(FastAPI 등)는 외부 앱/키오스크 연동용 gateway로만 둔다.

- 기준 패턴: `voiceFlow.gateways.asr_ingress_server.AsrIngressServer`
- 프로토콜: `common.protocols.nfcp` (`read_frame` / `write_frame` / `build_*_frame`)
- 모든 서버는 `PING`, `DESCRIBE` 핸들러를 기본 포함한다
- dispatch loop → handle_client → run 구조를 따른다
- 추가 의존성(FastAPI, uvicorn 등) 없이 asyncio + NFCP만으로 구현한다
- REST가 필요한 경우 core/service를 재구현하지 않고 같은 handler 위에 얇은 gateway만 추가한다
- 참고 구현: `CamHubServer` (`visionflow.camhub.server`)

## Unity 연동 문서 메모

- `_forAI/developer_promise_system.md`는 Unity/C# 개발자를 위한 ASR/TTS 연동 가이드다.
- 예제 주소는 `192.168.4.218`로 고정한다.
- 현재 서비스 포트는 ASR TCP `26100`, TTS TCP `26120`, TTS REST `26121`이다.
- TCP 연동은 NFCP 64-byte header + meta JSON + data bytes 구조를 따른다.
- Unity 키오스크 TTS는 REST gateway 사용을 우선 권장한다.

## 현재 고정된 판단

- `ASRFLOW_STT_*`는 chunk / batch 계열 설정이다.
- `ASRFLOW_STREAM_*`는 native stream 계열 설정이다.
- `nf-asr-stream-realtime`는 현재 `Qwen ASR + qwen_transformers` 전용 경로다.
- `nf-asr-server`는 `voiceFlow.gateways.asr_ingress_server`를 canonical server로 쓴다.
- `nf-tts-server`는 `ttsFlow.gateways.tts_server`를 canonical server로 쓴다.
- `nf-tts-rest-server`는 `ttsFlow.gateways.rest_tts_server`를 외부 HTTP gateway로 쓴다.
- `NF_ASR_STREAM_ENABLED=true`면 NFCP server가 streaming command도 함께 노출한다.
- native stream 경로는 현재 기본 dependency 안에서 동작한다.
- TTS는 키오스크용 품질 우선 경로로 `speecht5-ko` CPU 실행을 기본값으로 둔다.

## 열린 질문

- `asrFlow/contracts`를 언제 실제 `StreamingSession` / `AsrEngine` 계약으로 채울지?
- `qwen_streaming_asr.py`를 stream vendor/service 계층으로 더 아래로 내릴지?
- stream 예제를 앞으로 Qwen 외 다른 모델 family까지 넓힐지?
- server streaming path에 partial/final 외 운영 메타를 얼마나 더 실을지?

## 다음 작업 추천

- `stream` 쪽에 실제 session contract를 도입
- `qwen_streaming_asr.py`의 계층 위치를 다시 판단
- README와 `_forAI`를 현재 canonical 구조 기준으로 계속 짧게 유지
- Unity/키오스크 연동 변경 시 `developer_promise_system.md` 예제를 함께 갱신

## 2026-04-28 프로젝트 파악 메모

- `forai-scaffold` 재실행 결과 `_forAI` 표준 문서 5개는 모두 기존 파일 유지 상태였다.
- 현재 public surface는 `pyproject.toml`의 `project.scripts` 기준 10개 entry point다.
- 서버 통신 표준은 실제 코드 기준 `asyncio TCP + common.protocols.nfcp`이며, CamHub도 REST/FastAPI가 아니라 NFCP TCP다. 단, TTS는 키오스크 편의용 REST gateway를 별도 entry point로 제공한다.
- NFCP 헤더는 구현과 프로토콜 문서 기준 `64 bytes`다.
- 자동 테스트 디렉터리나 pytest 설정은 현재 저장소에서 확인되지 않았다.

## TTS 통합 방향

- 별도 `NeuroFlow_TTS` 실험 프로젝트는 폐기하고 NeuroFlow 본체에 통합한다.
- 모든 실행은 `uv` 기준이다.
- 기본 내부 인터페이스는 NeuroFlow 표준에 맞춰 NFCP TCP로 둔다.
- 외부/키오스크 연동 편의를 위해 `nf-tts-rest-server` REST gateway도 제공한다.
- 기본 모델은 품질 우선 `ahnhs2k/speecht5-korean` (`NF_TTS_ENGINE=speecht5-ko`)이다.
- `NF_TTS_DEVICE=cpu`가 기본값이다. 현재 GPU 여유 메모리 약 3GB 상황에서는 TTS를 GPU에 올리지 않는다.
- `neurlang/piper-onnx-kss-korean` 계열 Piper ONNX 모델은 빠른 fallback으로만 둔다. Python `piper-tts` 런타임의 `pygoruut` 처리 한계 때문에 한국어 발음 품질이 낮았다.
