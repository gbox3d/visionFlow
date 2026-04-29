# NeuroFlow Unity 개발자 연동 가이드

> 목적
> Unity/C# 개발자가 NeuroFlow의 ASR/TTS 서비스를 붙일 때 어디를 보면 되는지 안내하는 인덱스 문서다.
> 긴 TCP/REST 예제는 별도 문서로 분리했다.

## 목차

1. [현재 서비스 주소](#1-현재-서비스-주소)
2. [어떤 문서를 보면 되나](#2-어떤-문서를-보면-되나)
3. [권장 연동 방식](#3-권장-연동-방식)
4. [빠른 확인 명령](#4-빠른-확인-명령)
5. [운영 체크리스트](#5-운영-체크리스트)

## 1. 현재 서비스 주소

| 서비스 | 방식 | 주소 | 포트 | 용도 |
| --- | --- | --- | ---: | --- |
| `nf-asr-server` | NFCP TCP | `192.168.4.218` | `21861` | 음성 -> 텍스트 |
| `nf-tts-server` | NFCP TCP | `192.168.4.218` | `26120` | 텍스트 -> WAV |
| `nf-tts-rest-server` | HTTP REST | `192.168.4.218` | `26121` | 텍스트 -> WAV |

참고: `.env`의 ASR 기본 포트는 `26100`이지만, 현재 PM2 실행 명령은 `uv run nf-asr-server --port 21861`로 오버라이드되어 있다.

## 2. 어떤 문서를 보면 되나

| 문서 | 내용 |
| --- | --- |
| [unity_nfcp_tcp_guide.md](./unity_nfcp_tcp_guide.md) | `nf-asr-server`, `nf-tts-server` TCP/NFCP 프로토콜과 Unity C# 예제 |
| [unity_tts_rest_guide.md](./unity_tts_rest_guide.md) | `nf-tts-rest-server` REST API와 Unity `UnityWebRequest` 예제 |
| [common_protocol.md](../src/common/protocols/common_protocol.md) | NFCP 프로토콜 원문 |

## 3. 권장 연동 방식

| 사용처 | 권장 방식 | 이유 |
| --- | --- | --- |
| Unity에서 ASR 호출 | `nf-asr-server` TCP | ASR은 현재 REST gateway가 없고 NFCP가 canonical |
| Unity에서 TTS를 빠르게 붙이기 | `nf-tts-rest-server` REST | `UnityWebRequest`로 구현이 가장 단순 |
| Unity에서 내부 표준에 맞춰 TTS 호출 | `nf-tts-server` TCP | NeuroFlow 내부 표준 프로토콜과 동일 |

키오스크 UI의 TTS는 REST 방식을 우선 권장한다. ASR은 TCP/NFCP batch 요청부터 붙이는 것이 가장 단순하다.

## 4. 빠른 확인 명령

```bash
ss -ltnp | rg ':21861|:26120|:26121'
curl -fsS http://192.168.4.218:26121/health
```

TTS REST WAV 생성:

```bash
curl -sS -o /tmp/neuroflow_tts.wav \
  -X POST http://192.168.4.218:26121/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"안녕하세요. 안내를 시작합니다.","audio_format":"wav","language":"ko","speed":1.0}'
```

## 5. 운영 체크리스트

- Unity PC와 NeuroFlow 서버가 같은 네트워크에서 `192.168.4.218`로 접근 가능한지 확인한다.
- 방화벽에서 `21861`, `26120`, `26121` 포트가 열려 있어야 한다.
- ASR은 WAV 또는 PCM S16LE mono 16kHz를 우선 사용한다.
- TTS REST는 키오스크 클라이언트용 기본 권장 경로다.
- TCP NFCP는 내부 표준 연동이나 긴 세션이 필요한 경우 사용한다.
- TTS 현재 기본 엔진은 CPU 실행이므로 GPU VRAM을 추가로 쓰지 않는다.
