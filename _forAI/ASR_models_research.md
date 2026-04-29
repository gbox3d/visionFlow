# ASR 모델 리서치 요약

> 용도
> 실시간 ASR 모델을 빠르게 비교하기 위한 서베이용 요약 문서다.
> 수치와 모델 상태는 원문 리서치 기준이므로, 실제 도입 전에는 모델 카드, 라이선스, 실행 환경에서 재검증한다.

## 목차

1. [핵심 요약](#1-핵심-요약)
2. [Whisper의 실시간 한계](#2-whisper의-실시간-한계)
3. [스트리밍 ASR 구조 키워드](#3-스트리밍-asr-구조-키워드)
4. [후보 모델 요약](#4-후보-모델-요약)
5. [비교표](#5-비교표)
6. [NeuroFlow 적용 방향](#6-neuroflow-적용-방향)
7. [평가 체크리스트](#7-평가-체크리스트)

## 1. 핵심 요약

실시간 ASR은 단순히 WER이 낮은 모델보다 latency, partial 안정성, 무음 구간 환각, 동시 처리 비용이 중요하다.

Whisper 계열은 batch 전사 품질은 좋지만, 실시간에서는 짧은 chunk를 반복 처리하는 pseudo-streaming 방식이 된다. 이 구조는 지연 시간과 문맥 단절이 생기고, 무음/잡음에서 환각이 발생할 수 있다.

실시간 전사에는 처음부터 streaming을 고려한 모델이 더 적합하다. 후보군은 Qwen3-ASR, Voxtral Realtime, NVIDIA Nemotron/Parakeet, Cohere Transcribe, SenseVoice 계열로 정리할 수 있다.

NeuroFlow는 이미 `Qwen/Qwen3-ASR-0.6B` 기반 streaming 실험 경로가 있으므로, 우선 이 모델을 baseline으로 두고 다른 모델은 adapter 방식으로 비교하는 것이 좋다.

## 2. Whisper의 실시간 한계

### 2.1 Pseudo-streaming 문제

Whisper는 원래 일정 길이의 오디오를 한 번에 처리하는 offline-first 모델이다. 실시간처럼 사용하려면 1-5초 단위로 오디오를 잘라 넣고 결과를 이어 붙여야 한다.

주요 문제:

- end-to-end latency가 커진다.
- chunk 경계에서 단어와 문장 문맥이 끊긴다.
- 문장부호, 대소문자, 단어 경계가 흔들릴 수 있다.
- overlap, timestamp 보정, VAD 등을 붙이면 파이프라인이 복잡해진다.

### 2.2 무음/잡음 환각

Whisper는 무음, 배경 소음, 긴 pause에서 실제 발화가 아닌 문장을 생성할 수 있다. 회의록, 의료, 법률, 방송처럼 기록 신뢰성이 중요한 환경에서는 큰 리스크다.

완화책은 VAD로 무음 구간을 걸러내는 것이지만, 이 역시 별도 지연과 구현 복잡도를 만든다.

### 2.3 비용 문제

Whisper Large 계열은 모델 weight, KV cache, chunk overlap, VAD, 동시 stream scheduling 비용이 함께 발생한다. 동시 접속이 늘어나는 서비스에서는 GPU 비용이 빠르게 커질 수 있다.

## 3. 스트리밍 ASR 구조 키워드

| 구조 | 요약 | 장점 |
| --- | --- | --- |
| Cache-Aware RNN-T | 이전 frame의 hidden state를 cache로 재사용 | 낮은 지연, 긴 문맥 유지 |
| TDT | token과 duration을 함께 예측해 blank frame을 줄임 | 매우 빠른 처리 |
| Causal Encoder | 미래 frame을 보지 않고 현재까지의 audio만 사용 | 실시간 causality 보장 |
| Delayed Streams | 지연 시간을 명시적으로 조절 | latency/정확도 trade-off 가능 |
| NAR | token을 순차 생성하지 않고 병렬 예측 | 높은 throughput |

## 4. 후보 모델 요약

### Qwen3-ASR Family

범용 다국어 ASR 후보. 원문 기준으로 Qwen3-ASR-1.7B는 고품질 전사, Qwen3-ASR-0.6B는 효율과 낮은 latency가 장점이다.

NeuroFlow와 가장 잘 맞는 후보이며, 현재 코드도 Qwen3-ASR-0.6B streaming 실험 경로를 갖고 있다.

### Voxtral-Mini-4B-Realtime

Mistral 계열 실시간 ASR 후보. causal encoder와 delayed stream 구조로 latency를 조절할 수 있다는 점이 핵심이다.

방송급 caption이나 품질 우선 실시간 전사에 적합하지만, 4B급이라 자원 요구량을 확인해야 한다.

### NVIDIA Nemotron-Speech-Streaming-En-0.6B

영어 전용 streaming ASR 후보. Cache-Aware FastConformer RNN-T 기반으로 ultra-low latency 목적에 맞다.

영어 voice agent나 영어 회의 전사에는 유리하지만, 한국어/다국어 요구가 있으면 주력 후보로는 제한적이다.

### NVIDIA Parakeet-TDT-0.6B-v3

TDT 기반 고속 ASR 후보. blank frame을 줄여 처리 속도가 빠른 쪽에 초점이 있다.

속도 중심 실험에는 좋지만, 출력 가독성, punctuation, 문맥 품질은 별도 평가가 필요하다.

### Cohere-Transcribe-03-2026

enterprise 영어 전사 후보. 원문 기준 영어 WER이 강점으로 언급된다.

다만 language tag, timestamp, diarization, VAD 필요 여부를 확인해야 한다. non-speech noise에 민감할 수 있다.

### SenseVoice-Small

가벼운 NAR encoder-only 계열 후보. 작은 메모리, 빠른 처리, emotion/event tag가 장점으로 정리된다.

edge, WebRTC, 로컬 앱, CPU 기반 실험 후보로 적합하다.

## 5. 비교표

| 모델 | 크기 | 구조 | 강점 | 주의점 |
| --- | ---: | --- | --- | --- |
| Qwen3-ASR-1.7B | 1.7B | NAR / omni ASR | 범용 다국어, 고품질 | vLLM/자원 요구 재검증 |
| Qwen3-ASR-0.6B | 0.6B급 | Qwen streaming | NeuroFlow 현재 baseline 후보 | timestamp/partial 품질 확인 |
| Voxtral-Mini-4B-Realtime | 4B | Causal encoder | latency 조절, 방송 caption 후보 | VRAM 요구량 큼 |
| Nemotron-Speech-En-0.6B | 0.6B | Cache-Aware RNN-T | 영어 ultra-low latency | 영어 중심 |
| Parakeet-TDT-0.6B-v3 | 0.6B | TDT | 매우 빠른 처리 | 가독성/문맥 별도 검증 |
| Cohere-Transcribe-03-2026 | 2B | Conformer encoder-decoder | enterprise 영어 성능 | VAD 필요 가능성 |
| SenseVoice-Small | small | NAR encoder | edge/CPU, 이벤트 태그 | 실제 한국어 품질 검증 필요 |

## 6. NeuroFlow 적용 방향

현재 NeuroFlow 기준으로는 모델을 바로 많이 추가하기보다 평가 기준과 adapter 경계를 먼저 고정하는 것이 좋다.

단기 방향:

- `Qwen3-ASR-0.6B`를 streaming baseline으로 유지
- TTFT, RTF, partial 안정성, final 정확도 측정
- VAD gate를 stream 앞단에 붙이는 실험
- `ASR_CLEAR_BUFFER`와 session reset 흐름 정리

중기 방향:

- `StreamingAsrEngine` 같은 공통 interface 도입
- Qwen local, Qwen vLLM, SenseVoice, NVIDIA RNN-T/TDT 계열을 adapter로 분리 비교
- 모델 출력 형식을 `partial`, `final`, `new_text`, `full_text`, `latency_ms` 중심으로 표준화

## 7. 평가 체크리스트

| 항목 | 질문 |
| --- | --- |
| TTFT | 첫 글자가 몇 ms 안에 나오는가 |
| RTF | 실시간보다 충분히 빠른가 |
| partial 안정성 | 중간 결과가 너무 자주 뒤집히지 않는가 |
| final 정확도 | 최종 문장이 읽을 만한 품질인가 |
| 무음 안정성 | 무음/잡음에서 없는 말을 만들지 않는가 |
| 한국어 품질 | 띄어쓰기, 조사, 종결어미가 안정적인가 |
| 자원 사용량 | VRAM/RAM/CPU/GPU 사용량이 운영 가능한가 |
| 배포 경로 | local, vLLM, ONNX 중 어느 방식이 현실적인가 |
| 라이선스 | 상업적 사용과 재배포가 가능한가 |

## 한 줄 결론

실시간 ASR은 Whisper를 더 잘 쪼개는 문제가 아니라, streaming-native 모델과 안정적인 session contract를 고르는 문제다. NeuroFlow는 Qwen streaming 경로를 baseline으로 두고, VAD와 adapter 구조를 붙여 모델을 비교하는 방향이 가장 안전하다.
