# NeuroFlow

`NeuroFlow`는 vision, audio, ASR을 한 저장소에서 운영하는 멀티모달 런타임이다.

## 이 문서를 처음 보는 사람을 위한 지침

1. **설치**: `uv sync` → `uv run nf-vision-models-download` 순서로 실행한다.
2. **환경 설정**: `sample.env`를 복사하여 프로젝트 루트에 `.env`로 배치한다.
3. **프로토콜**: 모든 서버 간 통신은 **NFCP TCP** 기반이다. REST/HTTP는 사용하지 않는다.
4. **구조 파악**: 코드 구조와 아키텍처는 [`_forAI/`](_forAI/) 디렉터리 문서를 먼저 읽는다.
5. **새 서버 추가 시**: `common.protocols.nfcp`의 `read_frame`/`write_frame`을 사용하고, `PING`/`DESCRIBE` 핸들러를 반드시 포함한다. 참고 구현은 `AsrIngressServer`와 `CamHubServer`다.

---

## 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [설치](#설치)
3. [Public Entry Points](#public-entry-points)
   - [visionFlow](#visionflow)
   - [voiceFlow](#voiceflow)
   - [asrFlow](#asrflow)
4. [사용법 상세](#사용법-상세)
   - [nf-vision-camhub (카메라 허브 서버)](#nf-vision-camhub-카메라-허브-서버)
   - [nf-vision-camhub-client (카메라 클라이언트)](#nf-vision-camhub-client-카메라-클라이언트)
   - [nf-asr-server (ASR 서버)](#nf-asr-server-asr-서버)
   - [nf-voice-mic-client (마이크 클라이언트)](#nf-voice-mic-client-마이크-클라이언트)
   - [nf-asr-chunk-realtime (Chunk ASR UI)](#nf-asr-chunk-realtime-chunk-asr-ui)
   - [nf-audiomi-asr-chunk-realtime (AudioMi Chunk UI)](#nf-audiomi-asr-chunk-realtime-audiomi-chunk-ui)
   - [nf-asr-stream-realtime (Native Streaming UI)](#nf-asr-stream-realtime-native-streaming-ui)
   - [nf-vision (Vision 샘플 런처)](#nf-vision-vision-샘플-런처)
   - [nf-vision-models-download (모델 다운로드)](#nf-vision-models-download-모델-다운로드)
   - [nf-voice (Voice 샘플 런처)](#nf-voice-voice-샘플-런처)
5. [ASR Runtime Split](#asr-runtime-split)
6. [NFCP 프로토콜](#nfcp-프로토콜)
7. [환경 변수](#환경-변수)
8. [Layout](#layout)

---

## 아키텍처 개요

```text
common         -> 공용 contract, protocol (NFCP), runtime (TopicBus)
visionflow     -> 카메라, MediaPipe 추론, camhub (영상 중계 서버)
voiceFlow      -> 오디오 source, client, ingress server
asrFlow        -> bootstrap, processor, worker, vendor, ASR handler
neuroflow.app  -> composition root, app-level entry point
```

NFCP ASR 서버는 `voiceFlow` transport와 `asrFlow` core를 `neuroflow.app.asr_server`에서 조립해 띄운다.
CamHub 서버는 `visionflow.camhub`에서 NFCP TCP로 카메라 프레임을 중계한다.

---

## 설치

```bash
uv sync
uv run nf-vision-models-download
```

- Python `>=3.11` 기준
- `.env`가 필요하면 `sample.env`를 복사하여 프로젝트 루트에 배치한다

---

## Public Entry Points

### visionFlow

| 커맨드 | 역할 |
| --- | --- |
| `uv run nf-vision` | vision sample menu launcher |
| `uv run nf-vision-models-download` | MediaPipe 기본 모델 다운로드 |
| `uv run nf-vision-camhub` | 카메라 이미지 중계 허브 서버 (NFCP TCP) |
| `uv run nf-vision-camhub-client` | 로컬 카메라 → camhub 전송 클라이언트 (NFCP TCP) |

### voiceFlow

| 커맨드 | 역할 |
| --- | --- |
| `uv run nf-voice` | voice source / device / sample app launcher |
| `uv run nf-voice-mic-client` | 마이크 녹음 후 NFCP ASR ingress 서버로 batch 전송 |

### asrFlow

| 커맨드 | 역할 |
| --- | --- |
| `uv run nf-asr-server` | canonical NFCP ASR 서버 (batch + optional stream) |
| `uv run nf-asr-chunk-realtime` | 로컬 마이크 chunk ASR 실험 UI |
| `uv run nf-audiomi-asr-chunk-realtime` | audioMi accumulate chunk ASR 실험 UI |
| `uv run nf-asr-stream-realtime` | Qwen native streaming 실험 UI |

---

## 사용법 상세

### nf-vision-camhub (카메라 허브 서버)

여러 카메라 클라이언트가 JPEG 프레임을 고유한 이름(`name`)으로 보내면, 이름별로 최신 프레임을 인메모리에 유지한다. AI 클라이언트가 카메라 이름으로 요청하면 현재 프레임을 NFCP 응답으로 전송한다.

```bash
# 기본 실행 (0.0.0.0:26200)
uv run nf-vision-camhub

# 특정 주소/포트로 바인드
uv run nf-vision-camhub --host 127.0.0.1 --port 8080
```

CLI 인자:

| 옵션 | 타입 | 기본값 | 환경 변수 | 설명 |
| --- | --- | --- | --- | --- |
| `--host` | `str` | `0.0.0.0` | `NF_CAMHUB_HOST` | TCP 바인드 주소. `0.0.0.0`이면 모든 인터페이스에서 수신 |
| `--port` | `int` | `26200` | `NF_CAMHUB_PORT` | TCP 리스닝 포트 |

NFCP 커맨드:

| Command | Code | 방향 | 역할 |
| --- | --- | --- | --- |
| `VISION_UPLOAD_FRAME` | 5003 | camera → hub | JPEG 프레임 업로드. meta: `{name, width, height}`, data: JPEG bytes |
| `VISION_GET_FRAME` | 5004 | AI → hub | 최신 프레임 조회. meta: `{name}` → 응답 data: JPEG bytes |
| `VISION_LIST_CAMERAS` | 5005 | any → hub | 등록된 카메라 목록 + 메타데이터 |
| `PING` | 99 | any → hub | 서비스 상태 확인 |
| `DESCRIBE` | 101 | any → hub | 서비스 상세 정보 + 현재 카메라 목록 |

### nf-vision-camhub-client (카메라 클라이언트)

로컬 카메라에서 프레임을 캡처하여 camhub 서버로 NFCP `VISION_UPLOAD_FRAME`을 반복 전송한다. 시작 시 `PING`으로 서버 연결을 확인한 뒤, 지정된 FPS로 JPEG 인코딩 프레임을 계속 보낸다.

```bash
# 기본 실행 (cam0, camera index 0, 10fps)
uv run nf-vision-camhub-client

# 카메라 이름과 인덱스 지정
uv run nf-vision-camhub-client --name front-cam --camera-id 0

# 두 번째 카메라를 다른 이름으로 (별도 터미널)
uv run nf-vision-camhub-client --name side-cam --camera-id 1

# FPS, 해상도, JPEG 품질 조절
uv run nf-vision-camhub-client --name hd-cam --fps 15 --width 1280 --height 720 --quality 90

# 원격 허브 서버에 연결
uv run nf-vision-camhub-client --host 192.168.0.10 --port 26200 --name remote-cam
```

CLI 인자:

| 옵션 | 타입 | 기본값 | 환경 변수 | 설명 |
| --- | --- | --- | --- | --- |
| `--host` | `str` | `127.0.0.1` | `NF_CAMHUB_HOST` | camhub 서버 주소 |
| `--port` | `int` | `26200` | `NF_CAMHUB_PORT` | camhub 서버 포트 |
| `--name` | `str` | `cam0` | `NF_CAMHUB_CLIENT_NAME` | 카메라 식별 이름. hub에서 이 이름으로 프레임을 관리한다 |
| `--camera-id` | `int` | `0` | `CAMERA_ID` | OpenCV 카메라 인덱스. `0`은 시스템 기본 카메라 |
| `--fps` | `float` | `10` | `NF_CAMHUB_CLIENT_FPS` | 목표 전송 FPS. 실제 FPS는 카메라/네트워크에 따라 낮아질 수 있다 |
| `--quality` | `int` | `80` | `NF_CAMHUB_CLIENT_QUALITY` | JPEG 인코딩 품질 (1-100). 높을수록 화질↑ 대역폭↑ |
| `--width` | `int` | `640` | `CAMERA_RESOLUTION` | 캡처 요청 너비 (px) |
| `--height` | `int` | `480` | `CAMERA_RESOLUTION` | 캡처 요청 높이 (px) |

일반적인 사용 흐름:

```bash
# 터미널 1: 허브 서버 시작
uv run nf-vision-camhub

# 터미널 2: 카메라 1 연결
uv run nf-vision-camhub-client --name front-cam

# 터미널 3: 카메라 2 연결
uv run nf-vision-camhub-client --name side-cam --camera-id 1

# 터미널 4: AI 클라이언트에서 프레임 조회 (Python 예시)
# NFCP로 VISION_GET_FRAME(5004) 요청, meta: {"name": "front-cam"}
```

### nf-asr-server (ASR 서버)

NFCP TCP 기반 canonical ASR 서버. `voiceFlow` transport와 `asrFlow` core를 조립한다.

```bash
# 기본 실행 (0.0.0.0:26100)
uv run nf-asr-server

# env 파일 경로 지정
uv run nf-asr-server --env ./my_config.env

# 주소/포트 오버라이드
uv run nf-asr-server --host 127.0.0.1 --port 30000
```

CLI 인자:

| 옵션 | 타입 | 기본값 | 환경 변수 | 설명 |
| --- | --- | --- | --- | --- |
| `--env` | `str` | `None` | - | env 파일 경로를 직접 지정. 생략 시 `.env` 자동 탐색 |
| `--host` | `str` | `None` | `NF_ASR_SERVER_HOST` | TCP 바인드 주소. 생략 시 env 값 또는 `0.0.0.0` |
| `--port` | `int` | `None` | `NF_ASR_SERVER_PORT` | TCP 리스닝 포트. 생략 시 env 값 또는 `26100` |

NFCP 커맨드:

| Command | Code | 설명 |
| --- | --- | --- |
| `PING` | 99 | 서비스 상태 확인 |
| `DESCRIBE` | 101 | 서비스 상세 정보 (지원 커맨드, 모델, streaming 여부) |
| `ASR_TRANSCRIBE` | 1001 | batch 음성 인식. meta: `{audio_format, samplerate, channels}`, data: audio bytes |
| `ASR_TRANSCRIBE_STREAM` | 1002 | native streaming. `NF_ASR_STREAM_ENABLED=true` 필요. meta `action`: `start`/`end`/생략(chunk) |

### nf-voice-mic-client (마이크 클라이언트)

로컬 마이크를 지정된 시간만큼 녹음한 뒤 NFCP `ASR_TRANSCRIBE` batch 요청을 보내고 결과를 출력한다. 시작 시 `PING`으로 서버 연결을 확인한다.

```bash
# 기본 실행 (127.0.0.1:26100, 4초 녹음)
uv run nf-voice-mic-client

# 전체 옵션 지정
uv run nf-voice-mic-client --host 192.168.0.10 --port 26100 --duration 6 --samplerate 16000 --channels 1 --device 2

# PING 건너뛰기
uv run nf-voice-mic-client --skip-ping --duration 3
```

CLI 인자:

| 옵션 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `--host` | `str` | `127.0.0.1` | ASR 서버 주소 |
| `--port` | `int` | `26100` | ASR 서버 포트 |
| `--duration` | `float` | `4.0` | 마이크 녹음 시간 (초) |
| `--samplerate` | `int` | `16000` | 녹음 샘플레이트 (Hz) |
| `--channels` | `int` | `1` | 오디오 채널 수 (1=모노, 2=스테레오) |
| `--device` | `int` | `None` | sounddevice 입력 장치 인덱스. 생략 시 시스템 기본 입력 |
| `--skip-ping` | flag | `false` | 설정 시 시작 PING을 건너뛴다 |

점검 순서:

```bash
# 터미널 1: ASR 서버 시작
uv run nf-asr-server

# 터미널 2: 마이크 클라이언트로 테스트
uv run nf-voice-mic-client --host 127.0.0.1 --port 26100 --duration 4
```

### nf-asr-chunk-realtime (Chunk ASR UI)

로컬 마이크 기반 chunk 실험 UI. Whisper(`ct2`, `hf_generate`, `hf_pipeline`)와 Qwen ASR(`qwen_transformers`)를 시험한다. CLI 인자 없이 `.env`만으로 동작한다.

```bash
uv run nf-asr-chunk-realtime
```

관련 환경 변수:

| 환경 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `ASRFLOW_STT_BACKEND` | `qwen_transformers` | backend 선택: `ct2`, `hf_generate`, `hf_pipeline`, `qwen_transformers` |
| `ASRFLOW_STT_MODEL` | `Qwen/Qwen3-ASR-0.6B` | HF model id 또는 alias |
| `ASRFLOW_STT_CHUNK_SEC` | `3.0` | chunk 길이 (초). 짧으면 반응↑ 정확도↓ |
| `ASRFLOW_STT_SAMPLERATE` | `16000` | 마이크 샘플레이트 |
| `ASRFLOW_STT_DEVICE` | `auto` | 추론 디바이스: `auto`, `cuda`, `cpu` |
| `ASRFLOW_STT_FP16` | `true` | FP16 추론 사용 여부 |
| `ASRFLOW_STT_LANGUAGE` | `auto` | 언어 코드: `auto`, `ko`, `en` 등 |

### nf-audiomi-asr-chunk-realtime (AudioMi Chunk UI)

audioMi 입력 기반 accumulate chunk 실험 UI. 누적 buffer 전체를 다시 추론하면서 suffix만 UI에 반영한다. CLI 인자 없이 `.env`만으로 동작한다.

```bash
uv run nf-audiomi-asr-chunk-realtime
```

관련 환경 변수 (`ASRFLOW_STT_*` 외 추가분):

| 환경 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `AUDIOMI_HOST` | `127.0.0.1` | audioMi 서버 주소 |
| `AUDIOMI_PORT` | `26070` | audioMi 서버 포트 |
| `AUDIOMI_CHECKCODE` | `20250918` | audioMi 인증 코드 |
| `ASR_STEP_S` | `1.5` | accumulate 추론 간격 (초) |
| `ASR_MAX_WINDOW_S` | `25` | 최대 누적 윈도우 길이 (초). 초과 시 앞부분 버린다 |

### nf-asr-stream-realtime (Native Streaming UI)

로컬 마이크 기반 native streaming 실험 UI. 현재 canonical stream 경로는 `Qwen ASR + qwen-asr transformers backend`다. CLI 인자 없이 `.env`만으로 동작한다.

```bash
uv run nf-asr-stream-realtime
```

관련 환경 변수:

| 환경 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `ASRFLOW_STREAM_MODEL` | `Qwen/Qwen3-ASR-0.6B` | streaming용 HF model id |
| `ASRFLOW_STREAM_LANGUAGE` | `auto` | 언어 코드 |
| `ASRFLOW_STREAM_CHUNK_SEC` | `2.0` | 오디오 chunk 길이 (초) |
| `ASRFLOW_STREAM_SAMPLERATE` | `16000` | 마이크 샘플레이트 |
| `ASRFLOW_STREAM_MAX_NEW_TOKENS` | `512` | 디코딩 최대 토큰 수 |
| `ASRFLOW_STREAM_UNFIXED_CHUNK_NUM` | `2` | 확정되지 않은 chunk 유지 수 |
| `ASRFLOW_STREAM_UNFIXED_TOKEN_NUM` | `5` | 확정되지 않은 토큰 유지 수 |

### nf-vision (Vision 샘플 런처)

vision 관련 sample을 메뉴로 선택하여 실행하는 런처. CLI 인자 없이 실행하면 대화형 메뉴가 나온다.

```bash
uv run nf-vision
```

메뉴 항목:

| 번호 | 모듈 | 설명 |
| --- | --- | --- |
| 1 | `camera.simple` | 기본 카메라 뷰어 |
| 2 | `camera.list_cameras` | 카메라 디바이스 목록 출력 |
| 3 | `face_detection.simple` | 얼굴 검출 |
| 4 | `pose.simple` | 포즈 감지 |
| 5 | `detect_test` | 전체 파이프라인 테스트 |

개별 sample 직접 실행:

```bash
uv run python -m visionflow.sample.camera.list_cameras
uv run python -m visionflow.sample.camera.simple --camera-id 0 --width 1280 --height 720
uv run python -m visionflow.sample.face_detection.simple --running-mode LIVE_STREAM --min-score 0.6
uv run python -m visionflow.sample.pose.simple --running-mode LIVE_STREAM
uv run python -m visionflow.sample.detect_test
```

### nf-vision-models-download (모델 다운로드)

vision sample이 요구하는 MediaPipe 모델 파일을 `models/` 디렉터리에 다운로드한다.

```bash
# 전체 모델 다운로드
uv run nf-vision-models-download

# 모델 목록 확인
uv run nf-vision-models-download --list

# 강제 재다운로드
uv run nf-vision-models-download --force

# 특정 모델만 선택
uv run nf-vision-models-download --only face-detector face-landmarker
uv run nf-vision-models-download --only pose-full pose-lite
```

CLI 인자:

| 옵션 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `--output-dir` | `path` | `./models` | 모델 파일 저장 디렉터리 |
| `--only` | `str[]` | 전체 | 다운로드할 asset 키 선택. 가능한 값: `face-detector`, `face-landmarker`, `pose-full`, `pose-lite` |
| `--force` | flag | `false` | 이미 존재하는 파일도 덮어쓴다 |
| `--list` | flag | `false` | 다운로드 가능한 asset 키 목록을 출력하고 종료 |

다운로드되는 파일:

| Asset 키 | 파일명 | 설명 |
| --- | --- | --- |
| `face-detector` | `blaze_face_short_range.tflite` | MediaPipe Face Detector (short range) |
| `face-landmarker` | `face_landmarker.task` | MediaPipe Face Landmarker |
| `pose-full` | `pose_landmarker.task` | MediaPipe Pose Landmarker (full) |
| `pose-lite` | `pose_landmarker_lite.task` | MediaPipe Pose Landmarker (lite) |

### nf-voice (Voice 샘플 런처)

voice source / device / sample app을 메뉴로 선택하여 실행하는 런처. CLI 인자 없이 대화형 메뉴로 동작한다.

```bash
uv run nf-voice
```

---

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

- 설정 키: `ASRFLOW_STT_*`
- backend: `ct2`, `hf_generate`, `hf_pipeline`, `qwen_transformers`
- `ASRFLOW_STT_MODEL_PATH`를 주면 로컬 모델 경로를 우선 사용

### 2. Native Stream

```text
MicrophoneSource
  -> TopicBus("audio/raw")
  -> StreamingAsrWorker
  -> QwenStreamingAsrProcessor
  -> TopicBus("text/asr")
  -> UI or NFCP stream response
```

- 설정 키: `ASRFLOW_STREAM_*`
- canonical model family: `Qwen/Qwen3-ASR-0.6B`
- `qwen_asr`의 streaming 로직을 transformers backend로 재현, `vLLM` 없이 동작

---

## NFCP 프로토콜

모든 서버 간 통신은 `common.protocols.nfcp` 기반 TCP binary framing을 사용한다.

```text
[Header 72 bytes] [meta JSON] [data bytes]
```

모든 서버는 아래 공통 커맨드를 지원한다:

| Command | Code | 역할 |
| --- | --- | --- |
| `PING` | 99 | 서비스 상태 확인 |
| `DESCRIBE` | 101 | 서비스 상세 정보 |

서비스별 커맨드:

| Service | Command | Code | 역할 |
| --- | --- | --- | --- |
| ASR | `ASR_TRANSCRIBE` | 1001 | batch 음성 인식 |
| ASR | `ASR_TRANSCRIBE_STREAM` | 1002 | native streaming 음성 인식 |
| Vision | `VISION_UPLOAD_FRAME` | 5003 | 카메라 프레임 업로드 |
| Vision | `VISION_GET_FRAME` | 5004 | 최신 프레임 조회 |
| Vision | `VISION_LIST_CAMERAS` | 5005 | 카메라 목록 |

새 서버를 추가할 때는 `AsrIngressServer` 또는 `CamHubServer`의 패턴을 따른다:
- `_dispatch` → `_handle_client` → `run` 구조
- `PING`/`DESCRIBE` 핸들러 기본 포함
- asyncio TCP + NFCP만으로 구현, HTTP 프레임워크 금지

---

## 환경 변수

`sample.env`를 복사하여 `.env`로 사용한다. 주요 설정:

### chunk / batch 계열 (`ASRFLOW_STT_*`)

```env
ASRFLOW_STT_BACKEND=qwen_transformers    # ct2 | hf_generate | hf_pipeline | qwen_transformers
ASRFLOW_STT_MODEL=Qwen/Qwen3-ASR-0.6B   # HF model id or alias
ASRFLOW_STT_MODEL_PATH=                  # 로컬 모델 경로 (optional)
ASRFLOW_STT_DEVICE=auto                  # auto | cuda | cpu
ASRFLOW_STT_FP16=true
ASRFLOW_STT_LANGUAGE=auto                # auto | ko | en | ...
ASRFLOW_STT_TASK=transcribe              # transcribe | translate
ASRFLOW_STT_CHUNK_SEC=3.0
ASRFLOW_STT_SAMPLERATE=16000
```

### native stream 계열 (`ASRFLOW_STREAM_*`)

```env
ASRFLOW_STREAM_MODEL=Qwen/Qwen3-ASR-0.6B
ASRFLOW_STREAM_LANGUAGE=auto
ASRFLOW_STREAM_CHUNK_SEC=2.0
ASRFLOW_STREAM_SAMPLERATE=16000
ASRFLOW_STREAM_MAX_NEW_TOKENS=512
NF_ASR_STREAM_ENABLED=true
```

### 서버 / 네트워크

```env
NF_ASR_SERVER_HOST=0.0.0.0
NF_ASR_SERVER_PORT=26100
NF_CAMHUB_HOST=0.0.0.0
NF_CAMHUB_PORT=26200
AUDIOMI_HOST=127.0.0.1
AUDIOMI_PORT=26070
```

### camhub 클라이언트

```env
NF_CAMHUB_CLIENT_NAME=cam0
NF_CAMHUB_CLIENT_FPS=10
NF_CAMHUB_CLIENT_QUALITY=80
```

### 카메라 / 디바이스

```env
CAMERA_ID=0
CAMERA_RESOLUTION=640x480
CAMERA_USE_DSHOW=true
MIC_DEVICE=0
MIC_SAMPLERATE=16000
```

---

## Layout

```text
src/
  common/
    contracts/        # packet, gateway contract
    protocols/        # NFCP binary protocol
    runtime/          # TopicBus, audio codec
    tools/            # model download 등
  neuroflow/
    app/              # composition root (asr_server, chunk/stream UI)
  visionflow/
    camhub/           # CamHubServer, FrameHub, camera client
    processors/       # face detector, pose landmarker
    sample/           # camera, face detection, pose sample
    sources/          # CameraSource
    workers/          # async vision workers
  voiceFlow/
    gateways/         # AsrIngressServer (NFCP TCP)
    sample/           # microphone client
    sources/          # microphone, audioMi source
    utils/
  asrFlow/
    bootstrap.py
    processors/       # MisoSttAsr, QwenStreamingAsr
    services/         # NFCP ASR handler
    workers/          # chunk, accumulate, streaming worker
    vendors/          # whisper, qwen_asr runtime
    utils/
```

상세 구조는 [`_forAI/architecture.md`](_forAI/architecture.md), 현재 인벤토리는 [`_forAI/inventory.md`](_forAI/inventory.md)를 보면 된다.
