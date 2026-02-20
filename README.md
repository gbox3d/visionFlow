# VisionFlow

Realtime vision pipeline (camera → processors → render)

OpenCV 카메라 캡처, MediaPipe 추론, Pygame 렌더링을 **TopicBus** 기반 pub/sub 파이프라인으로 연결하는 프레임워크입니다.

## Architecture

```
CameraSource ──publish──▶ TopicBus ──subscribe──▶ Workers ──publish──▶ TopicBus
                              ▲                                           │
                              └─────────── Application (Render) ◀─────────┘
```

- **Main Thread** : Pygame 렌더링만 담당
- **Camera Thread** : 프레임 캡처만 담당
- **Inference Thread** : 모델 추론만 담당 (IMAGE/VIDEO 동기 또는 LIVE_STREAM 비동기)
- **TopicBus** : 스레드 안전 pub/sub 메시지 버스, 버전 기반 구독

## Install

```bash
# uv (권장)
uv sync

# pip
pip install -e .
```

## Quick Start

### 런처로 샘플 선택 실행

```bash
uv run visionflow
```

번호를 입력하면 해당 샘플이 실행됩니다.

```
==================================================
  VisionFlow Sample Launcher
==================================================

  [Camera]
    1) camera.simple                  - 기본 카메라 뷰어
    2) camera.dual_cam                - 듀얼 카메라
    3) camera.list_cameras            - 카메라 디바이스 목록
    4) camera.list_resolutions        - 카메라 해상도 목록

  [Face Detection]
    5) face_detection.simple          - 얼굴 검출
    6) face_detection.simple_landmark - 얼굴 랜드마크
    7) face_detection.transform_3d    - 3D 얼굴 변환

  [Pose]
    8) pose.simple                    - 포즈 감지

  [Full Pipeline]
    9) detect_test                    - 전체 파이프라인 테스트

    0) 종료
```

### 개별 스크립트 직접 실행

`[project.scripts]`로 등록된 커맨드를 사용합니다.

```bash
# Camera
uv run vf-camera-simple
uv run vf-camera-simple --camera-id 0 --width 1280 --height 720
uv run vf-camera-dual
uv run vf-camera-list
uv run vf-camera-resolutions --camera-id 0

# Face Detection
uv run vf-face-detect --running-mode LIVE_STREAM --min-score 0.6
uv run vf-face-detect --running-mode IMAGE
uv run vf-face-landmark
uv run vf-face-3d

# Pose
uv run vf-pose --running-mode LIVE_STREAM
uv run vf-pose --running-mode IMAGE

# Full Pipeline Test
uv run vf-detect-test
```

## Project Structure

```
src/visionflow/
  __init__.py
  main.py                  # 인터랙티브 샘플 런처
  pipeline/
    bus.py                 # TopicBus - 스레드 안전 pub/sub 메시지 버스
    packet.py              # FramePacket 데이터 클래스
  sources/
    camera_source.py       # OpenCV 카메라 캡처 (자동 재연결)
  processors/
    face_detector.py       # FaceDetector (동기, IMAGE/VIDEO)
    pose_landmarker.py     # PoseLandmarker (동기, IMAGE/VIDEO)
  workers/
    sync_inference_worker.py              # 동기 추론 워커
    mp_face_detector_async_worker.py      # Face Detector (LIVE_STREAM)
    mp_face_landmarker_async_worker.py    # Face Landmarker (LIVE_STREAM)
    mp_pose_landmarker_async_worker.py    # Pose Landmarker (LIVE_STREAM)
  utils/
    draw_utils.py          # 렌더링 유틸 (DrawUtils, Colors)
    image.py               # 이미지 변환 (cv2 <-> pygame)
    face_geometry.py       # 얼굴 3D 기하 계산
    etc.py                 # 기타 유틸 (resource_path 등)
  sample/
    detect_test.py         # 전체 파이프라인 테스트
    camera/
      simple.py            # 기본 카메라 뷰어
      dual_cam.py          # 듀얼 카메라
      list_cameras.py      # 카메라 디바이스 목록
      list_resolutions.py  # 해상도 목록
    face_detection/
      simple.py            # 얼굴 검출
      simple_landmark.py   # 얼굴 랜드마크
      transform_3d.py      # 3D 변환
    pose/
      simple.py            # 포즈 감지
```

## Scripts Reference

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `visionflow` | `visionflow.main:main` | 인터랙티브 샘플 런처 |
| `vf-camera-simple` | `visionflow.sample.camera.simple:main` | 기본 카메라 뷰어 |
| `vf-camera-dual` | `visionflow.sample.camera.dual_cam:main` | 듀얼 카메라 |
| `vf-camera-list` | `visionflow.sample.camera.list_cameras:main` | 카메라 디바이스 목록 |
| `vf-camera-resolutions` | `visionflow.sample.camera.list_resolutions:main` | 해상도 목록 |
| `vf-face-detect` | `visionflow.sample.face_detection.simple:main` | 얼굴 검출 |
| `vf-face-landmark` | `visionflow.sample.face_detection.simple_landmark:main` | 얼굴 랜드마크 |
| `vf-face-3d` | `visionflow.sample.face_detection.transform_3d:main` | 3D 얼굴 변환 |
| `vf-pose` | `visionflow.sample.pose.simple:main` | 포즈 감지 |
| `vf-detect-test` | `visionflow.sample.detect_test:main` | 전체 파이프라인 테스트 |

## Requirements

- Python >= 3.11
- MediaPipe >= 0.10.32
- OpenCV >= 4.8
- Pygame >= 2.5

## voiceFlow ASR (miso_stt)

`voiceFlow` ASR는 `miso_stt`를 벤더링해 3개 백엔드를 지원합니다.

- `ct2`
- `hf_generate`
- `hf_pipeline`

실행:

```bash
uv run python -m voiceFlow.sample.asr_realtime
uv run python -m voiceFlow.sample.audiomi_asr_realtime
```

UI에서 다음을 선택할 수 있습니다.
- backend
- model size
- model path (optional)
- accumulate step / max window (audiomi 샘플)

모델 지정 우선순위:
1. model path
2. model name(alias/HF ID)

주요 환경 변수:
- `VOICEFLOW_STT_TEMPERATURE` (기본 `0.0`)
- `VOICEFLOW_STT_CT2_VAD_FILTER` (기본 `false`)
- `ENABLE_LOGGING` (`true`일 때만 `logs/` 저장)

`audiomi_asr_realtime` 누적 모드:
- ingest/infer 스레드 분리로 수신과 추론 경로 분리
- `step_s` 단위 청크 큐 누적 후 전체 버퍼 추론
- `max_window_s` 초과 시 앞 청크 트림
- 워밍업 진행률(`text/asr_status`)을 UI progress bar로 표시
- 상태줄에 queue/infer 시간 표시
- 모델명 + `cache root(abs)` 표시
- 로깅 사용 시 추론 텍스트/오디오 쌍 저장(`logs/*.txt`, `logs/*.wav`)

CUDA 정책:
- PyTorch CUDA wheel 경로(`pytorch-cu128`)를 사용합니다.
- `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` 직접 의존은 제거했습니다.

운영 가이드:
- 의존성/락 파일 반영 후 `uv sync`로 환경을 재동기화하세요.
- 락 업데이트가 필요하면 `uv lock` 후 `uv sync`를 권장합니다.

HF LoRA 주의:
- adapter-only 경로는 지원하지 않습니다.
- 병합된 checkpoint 디렉토리를 model path로 지정하세요.

통합 데모(`main.py`, pygame) 해상도 튜닝:

```env
# 우선 사용 (권장): WxH 포맷
CAMERA_RESOLUTION=640x480

# fallback (CAMERA_RESOLUTION이 비어있을 때)
CAMERA_WIDTH=640
CAMERA_HEIGHT=480

# detection counts
MAX_FACE=1
MAX_POSE=1
```

`face + pose` 동시 추론에서는 해상도를 높일수록 FPS가 크게 떨어질 수 있습니다.

백엔드 가이드:
- `hf_generate`, `hf_pipeline`: LoRA/파인튜닝 확장 대비 권장
- `ct2`: 빠른 추론에 유리, GPU 초기화 실패 시 CPU fallback이 자동 적용될 수 있음

## Keyboard Controls (detect_test)

| Key | Action |
|-----|--------|
| `1` | Face Detection 토글 |
| `2` | Face Landmark 토글 |
| `3` | Pose Landmark 토글 |
| `4` | 카메라 전환 (0 <-> 1) |
| `H` | 도움말 토글 |
| `ESC` | 종료 |
