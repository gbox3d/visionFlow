# NeuroFlow Common Protocol v1

## 1. 목적

이 문서는 `asrFlow`, `llmFlow`, `ttsFlow`, `backend`, `visionflow`까지 공통으로 사용할 TCP 기반 바이너리 프로토콜 초안이다.

기존 STT/TTS 프로토타입의 장점은 유지하되, 실제 서비스 운영에 맞게 다음을 강화한다.

- 모든 숫자 필드는 `little-endian`
- 모든 서버/gateway는 `PING` 필수
- `request_id`, `session_id` 기반 추적
- `ACK -> EVENT/PARTIAL -> RESULT/ERROR` 흐름 지원
- JSON 메타데이터와 바이너리 데이터 동시 전송 지원
- STT/LLM/TTS/Pipeline을 하나의 헤더 구조로 통일


## 2. 설계 원칙

### 2.1 유지할 것

- TCP 기반
- 단순한 고정 길이 헤더
- `PING=99` 유지
- 오디오 raw bytes 직접 전송 가능

### 2.2 개선할 것

- legacy의 `checkcode` 방식은 더 이상 표준으로 쓰지 않음
- 서비스/요청/상태를 분리해 표현
- 요청 본문을 `meta(JSON) + data(binary)`로 분리
- partial/progress/final result를 같은 프로토콜에서 처리

### 2.3 범위 밖

- 인증/권한
- TLS
- 압축 표준
- 영속 큐/브로커

위 항목은 프로토콜이 아니라 gateway 또는 인프라 계층에서 처리하는 것을 권장한다.


## 3. 전송 규칙

- 전송 계층: TCP
- 문자열 인코딩: UTF-8
- 숫자 필드 인코딩: `little-endian`
- 프레임 단위: `고정 64바이트 헤더 + meta bytes + data bytes`
- `meta`는 UTF-8 JSON
- `data`는 raw binary

권장 연결 정책:

- 클라이언트는 keep-alive 가능한 장기 연결 사용
- 서버는 요청당 하나의 terminal frame을 반드시 반환
- 장기 작업은 `ACK`를 먼저 보내고 이후 `EVENT` 또는 `RESULT`를 보낸다


## 4. 헤더 정의

### 4.1 고정 크기

- 헤더 크기: `64 bytes`
- Python `struct` 포맷: `"<4sBBHBBHHHQQIIIIQII"`

### 4.2 필드 레이아웃

| Offset | Size | Type | Field | 설명 |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | `char[4]` | `magic` | 항상 `NFCP` |
| 4 | 1 | `u8` | `version_major` | 메이저 버전 |
| 5 | 1 | `u8` | `version_minor` | 마이너 버전 |
| 6 | 2 | `u16` | `header_size` | 항상 `64` |
| 8 | 1 | `u8` | `message_type` | 요청/응답 종류 |
| 9 | 1 | `u8` | `service_type` | ASR/LLM/TTS/PIPELINE/VISION |
| 10 | 2 | `u16` | `command` | 요청 코드 |
| 12 | 2 | `u16` | `status_code` | 상태 코드 |
| 14 | 2 | `u16` | `flags` | 옵션 비트 |
| 16 | 8 | `u64` | `session_id` | 세션 식별자 |
| 24 | 8 | `u64` | `request_id` | 요청 식별자 |
| 32 | 4 | `u32` | `sequence_no` | frame 순번 |
| 36 | 4 | `u32` | `meta_length` | meta JSON 길이 |
| 40 | 4 | `u32` | `data_length` | binary data 길이 |
| 44 | 4 | `u32` | `timeout_ms` | 요청 타임아웃 |
| 48 | 8 | `u64` | `timestamp_ms` | Unix epoch ms |
| 56 | 4 | `u32` | `body_crc32` | body CRC32, 사용 안 하면 0 |
| 60 | 4 | `u32` | `reserved` | 현재 0 |

### 4.3 Python 예시

```python
import struct

HEADER_STRUCT = struct.Struct("<4sBBHBBHHHQQIIIIQII")
HEADER_SIZE = HEADER_STRUCT.size  # 64
```

### 4.4 C/C++ 예시

```c
#pragma pack(push, 1)
typedef struct {
    char     magic[4];       // "NFCP"
    uint8_t  version_major;
    uint8_t  version_minor;
    uint16_t header_size;
    uint8_t  message_type;
    uint8_t  service_type;
    uint16_t command;
    uint16_t status_code;
    uint16_t flags;
    uint64_t session_id;
    uint64_t request_id;
    uint32_t sequence_no;
    uint32_t meta_length;
    uint32_t data_length;
    uint32_t timeout_ms;
    uint64_t timestamp_ms;
    uint32_t body_crc32;
    uint32_t reserved;
} NFCPHeader;
#pragma pack(pop)
```


## 5. Body 구조

헤더 뒤에는 아래 순서로 body가 온다.

```text
[meta JSON bytes][data bytes]
```

- `meta_length == 0`이면 meta 없음
- `data_length == 0`이면 data 없음
- JSON은 반드시 UTF-8
- JSON 최상위는 object 권장

예:

```json
{
  "audio_format": "wav",
  "samplerate": 16000,
  "language": "ko",
  "partial": true
}
```


## 6. 공통 enum

### 6.1 `message_type`

| 값 | 이름 | 설명 |
| ---: | --- | --- |
| `1` | `REQUEST` | 작업 요청 |
| `2` | `ACK` | 요청 수락/큐잉 확인 |
| `3` | `EVENT` | 진행률/partial/stream 이벤트 |
| `4` | `RESULT` | 최종 성공 결과 |
| `5` | `ERROR` | 최종 실패 결과 |
| `6` | `CANCEL` | 작업 취소 요청 |

### 6.2 `service_type`

| 값 | 이름 |
| ---: | --- |
| `0` | `COMMON` |
| `1` | `ASR` |
| `2` | `LLM` |
| `3` | `TTS` |
| `4` | `PIPELINE` |
| `5` | `VISION` |

### 6.3 `flags`

| 비트 | 이름 | 설명 |
| ---: | --- | --- |
| `0x0001` | `ACK_REQUIRED` | ACK를 기대함 |
| `0x0002` | `MORE_FRAMES` | 같은 요청의 추가 frame이 이어짐 |
| `0x0004` | `END_OF_STREAM` | 스트림 마지막 frame |
| `0x0008` | `BODY_CRC32_PRESENT` | `body_crc32` 사용 |
| `0x0010` | `META_COMPRESSED` | 예약 |
| `0x0020` | `DATA_COMPRESSED` | 예약 |


## 7. 상태 코드

### 7.1 정상/진행 상태

| 값 | 이름 | 의미 |
| ---: | --- | --- |
| `0` | `NONE` | 요청 frame에서 사용 |
| `100` | `ACCEPTED` | 요청 수락 |
| `101` | `RUNNING` | 작업 진행 중 |
| `102` | `PARTIAL` | 부분 결과 |
| `103` | `COMPLETED` | 최종 성공 |
| `140` | `CANCELLED` | 요청 취소 |
| `141` | `REJECTED` | 서버가 작업 거절 |

### 7.2 오류 상태

| 값 | 이름 | 의미 |
| ---: | --- | --- |
| `200` | `BAD_HEADER` | 헤더 파싱 실패 |
| `201` | `BAD_REQUEST` | 필수 필드 누락/잘못된 요청 |
| `202` | `BAD_PAYLOAD` | body 형식 오류 |
| `203` | `UNSUPPORTED_COMMAND` | 지원하지 않는 command |
| `204` | `UNSUPPORTED_MEDIA` | 지원하지 않는 포맷 |
| `205` | `NOT_READY` | 모델/서비스 준비 안 됨 |
| `206` | `BUSY` | 서버 과부하/큐 포화 |
| `207` | `TIMEOUT` | 처리 시간 초과 |
| `208` | `INTERNAL_ERROR` | 내부 예외 |
| `209` | `UNAUTHORIZED` | 인증 실패 |
| `210` | `CHECKSUM_MISMATCH` | CRC32 불일치 |
| `211` | `TOO_LARGE` | 허용 크기 초과 |


## 8. Command 코드

### 8.1 Common

| 값 | 이름 | 설명 |
| ---: | --- | --- |
| `99` | `PING` | 모든 서버 필수 |
| `100` | `HEALTH` | 상세 상태 질의 |
| `101` | `DESCRIBE` | capabilities 조회 |
| `102` | `SERVER_INFO` | 버전/호스트/포트/uptime 등 서버 메타 조회 |

### 8.2 ASR

| 값 | 이름 | 설명 |
| ---: | --- | --- |
| `1001` | `ASR_TRANSCRIBE` | 단건 음성 -> 텍스트 |
| `1002` | `ASR_TRANSCRIBE_STREAM` | 스트림/청크 입력 |
| `1003` | `ASR_CLEAR_BUFFER` | 서버 내 ASR 스트리밍 버퍼/세션 상태 초기화 |

### 8.3 LLM

| 값 | 이름 | 설명 |
| ---: | --- | --- |
| `2001` | `LLM_GENERATE` | 일반 생성 |
| `2002` | `LLM_SUMMARIZE` | 메모리/대화 요약 |

### 8.4 TTS

| 값 | 이름 | 설명 |
| ---: | --- | --- |
| `3001` | `TTS_SYNTHESIZE` | 텍스트 -> 음성 |

### 8.5 Pipeline

| 값 | 이름 | 설명 |
| ---: | --- | --- |
| `9001` | `PIPELINE_RUN` | `ASR -> LLM -> TTS` 통합 실행 |

### 8.6 Vision

| 값 | 이름 | 설명 |
| ---: | --- | --- |
| `5001` | `VISION_PROCESS_FRAME` | 예약 |
| `5002` | `VISION_GET_STATE` | 예약 |
| `5003` | `VISION_UPLOAD_FRAME` | 카메라 클라이언트가 JPEG 프레임 업로드 |
| `5004` | `VISION_GET_FRAME` | 최신 카메라 프레임 조회 |
| `5005` | `VISION_LIST_CAMERAS` | 등록된 카메라 목록 조회 |


## 9. 필수 서버 동작 규칙

### 9.1 모든 서버는 반드시 `PING`를 지원한다

필수 대상:

- `asrFlow` gateway
- `llmFlow` gateway
- `ttsFlow` gateway
- `backend` gateway
- 추후 `visionflow` gateway

### 9.2 `PING` 응답 규칙

- 모델 로딩 여부와 상관없이 응답 가능해야 한다
- `request_id`, `session_id`를 그대로 echo 해야 한다
- 응답은 `message_type=RESULT`, `command=99`, `status_code=103` 권장
- meta에는 최소 아래 정보 포함 권장

```json
{
  "service": "asrFlow",
  "ready": true,
  "version": "0.2.1",
  "uptime_ms": 154233,
  "active_jobs": 2
}
```

### 9.3 ACK 규칙

장기 작업 서버는 다음 규칙을 따른다.

- 처리 시간이 짧으면 `RESULT`를 바로 반환 가능
- 처리 시간이 길면 `ACK`를 먼저 반환
- `ACK`는 body 없이 보내도 됨
- `ACK`의 `status_code`는 `ACCEPTED`

권장 ACK deadline:

- 요청 body를 모두 수신한 후 `1000ms` 이내

### 9.4 terminal frame 규칙

요청 하나당 terminal frame은 정확히 하나여야 한다.

가능한 terminal frame:

- `RESULT + COMPLETED`
- `RESULT + CANCELLED`
- `ERROR + <오류 상태>`

### 9.5 sequence 규칙

- 첫 `ACK`는 `sequence_no=0`
- 이후 `EVENT`는 `1, 2, 3...`
- 최종 `RESULT` 또는 `ERROR`는 마지막 번호 사용


## 10. 서비스별 meta schema 초안

### 10.1 `PING`

요청 meta:

```json
{
  "echo": "optional",
  "want_details": true
}
```

응답 meta:

```json
{
  "service": "ttsFlow",
  "service_type": 3,
  "ready": true,
  "model_loaded": true,
  "version": "0.2.1",
  "uptime_ms": 21002,
  "active_jobs": 0,
  "echo": "optional"
}
```

### 10.2 `ASR_TRANSCRIBE`

요청:

- `service_type=ASR`
- `command=1001`
- data = 오디오 raw bytes

권장 meta:

```json
{
  "audio_format": "wav",
  "samplerate": 16000,
  "channels": 1,
  "language": "ko",
  "task": "transcribe",
  "partial": true,
  "timestamps": true,
  "backend": "ct2",
  "model": "large-v3"
}
```

partial event meta 예시:

```json
{
  "text": "안녕하세요",
  "language": "ko",
  "is_final": false
}
```

final result meta 예시:

```json
{
  "text": "안녕하세요. 무엇을 도와드릴까요?",
  "language": "ko",
  "segments": [
    {"start": 0.0, "end": 1.2, "text": "안녕하세요."}
  ]
}
```

### 10.3 `LLM_GENERATE`

요청:

- `service_type=LLM`
- `command=2001`
- 보통 data 없음

권장 meta:

```json
{
  "input_text": "사용자 질문",
  "model": "gemma3:4b",
  "prompt_id": "miracle",
  "temperature": 0.7,
  "top_p": 0.9,
  "memory_policy": "session",
  "session_options": {
    "max_messages": 20,
    "auto_summary": true
  }
}
```

stream event meta 예시:

```json
{
  "delta": "안녕하세요",
  "is_final": false
}
```

final result meta 예시:

```json
{
  "text": "안녕하세요. 어떤 도움이 필요하신가요?",
  "finish_reason": "stop"
}
```

### 10.4 `TTS_SYNTHESIZE`

요청:

- `service_type=TTS`
- `command=3001`

권장 meta:

```json
{
  "text": "안녕하세요",
  "audio_format": "wav",
  "language": "ko",
  "speed": 1.0,
  "speaker_id": 0
}
```

응답:

- meta = 출력 포맷/길이/엔진 처리 정보
- data = WAV bytes

result meta 예시:

```json
{
  "audio_format": "wav",
  "samplerate": 22050,
  "channels": 1,
  "duration_ms": 1840,
  "processor_meta": {
    "provider": "CPUExecutionProvider",
    "inference_ms": 42,
    "rtf": 0.03
  }
}
```

### 10.5 `PIPELINE_RUN`

요청:

- `service_type=PIPELINE`
- `command=9001`
- 입력이 오디오면 data에 audio bytes
- 입력이 텍스트면 data 없이 meta만 사용

권장 meta:

```json
{
  "input_type": "audio",
  "return_intermediate": true,
  "asr": {
    "audio_format": "wav",
    "samplerate": 16000,
    "language": "ko"
  },
  "llm": {
    "prompt_id": "miracle",
    "model": "gemma3:4b"
  },
  "tts": {
    "voice": "kr_female_01",
    "format": "mp3"
  }
}
```

final result meta 예시:

```json
{
  "asr_text": "사용자 발화",
  "answer_text": "응답 텍스트",
  "audio_format": "mp3"
}
```


## 11. 스트리밍 규칙

### 11.1 요청 스트리밍

대용량/실시간 입력이 필요하면 같은 `request_id`로 여러 frame을 보낼 수 있다.

규칙:

- 중간 frame에는 `MORE_FRAMES`
- 마지막 frame에는 `END_OF_STREAM`
- `ASR_TRANSCRIBE_STREAM` 같은 streaming command에서만 권장

### 11.2 응답 스트리밍

partial/token/audio chunk가 필요하면 서버는 `EVENT`를 여러 번 보내고 마지막에 `RESULT`로 종료한다.


## 12. 크기 제한 권장값

서비스 안전성을 위해 아래 권장값을 둔다.

- `meta_length <= 64 KiB`
- `data_length <= 64 MiB` per frame
- 요청 전체 timeout 기본값: `30000ms`

초과 시 서버는 `ERROR + TOO_LARGE` 또는 `ERROR + TIMEOUT`을 반환할 수 있다.


## 13. 구현 체크리스트

### 13.1 서버

- `NFCP` magic 검사
- 버전 검사
- header_size 검사
- little-endian 파싱
- `PING=99` 처리
- body 길이 검사
- 선택적 CRC32 검사
- ACK/RESULT/ERROR lifecycle 보장
- terminal frame 정확히 1회 보장

### 13.2 클라이언트

- `request_id`를 요청마다 고유하게 생성
- timeout과 재시도 정책 분리
- `ACK`와 terminal frame을 구분 처리
- `EVENT/PARTIAL` 수신 가능하게 구현
- 연결 종료를 실패로 처리


## 14. Legacy 프로토콜과의 관계

### 14.1 유지

- `PING=99`
- raw audio / raw audio bytes 응답 개념
- 단순 TCP framing 철학

### 14.2 변경

- `big-endian` -> `little-endian`
- `checkcode` -> `magic + version`
- 서비스별 서로 다른 헤더 -> `공통 64바이트 헤더`
- payload 단일 구조 -> `meta + data` 이중 구조

legacy 문서는 별도로 유지한다.

- `asr_tcp_legacy.md`
- `tts_tcp_legacy.md`


## 15. 1차 구현 권장 범위

처음부터 모든 기능을 다 넣지 말고 아래만 먼저 구현하는 것이 좋다.

1. `PING`
2. `ASR_TRANSCRIBE`
3. `LLM_GENERATE`
4. `TTS_SYNTHESIZE`
5. `PIPELINE_RUN`
6. `ACK`, `RESULT`, `ERROR`

그리고 이후 필요하면 아래를 확장한다.

- partial/event streaming
- request streaming
- `HEALTH`
- `DESCRIBE`
- `VISION` commands


## 16. 한 줄 결론

이 프로토콜은 기존 STT/TTS 프로토타입의 단순함을 유지하면서도, 실제 서비스 운영에 필요한 `little-endian`, `mandatory ping`, `meta+data`, `request_id/session_id`, `ACK/EVENT/RESULT`를 공통화한 실용형 TCP 프로토콜 초안이다.
