# Unity NFCP TCP 연동 가이드

> 대상
> `nf-asr-server`, `nf-tts-server`를 Unity/C#에서 TCP로 직접 호출하는 개발자용 문서다.

## 목차

1. [서비스 주소](#1-서비스-주소)
2. [NFCP frame 구조](#2-nfcp-frame-구조)
3. [명령 코드](#3-명령-코드)
4. [Unity C# 공통 NFCP 클라이언트](#4-unity-c-공통-nfcp-클라이언트)
5. [ASR TCP 사용법](#5-asr-tcp-사용법)
6. [Unity C# ASR 예제](#6-unity-c-asr-예제)
7. [TTS TCP 사용법](#7-tts-tcp-사용법)
8. [Unity C# TTS TCP 예제](#8-unity-c-tts-tcp-예제)

## 1. 서비스 주소

| 서비스 | 주소 | 포트 | command |
| --- | --- | ---: | --- |
| `nf-asr-server` | `192.168.4.218` | `21861` | `ASR_TRANSCRIBE(1001)` |
| `nf-tts-server` | `192.168.4.218` | `26120` | `TTS_SYNTHESIZE(3001)` |

## 2. NFCP frame 구조

```text
[64 bytes header][meta JSON bytes][data bytes]
```

기본 규칙:

- byte order는 little-endian이다.
- `magic`은 ASCII `NFCP`다.
- `meta`는 UTF-8 JSON object다.
- `data`는 WAV, PCM 등 raw binary다.
- 요청은 `message_type=REQUEST(1)`로 보낸다.
- 성공 응답은 보통 `message_type=RESULT(4)`, `status_code=COMPLETED(103)`이다.
- 실패 응답은 `message_type=ERROR(5)`이고 meta에 `error` 문자열이 들어간다.

헤더:

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | bytes | magic = `NFCP` |
| 4 | 1 | u8 | version_major = `1` |
| 5 | 1 | u8 | version_minor = `0` |
| 6 | 2 | u16 | header_size = `64` |
| 8 | 1 | u8 | message_type |
| 9 | 1 | u8 | service_type |
| 10 | 2 | u16 | command |
| 12 | 2 | u16 | status_code |
| 14 | 2 | u16 | flags |
| 16 | 8 | u64 | session_id |
| 24 | 8 | u64 | request_id |
| 32 | 4 | u32 | sequence_no |
| 36 | 4 | u32 | meta_length |
| 40 | 4 | u32 | data_length |
| 44 | 4 | u32 | timeout_ms |
| 48 | 8 | u64 | timestamp_ms |
| 56 | 4 | u32 | body_crc32, 현재 보통 `0` |
| 60 | 4 | u32 | reserved = `0` |

## 3. 명령 코드

| 종류 | 값 | 이름 |
| --- | ---: | --- |
| message_type | `1` | `REQUEST` |
| message_type | `4` | `RESULT` |
| message_type | `5` | `ERROR` |
| service_type | `0` | `COMMON` |
| service_type | `1` | `ASR` |
| service_type | `3` | `TTS` |
| command | `99` | `PING` |
| command | `101` | `DESCRIBE` |
| command | `102` | `SERVER_INFO` |
| command | `1001` | `ASR_TRANSCRIBE` |
| command | `3001` | `TTS_SYNTHESIZE` |
| status | `103` | `COMPLETED` |

## 4. Unity C# 공통 NFCP 클라이언트

`Assets/Scripts/NeuroFlow/NfcpClient.cs`처럼 추가한다.

```csharp
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

public static class Nfcp
{
    public const byte MessageRequest = 1;
    public const byte MessageResult = 4;
    public const byte MessageError = 5;

    public const byte ServiceAsr = 1;
    public const byte ServiceTts = 3;

    public const ushort CommandPing = 99;
    public const ushort CommandDescribe = 101;
    public const ushort CommandAsrTranscribe = 1001;
    public const ushort CommandTtsSynthesize = 3001;
}

public sealed class NfcpFrame
{
    public byte MessageType;
    public byte ServiceType;
    public ushort Command;
    public ushort StatusCode;
    public ulong SessionId;
    public ulong RequestId;
    public uint SequenceNo;
    public string MetaJson = "{}";
    public byte[] Data = Array.Empty<byte>();

    public bool IsError { get { return MessageType == Nfcp.MessageError; } }
}

public static class NfcpClient
{
    public static async Task<NfcpFrame> SendAsync(
        string host,
        int port,
        byte serviceType,
        ushort command,
        string metaJson,
        byte[] data,
        uint timeoutMs = 30000)
    {
        using (var client = new TcpClient())
        {
            await client.ConnectAsync(host, port);

            using (NetworkStream stream = client.GetStream())
            {
                byte[] metaBytes = Encoding.UTF8.GetBytes(string.IsNullOrEmpty(metaJson) ? "{}" : metaJson);
                byte[] dataBytes = data ?? Array.Empty<byte>();
                ulong requestId = (ulong)DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

                byte[] header = BuildHeader(
                    Nfcp.MessageRequest,
                    serviceType,
                    command,
                    0,
                    0,
                    0,
                    requestId,
                    1,
                    (uint)metaBytes.Length,
                    (uint)dataBytes.Length,
                    timeoutMs);

                await stream.WriteAsync(header, 0, header.Length);
                if (metaBytes.Length > 0) await stream.WriteAsync(metaBytes, 0, metaBytes.Length);
                if (dataBytes.Length > 0) await stream.WriteAsync(dataBytes, 0, dataBytes.Length);

                return await ReadFrameAsync(stream);
            }
        }
    }

    public static async Task<NfcpFrame> PingAsync(string host, int port, byte serviceType)
    {
        return await SendAsync(host, port, serviceType, Nfcp.CommandPing, "{}", Array.Empty<byte>());
    }

    private static async Task<NfcpFrame> ReadFrameAsync(NetworkStream stream)
    {
        byte[] header = await ReadExactAsync(stream, 64);
        uint metaLength = ReadU32(header, 36);
        uint dataLength = ReadU32(header, 40);

        byte[] metaBytes = metaLength > 0 ? await ReadExactAsync(stream, checked((int)metaLength)) : Array.Empty<byte>();
        byte[] dataBytes = dataLength > 0 ? await ReadExactAsync(stream, checked((int)dataLength)) : Array.Empty<byte>();

        return new NfcpFrame
        {
            MessageType = header[8],
            ServiceType = header[9],
            Command = ReadU16(header, 10),
            StatusCode = ReadU16(header, 12),
            SessionId = ReadU64(header, 16),
            RequestId = ReadU64(header, 24),
            SequenceNo = ReadU32(header, 32),
            MetaJson = Encoding.UTF8.GetString(metaBytes),
            Data = dataBytes
        };
    }

    private static async Task<byte[]> ReadExactAsync(NetworkStream stream, int length)
    {
        byte[] buffer = new byte[length];
        int offset = 0;
        while (offset < length)
        {
            int read = await stream.ReadAsync(buffer, offset, length - offset);
            if (read <= 0) throw new Exception("NFCP connection closed");
            offset += read;
        }
        return buffer;
    }

    private static byte[] BuildHeader(
        byte messageType,
        byte serviceType,
        ushort command,
        ushort statusCode,
        ushort flags,
        ulong sessionId,
        ulong requestId,
        uint sequenceNo,
        uint metaLength,
        uint dataLength,
        uint timeoutMs)
    {
        byte[] b = new byte[64];
        Encoding.ASCII.GetBytes("NFCP").CopyTo(b, 0);
        b[4] = 1;
        b[5] = 0;
        WriteU16(b, 6, 64);
        b[8] = messageType;
        b[9] = serviceType;
        WriteU16(b, 10, command);
        WriteU16(b, 12, statusCode);
        WriteU16(b, 14, flags);
        WriteU64(b, 16, sessionId);
        WriteU64(b, 24, requestId);
        WriteU32(b, 32, sequenceNo);
        WriteU32(b, 36, metaLength);
        WriteU32(b, 40, dataLength);
        WriteU32(b, 44, timeoutMs);
        WriteU64(b, 48, (ulong)DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());
        WriteU32(b, 56, 0);
        WriteU32(b, 60, 0);
        return b;
    }

    private static ushort ReadU16(byte[] b, int o)
    {
        return (ushort)(b[o] | (b[o + 1] << 8));
    }

    private static uint ReadU32(byte[] b, int o)
    {
        return (uint)(b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24));
    }

    private static ulong ReadU64(byte[] b, int o)
    {
        ulong v = 0;
        for (int i = 7; i >= 0; i--) v = (v << 8) | b[o + i];
        return v;
    }

    private static void WriteU16(byte[] b, int o, ushort v)
    {
        b[o] = (byte)v;
        b[o + 1] = (byte)(v >> 8);
    }

    private static void WriteU32(byte[] b, int o, uint v)
    {
        b[o] = (byte)v;
        b[o + 1] = (byte)(v >> 8);
        b[o + 2] = (byte)(v >> 16);
        b[o + 3] = (byte)(v >> 24);
    }

    private static void WriteU64(byte[] b, int o, ulong v)
    {
        for (int i = 0; i < 8; i++) b[o + i] = (byte)(v >> (8 * i));
    }
}
```

## 5. ASR TCP 사용법

접속 정보:

```text
host: 192.168.4.218
port: 21861
service_type: ASR(1)
command: ASR_TRANSCRIBE(1001)
```

WAV 요청 meta:

```json
{
  "audio_format": "wav",
  "samplerate": 16000,
  "channels": 1,
  "language": "ko",
  "task": "transcribe"
}
```

성공 응답 meta:

```json
{
  "text": "안녕하세요. 안내를 시작합니다.",
  "language": "ko",
  "segments": [],
  "samplerate": 16000,
  "audio_format": "wav"
}
```

## 6. Unity C# ASR 예제

```csharp
using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

[Serializable]
public class AsrResultMeta
{
    public string text;
    public string language;
    public int samplerate;
    public string audio_format;
}

public class NeuroFlowAsrExample : MonoBehaviour
{
    private const string Host = "192.168.4.218";
    private const int AsrPort = 21861;

    public async void TranscribeStreamingAssetsWav()
    {
        string wavPath = Path.Combine(Application.streamingAssetsPath, "sample_ko.wav");
        await TranscribeWavFile(wavPath);
    }

    public async Task TranscribeWavFile(string wavPath)
    {
        byte[] wavBytes = File.ReadAllBytes(wavPath);
        string metaJson =
            "{\"audio_format\":\"wav\",\"samplerate\":16000,\"channels\":1,\"language\":\"ko\",\"task\":\"transcribe\"}";

        NfcpFrame response = await NfcpClient.SendAsync(
            Host,
            AsrPort,
            Nfcp.ServiceAsr,
            Nfcp.CommandAsrTranscribe,
            metaJson,
            wavBytes,
            timeoutMs: 60000);

        if (response.IsError)
        {
            Debug.LogError("ASR error: status=" + response.StatusCode + ", meta=" + response.MetaJson);
            return;
        }

        AsrResultMeta result = JsonUtility.FromJson<AsrResultMeta>(response.MetaJson);
        Debug.Log("ASR text: " + result.text);
    }
}
```

연결 확인:

```csharp
NfcpFrame ping = await NfcpClient.PingAsync("192.168.4.218", 21861, Nfcp.ServiceAsr);
Debug.Log(ping.MetaJson);
```

## 7. TTS TCP 사용법

접속 정보:

```text
host: 192.168.4.218
port: 26120
service_type: TTS(3)
command: TTS_SYNTHESIZE(3001)
```

요청 meta:

```json
{
  "text": "안녕하세요. 안내를 시작합니다.",
  "audio_format": "wav",
  "language": "ko",
  "speed": 1.0
}
```

성공 응답:

| 위치 | 내용 |
| --- | --- |
| `response.MetaJson` | `duration_ms`, `samplerate`, `channels`, `processor_meta` |
| `response.Data` | WAV bytes |

## 8. Unity C# TTS TCP 예제

```csharp
using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

public class NeuroFlowTtsTcpExample : MonoBehaviour
{
    private const string Host = "192.168.4.218";
    private const int TtsTcpPort = 26120;

    public AudioSource audioSource;

    public async void Speak()
    {
        string wavPath = await SynthesizeToFile("안녕하세요. 안내를 시작합니다.");
        StartCoroutine(PlayWavFile(wavPath));
    }

    public async Task<string> SynthesizeToFile(string text)
    {
        string metaJson =
            "{\"text\":\"" + JsonEscape(text) + "\",\"audio_format\":\"wav\",\"language\":\"ko\",\"speed\":1.0}";

        NfcpFrame response = await NfcpClient.SendAsync(
            Host,
            TtsTcpPort,
            Nfcp.ServiceTts,
            Nfcp.CommandTtsSynthesize,
            metaJson,
            Array.Empty<byte>(),
            timeoutMs: 60000);

        if (response.IsError)
        {
            throw new Exception("TTS error: status=" + response.StatusCode + ", meta=" + response.MetaJson);
        }

        string path = Path.Combine(Application.persistentDataPath, "neuroflow_tts_tcp.wav");
        File.WriteAllBytes(path, response.Data);
        Debug.Log("TTS meta: " + response.MetaJson);
        return path;
    }

    private IEnumerator PlayWavFile(string path)
    {
        using (UnityWebRequest req = UnityWebRequestMultimedia.GetAudioClip("file://" + path, AudioType.WAV))
        {
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError(req.error);
                yield break;
            }

            AudioClip clip = DownloadHandlerAudioClip.GetContent(req);
            audioSource.clip = clip;
            audioSource.Play();
        }
    }

    private static string JsonEscape(string value)
    {
        return value
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"")
            .Replace("\r", "\\r")
            .Replace("\n", "\\n");
    }
}
```
