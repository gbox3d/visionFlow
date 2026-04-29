# Unity TTS REST 연동 가이드

> 대상
> `nf-tts-rest-server`를 Unity/C#에서 HTTP로 호출하는 개발자용 문서다.

## 목차

1. [서비스 주소](#1-서비스-주소)
2. [REST API](#2-rest-api)
3. [curl 확인](#3-curl-확인)
4. [Unity C# TTS REST 예제](#4-unity-c-tts-rest-예제)
5. [Unity C# 상태 확인 예제](#5-unity-c-상태-확인-예제)

## 1. 서비스 주소

```text
Base URL: http://192.168.4.218:26121
```

## 2. REST API

| Method | Path | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 서버 상태 확인 |
| `GET` | `/describe` | 엔진, 모델, 기본값 확인 |
| `POST` | `/tts` | JSON text 요청 -> `audio/wav` 응답 |

`POST /tts` 요청 body:

```json
{
  "text": "안녕하세요. 안내를 시작합니다.",
  "audio_format": "wav",
  "language": "ko",
  "speaker_id": null,
  "speed": 1.0
}
```

성공 응답:

| 항목 | 값 |
| --- | --- |
| HTTP status | `200 OK` |
| Content-Type | `audio/wav` |
| Body | WAV bytes |
| Header | `X-NeuroFlow-TTS-Samplerate`, `X-NeuroFlow-TTS-Duration-Ms`, `X-NeuroFlow-TTS-Inference-Ms`, `X-NeuroFlow-TTS-RTF` |

## 3. curl 확인

```bash
curl -fsS http://192.168.4.218:26121/health

curl -sS -D /tmp/tts_headers.txt \
  -o /tmp/neuroflow_tts.wav \
  -X POST http://192.168.4.218:26121/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"안녕하세요. 안내를 시작합니다.","audio_format":"wav","language":"ko","speed":1.0}'
```

## 4. Unity C# TTS REST 예제

```csharp
using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class NeuroFlowTtsRestExample : MonoBehaviour
{
    private const string TtsRestUrl = "http://192.168.4.218:26121/tts";

    public AudioSource audioSource;

    public void Speak()
    {
        StartCoroutine(SynthesizeAndPlay("안녕하세요. 안내를 시작합니다."));
    }

    private IEnumerator SynthesizeAndPlay(string text)
    {
        string json =
            "{\"text\":\"" + JsonEscape(text) + "\",\"audio_format\":\"wav\",\"language\":\"ko\",\"speed\":1.0}";
        byte[] body = Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest req = new UnityWebRequest(TtsRestUrl, "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(body);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("TTS REST error: " + req.responseCode + " " + req.error + " " + req.downloadHandler.text);
                yield break;
            }

            byte[] wavBytes = req.downloadHandler.data;
            string wavPath = Path.Combine(Application.persistentDataPath, "neuroflow_tts_rest.wav");
            File.WriteAllBytes(wavPath, wavBytes);

            Debug.Log("TTS duration ms: " + req.GetResponseHeader("X-NeuroFlow-TTS-Duration-Ms"));
            Debug.Log("TTS inference ms: " + req.GetResponseHeader("X-NeuroFlow-TTS-Inference-Ms"));

            yield return PlayWavFile(wavPath);
        }
    }

    private IEnumerator PlayWavFile(string path)
    {
        using (UnityWebRequest audioReq =
            UnityWebRequestMultimedia.GetAudioClip("file://" + path, AudioType.WAV))
        {
            yield return audioReq.SendWebRequest();

            if (audioReq.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError(audioReq.error);
                yield break;
            }

            AudioClip clip = DownloadHandlerAudioClip.GetContent(audioReq);
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

## 5. Unity C# 상태 확인 예제

```csharp
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class NeuroFlowTtsHealthExample : MonoBehaviour
{
    private IEnumerator Start()
    {
        using (UnityWebRequest req = UnityWebRequest.Get("http://192.168.4.218:26121/health"))
        {
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                Debug.Log(req.downloadHandler.text);
            }
            else
            {
                Debug.LogError(req.error);
            }
        }
    }
}
```
