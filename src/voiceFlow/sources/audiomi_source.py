from __future__ import annotations

"""
voiceFlow/sources/audiomi_source.py

audioMi TCP 서버에서 PCM16 mono 16kHz 오디오를 수신해
TopicBus("audio/raw")로 publish하는 소스.

프로토콜 (Little Endian):
  PING   Client->Server: <ii  (checkcode, cmd=99)
  ACK    Server->Client: <iiB (checkcode, cmd=99, status=0)
  Audio  Server->Client: <ii  (checkcode, cmd)
                         <i   (data_len)
                         [PCM16 bytes]

  cmd=0x01 loopback, cmd=0x02 mic
"""

import asyncio
import struct
import threading
import time
from typing import Optional

import numpy as np

from visionflow.pipeline.bus import TopicBus
from voiceFlow.pipeline.packet import AudioPacket

REQUEST_PING = 99
CMD_LOOPBACK = 0x01
CMD_MIC = 0x02

_SAMPLERATE = 16000
_BYTES_PER_SAMPLE = 2  # PCM16


class AudioMiSource:
    """
    audioMi TCP 서버 → TopicBus 소스.

    Parameters
    ----------
    bus         : TopicBus
    out_topic   : 오디오 패킷 publish 토픽 (기본 "audio/raw")
    host        : audioMi 서버 주소
    port        : audioMi 서버 포트
    checkcode   : 프로토콜 식별 코드
    cmd_filter  : 수신할 cmd (None=전체, CMD_LOOPBACK=0x01, CMD_MIC=0x02)
    reconnect_s : 재접속 대기 시간(초)
    source_id   : 패킷 식별자
    """

    def __init__(
        self,
        bus: TopicBus,
        out_topic: str = "audio/raw",
        host: str = "127.0.0.1",
        port: int = 26070,
        checkcode: int = 20250918,
        cmd_filter: Optional[int] = None,
        reconnect_s: float = 2.0,
        source_id: str = "audiomi",
    ):
        self.bus = bus
        self.out_topic = out_topic
        self.host = host
        self.port = port
        self.checkcode = checkcode
        self.cmd_filter = cmd_filter
        self.reconnect_s = reconnect_s
        self.source_id = source_id

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._connected = False

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[AudioMiSource] 시작: {self.host}:{self.port}  checkcode={self.checkcode}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=4.0)
            self._thread = None
        self._connected = False
        print("[AudioMiSource] 종료")

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ Internal

    def _run(self) -> None:
        """재접속 루프 – asyncio 이벤트 루프를 스레드 내부에서 실행."""
        while self._running:
            try:
                asyncio.run(self._async_receive())
            except Exception as e:
                print(f"[AudioMiSource] 연결 오류: {e}")
            self._connected = False
            if self._running:
                print(f"[AudioMiSource] {self.reconnect_s}초 후 재접속 시도...")
                time.sleep(self.reconnect_s)

    async def _async_receive(self) -> None:
        print(f"[AudioMiSource] 접속 중: {self.host}:{self.port}")
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            # 1) PING 전송
            writer.write(struct.pack("<ii", self.checkcode, REQUEST_PING))
            await writer.drain()

            # 2) ACK 수신 (9 bytes)
            ack = await reader.readexactly(9)
            recv_code, cmd, status = struct.unpack("<iiB", ack)
            if recv_code != self.checkcode or cmd != REQUEST_PING or status != 0:
                print(f"[AudioMiSource] PING ACK 실패: code={recv_code} cmd={cmd} status={status}")
                return

            self._connected = True
            print("[AudioMiSource] 연결 완료 – 오디오 수신 시작")

            # 3) 오디오 패킷 수신 루프
            while self._running:
                # header 8 bytes: <ii
                header = await reader.readexactly(8)
                h_code, h_cmd = struct.unpack("<ii", header)
                if h_code != self.checkcode:
                    print(f"[AudioMiSource] checkcode 불일치: {h_code}")
                    break

                # size 4 bytes: <i
                (size,) = struct.unpack("<i", await reader.readexactly(4))
                if size <= 0:
                    continue

                raw = await reader.readexactly(size)

                # cmd 필터
                if self.cmd_filter is not None and h_cmd != self.cmd_filter:
                    continue

                # PCM16 → float32 [-1, 1]
                pcm16 = np.frombuffer(raw, dtype=np.int16)
                audio_f32 = pcm16.astype(np.float32) / 32768.0  # (N,)

                self._seq += 1
                pkt = AudioPacket(
                    audio=audio_f32,
                    ts_ms=int(time.time() * 1000),
                    seq=self._seq,
                    source_id=self.source_id,
                    meta={
                        "samplerate": _SAMPLERATE,
                        "channels": 1,
                        "blocksize": len(audio_f32),
                        "audiomi_cmd": h_cmd,
                    },
                )
                self.bus.publish(self.out_topic, pkt)

        finally:
            self._connected = False
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
