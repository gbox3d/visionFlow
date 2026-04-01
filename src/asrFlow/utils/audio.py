"""Compatibility re-export — canonical path is common.runtime.audio_codec."""

from common.runtime.audio_codec import decode_audio_bytes, encode_wav_bytes

__all__ = ["decode_audio_bytes", "encode_wav_bytes"]
