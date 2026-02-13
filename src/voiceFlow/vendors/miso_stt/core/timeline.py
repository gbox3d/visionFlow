from __future__ import annotations

from typing import Iterable

from voiceFlow.vendors.miso_stt.core.types import Segment


def split_segments_by_words(segments: Iterable[Segment]) -> list[Segment]:
    """Split segment timeline into word-level timeline by uniform duration."""
    word_segments: list[Segment] = []
    for seg in segments:
        words = [w for w in seg.text.split() if w]
        if not words:
            continue
        duration = max(seg.end - seg.start, 0.0)
        step = duration / len(words) if words else 0.0
        for i, word in enumerate(words):
            w_start = seg.start + (i * step)
            w_end = seg.start + ((i + 1) * step) if i < len(words) - 1 else seg.end
            word_segments.append(Segment(round(w_start, 2), round(w_end, 2), word))
    return word_segments

