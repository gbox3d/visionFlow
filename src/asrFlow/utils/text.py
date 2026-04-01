from __future__ import annotations

from typing import Protocol


class FontLike(Protocol):
    def size(self, text: str) -> tuple[int, int]:
        ...


def wrap_text(text: str, font: FontLike, max_width: int, max_lines: int) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return [""]
    if max_lines <= 0:
        return []

    words = raw.split()
    lines: list[str] = []
    current = ""

    for word in words:
        trial = word if not current else f"{current} {word}"
        if font.size(trial)[0] <= max_width:
            current = trial
            continue

        if current:
            lines.append(current)
            if len(lines) >= max_lines:
                break
            current = word
        else:
            lines.append(word)
            if len(lines) >= max_lines:
                break
            current = ""

    if len(lines) < max_lines and current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    if len(lines) == max_lines and words:
        last_line = lines[-1]
        while font.size(last_line + " ...")[0] > max_width and len(last_line) > 1:
            last_line = last_line[:-1]
        lines[-1] = last_line + " ..."

    return lines
