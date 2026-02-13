from __future__ import annotations

from collections import Counter


def is_repetition_hallucination(text: str, repetition_threshold: float = 0.4) -> bool:
    """Detect repetitive hallucination-like text patterns."""
    if len(text) < 3:
        return False

    compact = text.replace(" ", "")
    if not compact:
        return False

    char_counts = Counter(compact)
    if char_counts:
        ratio = char_counts.most_common(1)[0][1] / len(compact)
        if ratio >= repetition_threshold:
            return True

    if len(compact) >= 6:
        trigrams = [compact[i : i + 3] for i in range(len(compact) - 2)]
        trigram_counts = Counter(trigrams)
        if trigram_counts and trigram_counts.most_common(1)[0][1] >= len(trigrams) * 0.5:
            return True

    words = [w for w in text.split() if w]
    if len(words) >= 8:
        word_counts = Counter(words)
        if word_counts and (word_counts.most_common(1)[0][1] / len(words)) >= 0.35:
            return True

    return False

