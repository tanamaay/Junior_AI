from __future__ import annotations

import re

from sentiment_eval.config import VALID_LABELS

# Match label words even inside longer replies ("The sentiment is positive.")
_LABEL_PATTERN = re.compile(
    r"\b(positive|neutral|negative)\b",
    re.IGNORECASE,
)


def normalize_label(raw: str | None) -> str | None:
    """Map free-form LLM output to positive | neutral | negative, or None."""
    if raw is None:
        return None

    text = raw.strip().lower()
    if not text:
        return None

    # Exact match after stripping punctuation
    cleaned = re.sub(r"[^\w]", "", text)
    if cleaned in VALID_LABELS:
        return cleaned

    # First valid label mentioned in the string
    match = _LABEL_PATTERN.search(text)
    if match:
        return match.group(1).lower()

    return None
