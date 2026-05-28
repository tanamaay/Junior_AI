from __future__ import annotations

import time
from typing import Callable

from google import genai
from google.genai import types

from sentiment_eval.normalize import normalize_label

SYSTEM_PROMPT = """You classify customer call transcript snippets by sentiment.

Return exactly one word: positive, neutral, or negative.
Do not add punctuation, explanation, or any other text."""

USER_PROMPT_TEMPLATE = """Classify the sentiment of this call snippet:

\"\"\"{snippet}\"\"\"
"""


def _response_text(response) -> str:
    """Extract model text; gemini-2.5 may use tokens on reasoning before output."""
    text = response.text
    if text:
        return text.strip()
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return ""
    return "".join(
        part.text for part in parts if getattr(part, "text", None)
    ).strip()


def build_classify_fn(
    *,
    client: genai.Client,
    model: str,
    max_retries: int,
) -> Callable[[str], tuple[str | None, str]]:
    """Return a function that classifies a snippet -> (normalized_label, raw_response)."""

    def classify(snippet: str) -> tuple[str | None, str]:
        last_error: Exception | None = None
        prompt = USER_PROMPT_TEMPLATE.format(snippet=snippet)

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0,
                        max_output_tokens=64,
                    ),
                )
                raw = _response_text(response)
                return normalize_label(raw), raw
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= max_retries:
                    break
                time.sleep(2**attempt)

        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")

    return classify
