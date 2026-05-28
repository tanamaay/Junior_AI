from __future__ import annotations

import time
from typing import Callable

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from sentiment_eval.normalize import normalize_label

SYSTEM_PROMPT = """You classify customer call transcript snippets by sentiment.

Return exactly one word: positive, neutral, or negative.
Do not add punctuation, explanation, or any other text."""

USER_PROMPT_TEMPLATE = """Classify the sentiment of this call snippet:

\"\"\"{snippet}\"\"\"
"""


def build_classify_fn(
    *,
    client: OpenAI,
    model: str,
    max_retries: int,
) -> Callable[[str], tuple[str | None, str]]:
    """Return a function that classifies a snippet -> (normalized_label, raw_response)."""

    def classify(snippet: str) -> tuple[str | None, str]:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=16,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": USER_PROMPT_TEMPLATE.format(snippet=snippet),
                        },
                    ],
                )
                raw = (response.choices[0].message.content or "").strip()
                return normalize_label(raw), raw
            except (RateLimitError, APIConnectionError, APIStatusError) as exc:
                last_error = exc
                if attempt + 1 >= max_retries:
                    break
                # Exponential backoff: 1s, 2s, 4s ...
                time.sleep(2**attempt)
            except Exception as exc:
                last_error = exc
                break

        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")

    return classify
