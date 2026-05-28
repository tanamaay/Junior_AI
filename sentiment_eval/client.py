from __future__ import annotations

import os

from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def create_llm_client() -> OpenAI:
    """OpenAI-compatible client configured for Groq."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
