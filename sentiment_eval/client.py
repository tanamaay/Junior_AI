from __future__ import annotations

import os

from google import genai


def create_llm_client() -> genai.Client:
    """Google Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your Gemini API key."
        )
    return genai.Client(api_key=api_key)
