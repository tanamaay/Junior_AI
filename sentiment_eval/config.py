from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

VALID_LABELS = frozenset({"positive", "neutral", "negative"})


@dataclass(frozen=True)
class EvalConfig:
    dataset_path: Path
    results_dir: Path
    model: str
    max_workers: int
    max_examples: int | None
    api_max_retries: int

    @classmethod
    def from_env(
        cls,
        dataset_path: Path | None = None,
        results_dir: Path | None = None,
    ) -> EvalConfig:
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your Gemini API key."
            )

        max_examples_raw = os.getenv("MAX_EXAMPLES", "").strip()
        max_examples: int | None = None
        if max_examples_raw:
            max_examples = int(max_examples_raw)
            if max_examples <= 0:
                max_examples = None

        return cls(
            dataset_path=dataset_path or Path("data/dataset.csv"),
            results_dir=results_dir or Path("results"),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            max_workers=max(1, int(os.getenv("MAX_WORKERS", "3"))),
            max_examples=max_examples,
            api_max_retries=max(1, int(os.getenv("API_MAX_RETRIES", "3"))),
        )
