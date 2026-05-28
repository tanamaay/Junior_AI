from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sentiment_eval.config import VALID_LABELS


@dataclass(frozen=True)
class Example:
    id: str
    snippet: str
    label: str


REQUIRED_COLUMNS = ("id", "snippet", "label")


def load_dataset(path: Path, max_examples: int | None = None) -> list[Example]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download the CSV from the assessment email and place it at data/dataset.csv"
        )

    examples: list[Example] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} appears to be empty or not a valid CSV.")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{path} is missing required columns {missing}. "
                f"Expected: {', '.join(REQUIRED_COLUMNS)}"
            )

        for row_num, row in enumerate(reader, start=2):
            example_id = (row.get("id") or "").strip()
            snippet = (row.get("snippet") or "").strip()
            label = (row.get("label") or "").strip().lower()

            if not example_id or not snippet:
                raise ValueError(f"{path} row {row_num}: id and snippet must be non-empty.")
            if label not in VALID_LABELS:
                raise ValueError(
                    f"{path} row {row_num}: invalid label '{label}'. "
                    f"Expected one of: {', '.join(sorted(VALID_LABELS))}"
                )

            examples.append(Example(id=example_id, snippet=snippet, label=label))

    if not examples:
        raise ValueError(f"{path} contains no data rows.")

    if max_examples is not None:
        examples = examples[:max_examples]

    return examples
