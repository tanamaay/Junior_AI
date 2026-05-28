#!/usr/bin/env python3
"""Quick preview of the assessment dataset — run this before evaluating."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the sentiment dataset CSV.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset.csv"),
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of rows to print (default: 5)",
    )
    args = parser.parse_args()

    path = args.dataset
    if not path.is_file():
        raise SystemExit(
            f"Dataset not found: {path}\n"
            "Download the CSV from the assessment email and save it as data/dataset.csv"
        )

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    labels = Counter((r.get("label") or "").strip().lower() for r in rows)
    lengths = [len((r.get("snippet") or "")) for r in rows]

    print(f"File: {path}")
    print(f"Rows: {len(rows)}")
    print(f"Label distribution: {dict(labels)}")
    if lengths:
        print(f"Snippet length (chars): min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")

    print(f"\nFirst {args.preview} examples:\n")
    for row in rows[: args.preview]:
        snippet = (row.get("snippet") or "").replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        print(f"  id={row.get('id')}  label={row.get('label')}")
        print(f"    {snippet}\n")


if __name__ == "__main__":
    main()
