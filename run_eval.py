#!/usr/bin/env python3
"""Evaluate an LLM sentiment classifier against a labeled CSV dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentiment_eval.config import EvalConfig
from sentiment_eval.dataset import load_dataset
from sentiment_eval.runner import print_report, run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM sentiment classification evaluation on a labeled dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to CSV (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for run outputs (default: results/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel API workers (default: MAX_WORKERS from .env, or 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N examples (smoke test)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = EvalConfig.from_env(
            dataset_path=args.dataset,
            results_dir=args.results_dir,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    max_examples = args.limit if args.limit is not None else config.max_examples
    workers = args.workers if args.workers is not None else config.max_workers
    config = EvalConfig(
        dataset_path=config.dataset_path,
        results_dir=config.results_dir,
        model=config.model,
        max_workers=workers,
        max_examples=max_examples,
        api_max_retries=config.api_max_retries,
        request_interval_sec=config.request_interval_sec,
    )

    try:
        examples = load_dataset(config.dataset_path, max_examples=config.max_examples)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(examples)} examples from {config.dataset_path}")
    print(f"Model: {config.model}  |  Workers: {config.max_workers}")

    _, metrics, run_dir = run_evaluation(examples, config)
    print_report(metrics, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
