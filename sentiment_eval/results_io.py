from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sentiment_eval.metrics import EvalMetrics, PredictionRecord


def create_run_dir(base: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_results(
    run_dir: Path,
    *,
    records: list[PredictionRecord],
    metrics: EvalMetrics,
    config_snapshot: dict,
) -> None:
    predictions_path = run_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "snippet",
                "true_label",
                "predicted_label",
                "raw_response",
                "correct",
                "error",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.id,
                    "snippet": record.snippet,
                    "true_label": record.true_label,
                    "predicted_label": record.predicted_label or "",
                    "raw_response": record.raw_response,
                    "correct": record.correct,
                    "error": record.error or "",
                }
            )

    misclassified_path = run_dir / "misclassified.csv"
    with misclassified_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "true_label",
                "predicted_label",
                "raw_response",
                "snippet",
            ],
        )
        writer.writeheader()
        for record in metrics.misclassified:
            writer.writerow(
                {
                    "id": record.id,
                    "true_label": record.true_label,
                    "predicted_label": record.predicted_label or "unparseable",
                    "raw_response": record.raw_response,
                    "snippet": record.snippet,
                }
            )

    summary = {
        "config": config_snapshot,
        "metrics": {
            "total": metrics.total,
            "correct": metrics.correct,
            "accuracy": metrics.accuracy,
            "unparseable_responses": metrics.unparseable,
            "per_class": metrics.per_class,
            "confusion_matrix": metrics.confusion,
            "failures_by_true_label": metrics.failures_by_true_label,
        },
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
