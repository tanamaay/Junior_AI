from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from sentiment_eval.config import VALID_LABELS


@dataclass
class PredictionRecord:
    id: str
    snippet: str
    true_label: str
    predicted_label: str | None
    raw_response: str
    correct: bool
    error: str | None = None


@dataclass
class EvalMetrics:
    total: int
    correct: int
    accuracy: float
    unparseable: int
    per_class: dict[str, dict[str, float | int]]
    confusion: dict[str, dict[str, int]]
    misclassified: list[PredictionRecord] = field(default_factory=list)
    failures_by_true_label: dict[str, dict[str, int]] = field(default_factory=dict)


def compute_metrics(records: list[PredictionRecord]) -> EvalMetrics:
    total = len(records)
    correct = sum(1 for r in records if r.correct)
    unparseable = sum(1 for r in records if r.predicted_label is None)

    confusion: dict[str, dict[str, int]] = {
        true_l: {pred_l: 0 for pred_l in VALID_LABELS} for true_l in VALID_LABELS
    }
    failures_by_true: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in records:
        if record.predicted_label in VALID_LABELS:
            confusion[record.true_label][record.predicted_label] += 1
        if not record.correct:
            pred_key = record.predicted_label or "unparseable"
            failures_by_true[record.true_label][pred_key] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in sorted(VALID_LABELS):
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in VALID_LABELS if other != label)
        fn = sum(confusion[label][other] for other in VALID_LABELS if other != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    misclassified = [r for r in records if not r.correct]

    return EvalMetrics(
        total=total,
        correct=correct,
        accuracy=round(correct / total, 4) if total else 0.0,
        unparseable=unparseable,
        per_class=per_class,
        confusion={
            true_l: dict(confusion[true_l]) for true_l in sorted(VALID_LABELS)
        },
        misclassified=misclassified,
        failures_by_true_label={
            k: dict(v) for k, v in sorted(failures_by_true.items())
        },
    )
