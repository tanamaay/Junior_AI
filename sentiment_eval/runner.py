from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from tqdm import tqdm

from sentiment_eval.client import create_llm_client
from sentiment_eval.config import EvalConfig
from sentiment_eval.dataset import Example
from sentiment_eval.llm import build_classify_fn
from sentiment_eval.metrics import EvalMetrics, PredictionRecord, compute_metrics
from sentiment_eval.results_io import create_run_dir, save_results


def _evaluate_one(
    example: Example,
    classify: Callable[[str], tuple[str | None, str]],
) -> PredictionRecord:
    try:
        predicted, raw = classify(example.snippet)
        correct = predicted == example.label
        return PredictionRecord(
            id=example.id,
            snippet=example.snippet,
            true_label=example.label,
            predicted_label=predicted,
            raw_response=raw,
            correct=correct,
        )
    except Exception as exc:
        return PredictionRecord(
            id=example.id,
            snippet=example.snippet,
            true_label=example.label,
            predicted_label=None,
            raw_response="",
            correct=False,
            error=str(exc),
        )


def run_evaluation(
    examples: list[Example],
    config: EvalConfig,
) -> tuple[list[PredictionRecord], EvalMetrics, Path]:
    client = create_llm_client()
    classify = build_classify_fn(
        client=client,
        model=config.model,
        max_retries=config.api_max_retries,
    )

    records: list[PredictionRecord] = []
    workers = min(config.max_workers, len(examples))

    if workers <= 1:
        iterator = tqdm(examples, desc="Classifying", unit="ex")
        for example in iterator:
            records.append(_evaluate_one(example, classify))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_evaluate_one, ex, classify): ex for ex in examples
            }
            with tqdm(total=len(examples), desc="Classifying", unit="ex") as bar:
                for future in as_completed(futures):
                    records.append(future.result())
                    bar.update(1)

    # Preserve dataset order for easier diffing
    order = {ex.id: idx for idx, ex in enumerate(examples)}
    records.sort(key=lambda r: order.get(r.id, 0))

    metrics = compute_metrics(records)
    run_dir = create_run_dir(config.results_dir)
    save_results(
        run_dir,
        records=records,
        metrics=metrics,
        config_snapshot={
            "provider": "groq",
            "model": config.model,
            "max_workers": config.max_workers,
            "dataset": str(config.dataset_path),
            "num_examples": len(examples),
        },
    )
    return records, metrics, run_dir


def print_report(metrics: EvalMetrics, run_dir: Path) -> None:
    print("\n=== Evaluation summary ===")
    print(f"Accuracy: {metrics.accuracy:.1%} ({metrics.correct}/{metrics.total})")
    print(f"Unparseable LLM responses: {metrics.unparseable}")

    print("\nPer-class metrics:")
    for label, stats in metrics.per_class.items():
        print(
            f"  {label:8s}  support={stats['support']:3d}  "
            f"precision={stats['precision']:.2f}  "
            f"recall={stats['recall']:.2f}  f1={stats['f1']:.2f}"
        )

    print("\nConfusion matrix (rows=true, cols=predicted):")
    labels = sorted(metrics.confusion)
    header = "          " + "".join(f"{c:>10s}" for c in labels)
    print(header)
    for true_label in labels:
        row = metrics.confusion[true_label]
        counts = "".join(f"{row[p]:>10d}" for p in labels)
        print(f"  {true_label:8s}{counts}")

    print("\nFailures by true label (what the model predicted instead):")
    for true_label, failures in metrics.failures_by_true_label.items():
        parts = ", ".join(f"{pred}={count}" for pred, count in sorted(failures.items()))
        print(f"  {true_label}: {parts}")

    print(f"\nResults saved to: {run_dir.resolve()}")
