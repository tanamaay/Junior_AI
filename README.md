# Sentiment Classifier Evaluation Harness

Python evaluation harness for an LLM-based sentiment classifier on customer call transcript snippets. Built for the ALCIAN / Boostt Junior AI Engineer technical assessment.

## What it does

1. Loads a labeled CSV (`id`, `snippet`, `label`)
2. Sends each snippet to an LLM via the [Groq API](https://groq.com/) (OpenAI-compatible)
3. Normalizes the model output to `positive`, `neutral`, or `negative`
4. Reports accuracy, per-class metrics, a confusion matrix, and failure breakdowns
5. Saves predictions, misclassified rows, and a JSON summary under `results/`

## Setup

```bash
cd "junir ai"
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows — use `cp` on macOS/Linux
```

Edit `.env` and set `GROQ_API_KEY` ([Groq console](https://console.groq.com/)). Optional: `GROQ_MODEL` (default `llama-3.3-70b-versatile`), `MAX_WORKERS`, `MAX_EXAMPLES`.

## Dataset

Download the CSV from the assessment email and save it as:

```
data/dataset.csv
```

**Inspect the data first** (recommended before running evaluation):

```bash
python inspect_dataset.py
```

Smoke test with the included sample file:

```bash
python inspect_dataset.py --dataset data/sample_dataset.csv
```

## Run evaluation

Full run (100 examples):

```bash
python run_eval.py
```

Quick smoke test (5 examples, serial):

```bash
python run_eval.py --limit 5 --workers 1
```

Custom paths:

```bash
python run_eval.py --dataset data/dataset.csv --results-dir results --workers 3
```

## Output

Each run creates a timestamped folder, e.g. `results/run_20260528_143022/`:

| File | Contents |
|------|----------|
| `summary.json` | Accuracy, per-class metrics, confusion matrix, failure breakdown |
| `predictions.csv` | Every example with true/predicted labels and raw LLM text |
| `misclassified.csv` | Rows where prediction ≠ ground truth |

## Design choices

### Label normalization

LLMs often return `positive!` or full sentences. `sentiment_eval/normalize.py` strips punctuation, accepts exact matches, then searches for the first `positive` / `neutral` / `negative` token in the response. Unparseable outputs are counted separately and treated as incorrect.

### Serial vs parallel

Default: **3 parallel workers** (`MAX_WORKERS` in `.env`). Tradeoffs:

- **Parallel**: faster for 100 rows; watch API rate limits.
- **Serial** (`--workers 1`): slower but easier to debug and gentler on quotas.

For **10,000 examples**, you would batch requests, add stronger rate-limit handling, cache predictions by snippet hash, and consider async I/O or a job queue — the current structure separates loading, classification, metrics, and I/O so those changes stay localized.

### Failure breakdown

Overall accuracy alone hides class imbalance. The harness reports:

- Per-class precision, recall, F1
- Confusion matrix (true × predicted)
- **Failures by true label** — e.g. “of all truly negative snippets, how many were called neutral?”

### Error handling

- Dataset validation (missing file, bad columns, invalid labels)
- API retries with exponential backoff on rate limits / connection errors
- Per-row error capture so one failure does not stop the run
- UTF-8 with BOM support (`utf-8-sig`) for Excel-exported CSVs

## Tests

```bash
python -m pytest tests/ -q
```

## AI tools (for your Loom video)

If you used Cursor, ChatGPT, or similar while building this, mention briefly **which tools** and **how** (e.g. scaffolding, debugging API retries, README wording). The assessment encourages transparency.

## Project layout

```
run_eval.py              # CLI entry point
inspect_dataset.py       # Dataset preview
sentiment_eval/
  config.py              # Environment / settings
  dataset.py             # CSV loading and validation
  client.py              # Groq client (OpenAI-compatible)
  llm.py                 # Classifier + retries
  normalize.py           # Map free-form output → label
  metrics.py             # Accuracy, confusion matrix, breakdowns
  results_io.py          # Save CSV + JSON
  runner.py              # Orchestration (serial / parallel)
data/
  dataset.csv            # Your 100-row file (not in repo)
tests/
```

## If you had more time

Ideas to mention in the live session: few-shot examples in the prompt, cost/latency logging, comparison across models, human review export for ambiguous cases, or a small CLI to re-score saved raw responses without re-calling the API.
