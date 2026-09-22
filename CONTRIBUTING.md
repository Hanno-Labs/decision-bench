# Contributing

DecisionBench separates benchmark execution from reviewed results:

- runtime, adapters, metrics, docs, and leaderboard code live in this repository;
- accepted result records live in
  [`Hanno-Labs/decision-bench-results`](https://github.com/Hanno-Labs/decision-bench-results).

Use the documentation guides for [adding a model](docs/contributing/adding_a_model.md),
[adding a task](docs/contributing/adding_a_task.md), [adding a benchmark](docs/contributing/adding_a_benchmark.md),
and [submitting results](docs/contributing/submitting_results.md).

Run formatting, linting, typing, and tests before submitting a change:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```
