#!/bin/sh
set -eu

cp -a /source /tmp/decision-bench
cd /tmp/decision-bench
uv sync --group dev
uv pip install torch==2.8.0 peft==0.21.0 transformers==5.17.0
uv run ruff check src tests jobs/run_hf_eval.py
uv run mypy --ignore-missing-imports --follow-imports=skip \
  src/decision_bench/models/public_hf.py \
  src/decision_bench/models/mojev_hf.py \
  src/decision_bench/evaluate.py \
  jobs/run_hf_eval.py
uv run pytest tests/test_hf_model.py tests/test_intern_decision_model.py tests/test_evaluate.py
