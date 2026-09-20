#!/bin/sh
set -eu

: "${MODEL_TYPE:?MODEL_TYPE is required}"
: "${MODEL_REPO:?MODEL_REPO is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${BATCH_SIZE:?BATCH_SIZE is required}"
: "${MAX_PROMPT_CHARACTERS_PER_BATCH:?MAX_PROMPT_CHARACTERS_PER_BATCH is required}"

source_dir=${SOURCE_DIR:-/source}
if [ -n "${DATASET_DIR:-}" ]; then
  runtime_source=/tmp/decision-bench-source
  rm -rf "$runtime_source"
  cp -R "$source_dir" "$runtime_source"
  mkdir -p "$runtime_source/datasets"
  ln -s "$DATASET_DIR" "$runtime_source/datasets/decisionbench-dev"
  source_dir=$runtime_source
fi

uvx --from huggingface-hub==1.32.0 hf download \
  "$MODEL_REPO" \
  --revision "$MODEL_REVISION" \
  --local-dir /tmp/model

set -- uv run "$source_dir/jobs/run_hf_eval.py" \
  --source-dir "$source_dir" \
  --model-dir /tmp/model \
  --model-type "$MODEL_TYPE" \
  --model-repo "$MODEL_REPO" \
  --model-revision "$MODEL_REVISION" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-prompt-characters-per-batch "$MAX_PROMPT_CHARACTERS_PER_BATCH"

if [ -n "${BASE_REVISION:-}" ]; then
  set -- "$@" --base-revision "$BASE_REVISION"
fi
if [ -n "${EXPECTED_WEIGHTS_SHA256:-}" ]; then
  set -- "$@" --expected-weights-sha256 "$EXPECTED_WEIGHTS_SHA256"
fi
if [ -n "${EXPECTED_ROWS:-}" ]; then
  set -- "$@" --expected-rows "$EXPECTED_ROWS"
fi
if [ -n "${EXPECTED_SUCCESSFUL_ROWS:-}" ]; then
  set -- "$@" --expected-successful-rows "$EXPECTED_SUCCESSFUL_ROWS"
fi
if [ -n "${EXPECTED_ERROR_ROWS:-}" ]; then
  set -- "$@" --expected-error-rows "$EXPECTED_ERROR_ROWS"
fi
if [ "${SMOKE:-0}" = "1" ]; then
  set -- "$@" --smoke
fi

exec "$@"
