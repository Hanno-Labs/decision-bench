#!/bin/sh
set -eu

: "${MODEL_TYPE:?MODEL_TYPE is required}"
: "${MODEL_REPO:?MODEL_REPO is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${BATCH_SIZE:?BATCH_SIZE is required}"
: "${MAX_PROMPT_CHARACTERS_PER_BATCH:?MAX_PROMPT_CHARACTERS_PER_BATCH is required}"

source_dir=${SOURCE_DIR:-/source}
checkpoint_dir=${CHECKPOINT_DIR:-"$OUTPUT_DIR/checkpoints"}
mkdir -p "$checkpoint_dir"
sync_pid=

sync_results() {
  if [ -n "${RESULTS_URI:-}" ] && [ -d "$OUTPUT_DIR" ]; then
    hf sync "$OUTPUT_DIR" "$RESULTS_URI"
  fi
}

cleanup() {
  run_status=$?
  trap - EXIT INT TERM
  set +e
  if [ -n "$sync_pid" ]; then
    kill "$sync_pid" 2>/dev/null
    wait "$sync_pid" 2>/dev/null
  fi
  sync_results
  sync_status=$?
  if [ "$run_status" -eq 0 ] && [ "$sync_status" -ne 0 ]; then
    run_status=$sync_status
  fi
  exit "$run_status"
}

if [ -n "${RESULTS_URI:-}" ]; then
  mkdir -p "$OUTPUT_DIR"
  hf sync "$RESULTS_URI" "$OUTPUT_DIR" || true
  if [ -f "$checkpoint_dir/raw.jsonl" ]; then
    cp "$checkpoint_dir/raw.jsonl" "$OUTPUT_DIR/raw.jsonl"
  fi
  (
    while sleep "${RESULTS_SYNC_INTERVAL_SECONDS:-120}"; do
      if ! sync_results; then
        printf '%s\n' "DecisionBench checkpoint sync failed; it will retry." >&2
      fi
    done
  ) &
  sync_pid=$!
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
fi

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

if [ "$MODEL_TYPE" = "tev1" ]; then
  set -- uv run --with torchvision==0.23.0 "$source_dir/jobs/run_hf_eval.py"
elif [ "$MODEL_TYPE" = "gliner25" ]; then
  set -- uv run --with gliner2==2.0.0 "$source_dir/jobs/run_hf_eval.py"
elif [ "$MODEL_TYPE" = "mojev" ]; then
  set -- uv run --with 'mojev[transformers] @ git+https://github.com/MoLeMo-Lab/mojev.git@a74d58cd19ec573e83e8e27f9fecd837b8d830fb' "$source_dir/jobs/run_hf_eval.py"
else
  set -- uv run "$source_dir/jobs/run_hf_eval.py"
fi
set -- "$@" \
  --source-dir "$source_dir" \
  --model-dir /tmp/model \
  --model-type "$MODEL_TYPE" \
  --model-repo "$MODEL_REPO" \
  --model-revision "$MODEL_REVISION" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-dir "$checkpoint_dir" \
  --checkpoint-interval-seconds "${CHECKPOINT_INTERVAL_SECONDS:-120}" \
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

"$@"
