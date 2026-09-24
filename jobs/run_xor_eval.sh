#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=""
OUTPUT_DIR=""
DURABLE_URI=""
MODEL_DIR=""
TASK_SPEC=""
SMOKE=0
CONCURRENCY=32
EXPECTED_ROWS=""
EXPECTED_SUCCESSFUL_ROWS=""
EXPECTED_ERROR_ROWS=""
MODEL_REPO="juspay/xor"
MODEL_REVISION="679decd4c669e5c37f4ac29dbd9957997424c876"
SERVING_BUNDLE_SHA256="0a63473caaa3c6bfc8bc15fbab62f0a9a84c7ebf4ab6e06d0699891b7be6159b"
INFERENCE_IMAGE="lmsysorg/sglang@sha256:6bcaa47db52f78ce0d67863b8b2431221b79bc23204a80cad757fa819d00e921"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --durable-uri) DURABLE_URI="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --task-spec) TASK_SPEC="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --expected-rows) EXPECTED_ROWS="$2"; shift 2 ;;
    --expected-successful-rows) EXPECTED_SUCCESSFUL_ROWS="$2"; shift 2 ;;
    --expected-error-rows) EXPECTED_ERROR_ROWS="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in SOURCE_DIR OUTPUT_DIR DURABLE_URI MODEL_DIR; do
  [[ -n "${!value}" ]] || { echo "missing --${value,,}" >&2; exit 2; }
done
TASK_SPEC="${TASK_SPEC:-$SOURCE_DIR/task_specs/decisionbench-dev.toml}"
SERVING_DIR="$(dirname "$MODEL_DIR")/xor-serving"
mkdir -p "$MODEL_DIR" "$OUTPUT_DIR" "$SERVING_DIR"

hf sync "$DURABLE_URI" "$OUTPUT_DIR" || true
hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
(
  cd "$MODEL_DIR"
  sha256sum -c checksums.sha256
  printf '%s  %s\n' "$SERVING_BUNDLE_SHA256" serving/xor-serving.tar.gz | sha256sum -c -
)
tar -xzf "$MODEL_DIR/serving/xor-serving.tar.gz" -C "$SERVING_DIR" --strip-components=1

SGLANG_PID=""
WRAPPER_PID=""
SYNC_PID=""
cleanup() {
  local status=$?
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null || true; wait "$SYNC_PID" 2>/dev/null || true; fi
  if [[ -n "$WRAPPER_PID" ]]; then kill "$WRAPPER_PID" 2>/dev/null || true; wait "$WRAPPER_PID" 2>/dev/null || true; fi
  if [[ -n "$SGLANG_PID" ]]; then kill "$SGLANG_PID" 2>/dev/null || true; wait "$SGLANG_PID" 2>/dev/null || true; fi
  exit "$status"
}
trap cleanup EXIT TERM INT

(
  while true; do
    sleep 300
    hf sync "$OUTPUT_DIR" "$DURABLE_URI"
  done
) &
SYNC_PID=$!

python3 -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --trust-remote-code \
  --tp-size 2 \
  --port 30000 \
  --host 127.0.0.1 \
  --max-prefill-tokens 250000 \
  --mem-fraction-static 0.85 &
SGLANG_PID=$!

for _ in $(seq 1 360); do
  if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then break; fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then wait "$SGLANG_PID"; fi
  sleep 5
done
curl --fail --silent http://127.0.0.1:30000/health >/dev/null

OPENJEV_SGLANG_URL=http://127.0.0.1:30000 \
OPENJEV_MODEL_ID=xor \
OPENJEV_BACKEND=Qwen/Qwen3.6-35B-A3B \
OPENJEV_CACHE=0 \
OPENJEV_IMAGES=1 \
python3 "$SERVING_DIR/server.py" 30002 &
WRAPPER_PID=$!
for _ in $(seq 1 60); do
  if curl --fail --silent http://127.0.0.1:30002/health >/dev/null; then break; fi
  if ! kill -0 "$WRAPPER_PID" 2>/dev/null; then wait "$WRAPPER_PID"; fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:30002/health >/dev/null
python3 "$SERVING_DIR/smoke_test.py" http://127.0.0.1:30002

evaluation=(
  uv run "$SOURCE_DIR/jobs/run_system_one_http_eval.py"
  --source-dir "$SOURCE_DIR"
  --task-spec "$TASK_SPEC"
  --output-dir "$OUTPUT_DIR"
  --model-repo "$MODEL_REPO"
  --model-revision "$MODEL_REVISION"
  --serving-bundle-sha256 "$SERVING_BUNDLE_SHA256"
  --inference-image "$INFERENCE_IMAGE"
  --concurrency "$CONCURRENCY"
)
if [[ "$SMOKE" == 1 ]]; then evaluation+=(--smoke); fi
if [[ -n "$EXPECTED_ROWS" ]]; then evaluation+=(--expected-rows "$EXPECTED_ROWS"); fi
if [[ -n "$EXPECTED_SUCCESSFUL_ROWS" ]]; then
  evaluation+=(--expected-successful-rows "$EXPECTED_SUCCESSFUL_ROWS")
fi
if [[ -n "$EXPECTED_ERROR_ROWS" ]]; then
  evaluation+=(--expected-error-rows "$EXPECTED_ERROR_ROWS")
fi
"${evaluation[@]}"
hf sync "$OUTPUT_DIR" "$DURABLE_URI"
