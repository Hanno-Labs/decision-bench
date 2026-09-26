#!/usr/bin/env bash
# DecisionBench evaluation of mohit67890/imajev-4b through its own Jev-compatible SystemOne server.
# Mirrors jobs/run_xor_eval.sh: pinned weights, checksummed files, one local server, run_system_one_http_eval.py.
set -euo pipefail

SOURCE_DIR=""
OUTPUT_DIR=""
DURABLE_URI=""
MODEL_DIR=""
TASK_SPEC=""
SMOKE=0
CONCURRENCY=8
EXPECTED_ROWS=""
EXPECTED_SUCCESSFUL_ROWS=""
EXPECTED_ERROR_ROWS=""
MODEL_REPO="mohit67890/imajev-4b"
MODEL_REVISION="c9e5f132465da85d31735ec502d5557982671a7d"
SERVING_BUNDLE_SHA256="88c2c44361e0c469352495abcfee789ff73a4deae0811168d9402cfc2b6e749c"   # adapter_model.safetensors
SERVER_REPO="https://github.com/mohit67890/imajev"
SERVER_COMMIT="a0134749e0900189c129cd6bb5000969f3b64bb5"
INFERENCE_IMAGE="github.com/mohit67890/imajev@${SERVER_COMMIT} scripts/playground/server.py --backend torch --rotations 4 --calibration calibration-rot4.json --max-input-tokens 65536"
PORT=30002

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

for value in SOURCE_DIR OUTPUT_DIR MODEL_DIR; do
  [[ -n "${!value}" ]] || { echo "missing --${value,,}" >&2; exit 2; }
done
TASK_SPEC="${TASK_SPEC:-$SOURCE_DIR/task_specs/decisionbench-dev.toml}"
SERVING_DIR="$(dirname "$MODEL_DIR")/imajev-serving"
mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"

if [[ -n "$DURABLE_URI" ]]; then hf sync "$DURABLE_URI" "$OUTPUT_DIR" || true; fi

# Pinned adapter: every file listed in the repository's SHA256SUMS must match, and the LoRA weights must match the pin above.
hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
(
  cd "$MODEL_DIR"
  sha256sum -c SHA256SUMS
  printf '%s  %s\n' "$SERVING_BUNDLE_SHA256" adapter_model.safetensors | sha256sum -c -
)

# Pinned server: the model's own repository at one commit. The base weights are pinned by artifacts/model-qwen4b.json.
if [[ ! -d "$SERVING_DIR/.git" ]]; then git clone --quiet "$SERVER_REPO" "$SERVING_DIR"; fi
git -C "$SERVING_DIR" checkout --quiet "$SERVER_COMMIT"
python3 -m venv "$SERVING_DIR/.venv"
"$SERVING_DIR/.venv/bin/pip" install --quiet --upgrade pip
# flash-linear-attention and tilelang give the DeltaNet layers their CUDA kernels; without them the server falls back to a slow path.
"$SERVING_DIR/.venv/bin/pip" install --quiet -e "$SERVING_DIR[serve,torch]" flash-linear-attention tilelang
"$SERVING_DIR/.venv/bin/python" "$SERVING_DIR/scripts/download_model.py" --bundle "$SERVING_DIR/artifacts/model-qwen4b.json"

SERVER_PID=""
SYNC_PID=""
cleanup() {
  local status=$?
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null || true; wait "$SYNC_PID" 2>/dev/null || true; fi
  if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
  exit "$status"
}
trap cleanup EXIT TERM INT
if [[ -n "$DURABLE_URI" ]]; then
  ( while true; do sleep 300; hf sync "$OUTPUT_DIR" "$DURABLE_URI"; done ) &
  SYNC_PID=$!
fi

(
  cd "$SERVING_DIR"
  PYTHONPATH=src:scripts .venv/bin/python scripts/playground/server.py \
    --backend torch \
    --model-bundle artifacts/model-qwen4b.json \
    --adapter "$MODEL_DIR" \
    --rotations 4 \
    --calibration "$MODEL_DIR/calibration-rot4.json" \
    --max-input-tokens 65536 \
    --model-name imajev-4b \
    --host 127.0.0.1 \
    --port "$PORT"
) &
SERVER_PID=$!
for _ in $(seq 1 360); do
  if curl --fail --silent "http://127.0.0.1:$PORT/v1/models" | grep -q '"loaded": *true'; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then wait "$SERVER_PID"; fi
  sleep 5
done
curl --fail --silent "http://127.0.0.1:$PORT/v1/models" | grep -q '"loaded": *true'

evaluation=(
  uv run "$SOURCE_DIR/jobs/run_system_one_http_eval.py"
  --source-dir "$SOURCE_DIR"
  --task-spec "$TASK_SPEC"
  --output-dir "$OUTPUT_DIR"
  --base-url "http://127.0.0.1:$PORT"
  --model imajev-4b
  --model-repo "$MODEL_REPO"
  --model-revision "$MODEL_REVISION"
  --serving-bundle-sha256 "$SERVING_BUNDLE_SHA256"
  --inference-image "$INFERENCE_IMAGE"
  --concurrency "$CONCURRENCY"
  --max-candidates 255
  --adapter-name imajev-serving-systemone-v1
  --probability-source trained_readout_4_rotation_mean_temperature_calibrated_v1
)
if [[ -n "$EXPECTED_ROWS" ]]; then evaluation+=(--expected-rows "$EXPECTED_ROWS"); fi
if [[ -n "$EXPECTED_SUCCESSFUL_ROWS" ]]; then evaluation+=(--expected-successful-rows "$EXPECTED_SUCCESSFUL_ROWS"); fi
if [[ -n "$EXPECTED_ERROR_ROWS" ]]; then evaluation+=(--expected-error-rows "$EXPECTED_ERROR_ROWS"); fi
if [[ "$SMOKE" == 1 ]]; then evaluation+=(--smoke); fi
"${evaluation[@]}"
if [[ -n "$DURABLE_URI" ]]; then hf sync "$OUTPUT_DIR" "$DURABLE_URI"; fi
