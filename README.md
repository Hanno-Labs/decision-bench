# DecisionBench

DecisionBench is an open benchmark for models that turn documents and structured
state into decisions. It evaluates three standard machine-learning primitives:

- **Binary classification** — decide between two exhaustive outcomes.
- **Candidate selection** — select one member of a runtime-defined candidate set.
- **Ordinal scoring** — place an example on an ordered, discrete scale.

The benchmark keeps code and data separate, following the useful part of MTEB's
architecture. This repository contains task metadata, loaders, model adapters,
evaluation, and result schemas. It does not commit benchmark rows.

## Benchmark data

The canonical frozen dataset is the private `Hanno-Labs/decision-bench`
Hugging Face dataset at the immutable revision recorded in the task spec. It is
one unified benchmark payload; the original, expanded, and reasoning sources
are represented by row metadata rather than separate benchmark targets.

```toml
[dataset]
backend = "huggingface"
split = "eval"
hf_repo = "Hanno-Labs/decision-bench"
hf_revision = "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443"
```

Authenticate with Hugging Face before loading the private dataset. The loader
also supports local payloads for development, but the checked-in canonical task
spec always resolves the pinned hosted revision.

## Generic chat-model baseline

The first adapter targets OpenRouter's `openai/gpt-5.6-luna`. It sends the state,
instruction, and ordered candidates to an ordinary chat model and requires a
strict JSON-schema response containing one probability per candidate. The same
contract supports all three primitives; the evaluator derives the argmax choice
and, for ordinal tasks, the probability-weighted score.

The prompt, schema, provider route, model ID, reasoning setting, raw response,
token usage, and latency are recorded for reproducibility. Reasoning is a run
setting rather than an implicit benchmark property.

The evaluator reports accuracy, negative log-likelihood, latency, and 15-bin
equal-width top-label expected calibration error (ECE) overall and by primitive,
family, and domain. ECE uses successful rows with returned probability
distributions; API failures remain visible through the separate coverage counts.
Saved `raw.jsonl` artifacts can be re-aggregated without inference:

```bash
decision-bench summarize-run results/<run-name>
```

The Jev adapter uses OpenRouter's native Decisions API, mapping binary
classification to Noul, candidate selection to Choice, and ordinal scoring to
Score while preserving the same normalized benchmark rows.

Model-facing text is deterministically fitted to each adapter's input ceiling
without changing the frozen row, candidate IDs or order, ordinal values, or gold
answer. Jev uses its documented 32k-token `state` plus longest-question limit,
counted with a pinned Qwen3 tokenizer proxy. Because Jev's private tokenizer and
request serialization are not exposed, the adapter reserves 2,048 proxy tokens
by default and fits to an effective 29,952-token budget. Both the documented
limit and reserve are recorded and configurable. Native HF models use the
checkpoint tokenizer over the final chat-templated prompt and the configured `--max-length`.
Every raw prediction records the original and final token counts, exact derived
model-facing example, truncation policy, and whether any text field was shortened.

The native top-logprobs OpenRouter adapter asks compatible chat models to emit
one candidate symbol and requests the maximum 20 token alternatives. It reports
conditional native probabilities only when every valid candidate symbol is
present; incomplete distributions are errors rather than silently treating
missing candidates as zero. Rows with more than 20 candidates are recorded as
ineligible, while accuracy and probability metrics are reported for eligible
candidate-count groups.

Native Hugging Face decision-token checkpoints use `decision-bench run-hf`.
The adapter verifies the model manifest, compiles the checkpoint's declared
stable-slot prompt contract, batches CUDA inference, and applies a masked
softmax over every valid candidate token. This yields native full-distribution
metrics, including ECE, for candidate sets from 2 through 255. The model
directory must contain `manifest.json`, `serving.json`, `adapter/`, `tokenizer/`,
and `decision_embeddings.safetensors`.

```bash
uv run --extra hf decision-bench run-hf \
  task_specs/decisionbench-dev.toml \
  /path/to/model \
  results/bosun-epoch-0 \
  --batch-size 64
```

## Local GGUF decision API

Any chat-capable GGUF model can be exposed through the Jev-shaped
`POST /v1/systemone` contract. The model is asked for a probability per runtime
candidate; the server projects that distribution into `choice`, `score`, or
`noul` answers. This is a generic chat-model adapter, not a claim that the GGUF
model has native decision heads.

Install the optional runtime and start the server:

```bash
uv sync --extra gguf
uv run decision-bench serve-gguf /path/to/model.gguf --n-gpu-layers 99
```

The server also exposes `GET /health` and `GET /v1/models`. Multiple questions
in one `/v1/systemone` request are evaluated independently and returned under
their request keys.

Every completed run writes `raw.jsonl`, `summary.json`, and a content-addressed
`manifest.json`. Native HF runs additionally bind the task spec, dataset,
dataset manifest, model manifest, serving contract, base revision, and selected
checkpoint epoch into the saved metadata.

## Status

DecisionBench is a single frozen 23,900-row evaluation suite spanning nine
decision families and all three primitives, with candidate sets from 2 through
255. Task provenance, family, primitive, and reasoning metadata provide filtered
views over that one collection. The canonical payload, task specification, raw
model outputs, and content-addressed manifests define the leaderboard.
