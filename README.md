# DecisionBench

**The evaluation ecosystem for decision models.**

DecisionBench aims to do for decision models what
[MTEB](https://github.com/embeddings-benchmark/mteb) does for embeddings: provide a shared
task taxonomy, reproducible evaluation, comparable results, and a clear path for adding models
and tasks.

Decision models turn documents and structured state into calibrated decisions. DecisionBench
tests that ability on one frozen suite of **23,900 examples**, spanning **nine decision families**,
**three primitives**, and candidate sets from **2 to 255**.

[Quick start](#quick-start) · [Benchmark](#benchmark) · [Metrics](#metrics-and-comparability) ·
[Models](#models) · [Results](#results) · [Contributing](#contributing)

## Quick start

The canonical dataset is currently private. You need access to
[`Hanno-Labs/decision-bench`](https://huggingface.co/datasets/Hanno-Labs/decision-bench) and an
authenticated Hugging Face environment.

From a checkout of this repository:

```bash
uv sync
hf auth login

# Validate the pinned task and inspect one benchmark row.
uv run decision-bench inspect task_specs/decisionbench-dev.toml --limit 1
```

Run a small end-to-end evaluation with an ordinary chat model through OpenRouter:

```bash
export OPENROUTER_API_KEY=...

uv run decision-bench run-openrouter \
  task_specs/decisionbench-dev.toml \
  results/luna-smoke \
  --model openai/gpt-5.6-luna \
  --reasoning-effort minimal \
  --smoke
```

Remove `--smoke` to evaluate the complete frozen suite. Every completed run writes:

```text
results/luna-smoke/
├── raw.jsonl       # Complete row-level inputs, outputs, probabilities, and errors
├── summary.json    # Aggregate metrics and filtered views
└── manifest.json   # Content hashes, model/data identity, and run configuration
```

Recompute every aggregate from the saved raw predictions without calling the model again:

```bash
uv run decision-bench summarize-run results/luna-smoke
```

## Benchmark

DecisionBench evaluates three model-independent output primitives:

| Primitive | Question answered | Output |
|---|---|---|
| **Binary classification** | Is this proposition true for the supplied state? | Probability over two exhaustive outcomes |
| **Candidate selection** | Which runtime-defined candidate best fits? | Probability over an unordered candidate set |
| **Ordinal scoring** | Where does the example fall on an ordered scale? | Probability over ordered levels plus an expected score |

The suite covers nine use-case families:

1. Document and record classification
2. Routing and triage
3. Rubric scoring and prioritization
4. Retrieval and verification
5. Bounded extraction
6. Entity alignment
7. Guardrails and moderation
8. Function, agent, and skill routing
9. Document workflows

### One suite, many views

The canonical dataset is one immutable evaluation collection. `task`, `family`, `domain`,
`primitive`, and reasoning metadata provide filtered views over that collection; there is no
separate “core” versus “expanded” leaderboard target.

Results can be reported for:

- the complete eligible benchmark;
- an individual primitive;
- a use-case family such as legal document workflows or routing and triage;
- a domain such as finance, privacy, online safety, or software agents; and
- reasoning and non-reasoning tracks, reported separately rather than blended into one opaque
  aggregate.

The checked-in task specification pins the hosted dataset to an immutable revision:

```toml
[dataset]
backend = "huggingface"
split = "eval"
hf_repo = "Hanno-Labs/decision-bench"
hf_revision = "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443"
```

Local Parquet payloads are supported for development, but official comparisons use the pinned
hosted revision.

## Metrics and comparability

The primary leaderboard metric is **accuracy over the frozen benchmark population**. Unsupported
rows and inference errors count as misses. This prevents a model from improving its headline score
by declining difficult examples.

Every result should also report:

| Metric | Why it matters |
|---|---|
| Supported-row accuracy | Quality where the adapter produced a valid answer |
| Coverage | Fraction of benchmark rows successfully evaluated |
| Negative log-likelihood | Probability assigned to the gold decision |
| 15-bin top-label ECE | Whether stated confidence matches observed correctness |
| Latency | Operational cost of producing the decision |

Metrics are available overall and by primitive, family, and domain. Calibration metrics are
computed only from rows with complete returned probability distributions; failures remain visible
through coverage and the primary all-row accuracy.

### Readout contracts are part of the result

DecisionBench supports models with different inference surfaces. A score is not fully described by
the model name alone, so every run records its probability source and eligibility rules.

| Readout | Probability source | Important boundary |
|---|---|---|
| Native decision-token model | Masked softmax over valid decision tokens | Full candidate distribution from 2–255 |
| Structured chat baseline | Model-generated JSON probability vector | Prompted probabilities, not native logits |
| Top-logprobs chat model | Native next-token log probabilities | Eligible only when every candidate token is returned |
| Jev-compatible decision API | Provider's Noul, Choice, or Score response | Provider tokenizer and internal scoring are private |

Leaderboard and comparison artifacts must therefore preserve the adapter, probability source,
coverage, eligible population, prompt contract, truncation policy, and immutable model revision.
DecisionBench does not hide these differences behind a single unsupported number.

## Models

DecisionBench adapters normalize different model APIs into the same row-level result schema.

| Model surface | Command | Intended use |
|---|---|---|
| Generic structured-output chat model | `run-openrouter` | Portable baseline across all three primitives |
| Native chat token logprobs | `run-openrouter-top-logprobs` | Native probabilities for candidate sets supported by the provider |
| Jev on OpenRouter | `run-jev-openrouter` | Native Noul, Choice, and Score API evaluation |
| Self-hosted Jev-compatible API | `run-jev-server` | Evaluate compatible local or hosted services |
| Native DecisionBench HF checkpoint | `run-hf` | Full 2–255 candidate distribution from decision tokens |
| Public HF decision/scorer models | `run-public-hf` | Pinned public-model adapters |
| Bespoke Nimble | `run-nimble-hf` | Pinned native candidate-logit contract |
| Chat-capable GGUF | `serve-gguf` | Expose a local model through the Jev-shaped API |

Use `uv run decision-bench --help` and `uv run decision-bench <command> --help` for the complete
command surface.

### Native DecisionBench checkpoints

Evaluate a compatible native checkpoint on CUDA:

```bash
uv run --extra hf decision-bench run-hf \
  task_specs/decisionbench-dev.toml \
  /path/to/model \
  results/native-model \
  --batch-size 64
```

The model directory must contain `manifest.json`, `serving.json`, `adapter/`, `tokenizer/`, and
`decision_embeddings.safetensors`. The adapter verifies the declared stable-slot prompt contract,
uses the checkpoint tokenizer over the final chat-templated prompt, and records the original and
final model-facing inputs for every row.

### Self-hosted GGUF models

Any chat-capable GGUF model can be exposed through the Jev-shaped `POST /v1/systemone` contract:

```bash
uv sync --extra gguf
uv run decision-bench serve-gguf /path/to/model.gguf --n-gpu-layers 99
```

The server projects a model-generated candidate distribution into `choice`, `score`, or `noul`
answers. It also exposes `GET /health` and `GET /v1/models`. This is a generic chat-model adapter,
not a claim that the underlying GGUF model has a native decision head.

## Results

DecisionBench treats complete, content-addressed run artifacts as the source of truth. A comparable
submission includes:

- the exact frozen dataset and task-spec revision;
- the immutable model revision and adapter configuration;
- every raw row-level input, output, probability, error, and latency;
- overall, supported-row, per-primitive, per-family, and per-domain metrics; and
- a manifest containing hashes for every durable artifact.

A hosted public leaderboard is not published yet. Until it is, verified run artifacts define the
result record and can be re-aggregated with `summarize-run`. The eventual leaderboard will be a
browser over those reproducible artifacts, not a separate source of truth.

## Contributing

DecisionBench is designed to grow along two independent axes.

### Add a model

Model adapters live in `src/decision_bench/models/`. A new adapter should:

1. preserve the frozen candidate IDs, order, values, and gold answer;
2. record its exact model-facing input and raw response;
3. return a complete probability distribution or an explicit unsupported/error result;
4. declare input limits, truncation, eligibility, and probability provenance; and
5. produce the standard raw, summary, and manifest artifacts.

### Add benchmark data

Dataset discovery, transformation, generation, import, and artifact validation live in the separate
[`Hanno-Labs/decision-bench-data-gen`](https://github.com/Hanno-Labs/decision-bench-data-gen)
repository. This repository owns evaluation, model adapters, scoring, and result artifacts.

New tasks must define their primitive, family, domain, provenance, licenses, candidate semantics,
and machine-checkable gold decision. Published frozen suites are never silently rewritten; additions
ship through a new immutable dataset revision and task specification.

## Project status

DecisionBench is currently private while the dataset, model adapters, result schema, and leaderboard
workflow are being stabilized. The benchmark code is Apache-2.0 licensed. Formal citation metadata
and the hosted leaderboard will be added before the public release.
