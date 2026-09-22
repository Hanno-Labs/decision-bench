# Run Evaluation

DecisionBench evaluates three output primitives:

| Primitive | Output contract |
|---|---|
| Binary classification | Probability over two exhaustive outcomes |
| Candidate selection | Probability over runtime-defined candidates |
| Ordinal scoring | Probability over ordered levels plus expected score |

## Evaluate a Hugging Face model

The normal contribution path starts with a model published on the Hugging Face Hub at an immutable
commit revision. Download that exact revision rather than evaluating a moving branch:

```bash
hf download ORG/MODEL \
  --revision COMMIT_SHA \
  --local-dir models/my-model
```

DecisionBench evaluates model-native decision probabilities. It does not treat arbitrary generated
text as a comparable score. If the model publishes the native DecisionBench checkpoint contract,
run it directly:

```bash
decision-bench run-hf \
  task_specs/decisionbench-dev.toml \
  models/my-model \
  results/my-model \
  --smoke
```

For another model architecture, first [add a model adapter](../../contributing/adding_a_model.md)
that maps the model's native output to candidate probabilities. Existing public model adapters use
`run-public-hf`; `decision-bench run-public-hf --help` lists their model-specific arguments. Start
with `--smoke`, inspect the saved raw rows, and then rerun without it for the complete benchmark.

Every run writes `raw.jsonl`, `summary.json`, and `manifest.json` to its output directory. After a
full run, follow [Submit Results](../../contributing/submitting_results.md) to validate the local run,
stage its compact result record, and open a result pull request.

## Evaluate a hosted API model

Hosted or closed models use their provider-specific adapter. For example, run a structured-output
chat baseline through OpenRouter:

```bash
decision-bench run-openrouter task_specs/decisionbench-dev.toml results/luna \
  --model openai/gpt-5.6-luna --reasoning-effort minimal
```

Other commands cover OpenRouter top-logprobs, Jev's Decisions API, self-hosted Jev-compatible
services, Nimble, and GGUF serving. Use `decision-bench --help` for the current surface. Hosted
results must record the provider route and request settings because they do not have an immutable
weight revision.

## Metrics

The primary metric is all-row accuracy: unsupported rows and errors count as misses. Each result
also reports supported-row accuracy, coverage, negative log-likelihood, 15-bin top-label ECE, and
latency. Views are available by task, family, domain, primitive, and candidate count.

## Comparability

Comparable results must share the same benchmark revision and declared view. Every record preserves
the model revision, adapter, probability source, prompt contract, eligibility definition,
truncation policy, and row counts. A submission may additionally link a content-addressed raw
artifact. Hosted models without an immutable provider revision are dated service snapshots, not
reproducible weight snapshots.
