# Run Evaluation

DecisionBench evaluates three output primitives:

| Primitive | Output contract |
|---|---|
| Binary classification | Probability over two exhaustive outcomes |
| Candidate selection | Probability over runtime-defined candidates |
| Ordinal scoring | Probability over ordered levels plus expected score |

Run a structured-output chat baseline:

```bash
decision-bench run-openrouter task_specs/decisionbench-dev.toml results/luna \
  --model openai/gpt-5.6-luna --reasoning-effort minimal
```

Other commands cover OpenRouter top-logprobs, Jev's Decisions API, self-hosted Jev-compatible
services, native DecisionBench checkpoints, public HF scorers, Nimble, and GGUF serving. Use
`decision-bench --help` for the current surface.

## Metrics

The primary metric is all-row accuracy: unsupported rows and errors count as misses. Each result
also reports supported-row accuracy, coverage, negative log-likelihood, 15-bin top-label ECE, and
latency. Views are available by task, family, domain, primitive, and candidate count.

## Comparability

Comparable results must share the same benchmark revision and declared view. Every record preserves
the model revision, adapter, probability source, prompt contract, eligibility definition,
truncation policy, row counts, and hashes of the durable raw artifacts. Hosted models without an
immutable provider revision are dated service snapshots, not reproducible weight snapshots.
