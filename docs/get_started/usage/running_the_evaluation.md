# Run Evaluation

DecisionBench evaluates three output primitives:

| Primitive | Output contract |
|---|---|
| Binary classification | Probability over two exhaustive outcomes |
| Candidate selection | Probability over runtime-defined candidates |
| Ordinal scoring | Probability over ordered levels plus expected score |

All evaluation commands execute through Inspect AI. Each benchmark row is an Inspect sample;
the native model adapter supplies its candidate probability vector through a custom model provider.
HF adapters retain their compatible-row batching, and hosted adapters retain their existing request
and retry contracts. Gold labels and source annotations are excluded from model-visible inputs.

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

## Check supported adapters first

Before adding anything, check [Supported adapters and models](../../overview/models.md). It names
the models and native output contracts already handled by DecisionBench, and the runner to use for
each one. If your model is listed, use that runner; you do not need to write an adapter.

If its native decision readout is not listed, [add a model adapter](../../contributing/adding_a_model.md)
that maps it to candidate probabilities. Start every new adapter with `--smoke`, inspect the saved
raw rows, and then rerun without it for the complete benchmark.

MoJev uses its packed candidate logits directly. Install the pinned adapter dependency and download
the immutable model revision before evaluating it:

```bash
uv sync --extra mojev
hf download MoLeMo-Lab/mojev \
  --revision 0c8695b6252f4205907433d4e196a94f032e60c3 \
  --local-dir models/mojev
decision-bench run-public-hf task_specs/decisionbench-dev.toml \
  models/mojev results/mojev \
  --model-type mojev \
  --model-repo MoLeMo-Lab/mojev \
  --model-revision 0c8695b6252f4205907433d4e196a94f032e60c3 \
  --smoke
```

Every run writes `raw.jsonl`, `summary.json`, `manifest.json`, and Inspect `.eval` logs under `inspect/`
to its output directory. The manifest hashes the Inspect logs as well as the canonical raw and
summary files. View the transcript, native requests/responses, sample scores, and aggregate metrics:

```bash
inspect view --log-dir results/my-model/inspect
```

Successful rows are skipped on resume; failed rows are retried. Each resumed invocation produces
a separate Inspect log covering its pending rows. The canonical summary combines the latest raw
record for each row across invocations. Historical raw rows retain their original provenance;
`inspect_rows` and `legacy_rows` show whether every final record was actually executed by Inspect.
Using Inspect does not itself grant a Hugging Face verified badge.

After a full run, follow [Submit Results](../../contributing/submitting_results.md) to validate the local run,
stage its compact result record, and open a result pull request.

## Opt into compact rows

Pass `--compact-fields` to any `run-*` evaluator to select the compact instruction,
state, and candidate descriptions on the same pinned public eval rows. The standard
fields remain the default. For a remote HF job, set `COMPACT_FIELDS=1`.

```bash
decision-bench run-public-hf task_specs/decisionbench-dev.toml \
  models/my-model results/my-model-compact \
  --model-type gliner25 --model-repo ORG/MODEL --model-revision COMMIT_SHA \
  --compact-fields --smoke
```

The run manifest records the pinned dataset revision and `compact_fields=true`.
Stage a completed compact run with `--tag compact` and that dataset revision.

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
