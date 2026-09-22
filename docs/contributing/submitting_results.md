# Submit Results

Evaluate an immutable model revision against a pinned benchmark release. The completed local run
must contain `raw.jsonl`, `summary.json`, and `manifest.json`. You do not need to create a Hugging
Face artifact repository to submit a result.

## Stage the result record

Fork and clone the official results repository:

```bash
cd ..
gh repo fork Hanno-Labs/decision-bench-results --clone
cd decision-bench
```

Then stage the reviewed record from the DecisionBench checkout. `stage-result` verifies the local
manifest, raw-row accounting, metric arithmetic, and immutable identities before it writes the
compact result. Declare the model type, adapter, and probability source explicitly; use the exact
values written by the run rather than the generic CLI defaults:

```bash
decision-bench stage-result results/my-model ../decision-bench-results \
  --model-id org/model --model-revision COMMIT_SHA \
  --dataset-revision b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443 \
  --model-type decision-model \
  --adapter ADAPTER_ID \
  --probability-source PROBABILITY_SOURCE \
  --create-pr
```

The submitter chooses `--model-type` from this behavioral contract:

- `decision-model`: trained to answer all three DecisionBench primitives (`noul`, `choice`, and
  `score`) through a native decision output rather than free-form generation;
- `language-model`: produces text tokens that an adapter parses or constrains into a benchmark
  answer;
- `classifier`: a fixed-purpose class, relevance, or scalar scorer that was not trained across all
  three decision primitives, even when an adapter can apply it to every benchmark row.

Architecture names and API access do not determine the type. A DeBERTa checkpoint with a native
three-primitive decision head is a decision model; a proprietary text generator remains a language
model. Reviewers verify the declared type against the published serving and training contract.

`--create-pr` commits the generated record, pushes a branch to your fork, and opens a pull request
against `Hanno-Labs/decision-bench-results` using the GitHub CLI. Omit it if you prefer to inspect,
commit, and submit the generated files manually.

## Optional raw evidence

You may publish the complete run artifact and add its immutable URL with `--artifact-uri`. This is
useful for official runs, unusual adapters, and submissions where reviewers ask for additional
evidence. It is optional and is not a prerequisite for a result pull request. Do not publish private
inputs or provider responses without permission.

Before submission, run the results repository's local checks:

```bash
cd ../decision-bench-results
uv sync --locked --group dev
make check
```

Results CI validates schema, directory identity, revisions, duplicate records, and metric arithmetic.
The pull request also receives an automatic comparison with the pinned Jev and Luna reference
results. Maintainers review that report and may request additional evidence or rerun a suspicious
submission. The comparison is a review aid, not a score threshold.
