# Submit Results

Evaluate an immutable model revision against a pinned benchmark release. The completed run must
contain `raw.jsonl`, `summary.json`, and `manifest.json`.

## Publish the complete artifact

The row-level artifact must remain publicly readable after the pull request merges. A Hugging Face
dataset repository is the simplest option:

```bash
hf auth login
hf repos create USER/decision-bench-artifacts --type dataset --public --exist-ok
hf upload USER/decision-bench-artifacts \
  results/my-model \
  runs/ORG--MODEL/COMMIT_SHA \
  --type dataset \
  --commit-message "Add DecisionBench run for ORG/MODEL@COMMIT_SHA"
```

Record the Hub commit returned by `hf upload`. Confirm that the uploaded `manifest.json` and every
file it names are publicly downloadable at that immutable revision. The compact result record
points reviewers to this artifact rather than copying raw responses into Git.

## Stage the result record

Fork and clone the official results repository:

```bash
cd ..
gh repo fork Hanno-Labs/decision-bench-results --clone
cd decision-bench
```

Then stage the reviewed record from the DecisionBench checkout. Declare the adapter and probability
source explicitly; use the exact values written by the run artifact rather than the generic CLI
defaults:

```bash
decision-bench stage-result results/my-model ../decision-bench-results \
  --model-id org/model --model-revision COMMIT_SHA \
  --artifact-uri https://huggingface.co/datasets/USER/decision-bench-artifacts/tree/ARTIFACT_COMMIT/runs/ORG--MODEL/COMMIT_SHA \
  --dataset-revision b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443 \
  --adapter ADAPTER_ID \
  --probability-source PROBABILITY_SOURCE \
  --create-pr
```

`--create-pr` commits the generated record, pushes a branch to your fork, and opens a pull request
against `Hanno-Labs/decision-bench-results` using the GitHub CLI. Omit it if you prefer to inspect,
commit, and submit the generated files manually.

Before submission, run the results repository's local checks:

```bash
cd ../decision-bench-results
uv sync --locked --group dev
make check
```

Results CI validates schema, directory identity, revisions, hashes, duplicate records, and metric
arithmetic. The pull request also receives an automatic comparison with the pinned Jev and Luna
reference results. Maintainers use that report plus the public row-level artifact to review the
submission; it is a comparison aid, not a score threshold.
