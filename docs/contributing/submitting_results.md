# Submit Results

Evaluate against a pinned benchmark release and preserve the complete `raw.jsonl`, `summary.json`,
and `manifest.json` in durable storage. Then stage a small reviewed record in a checkout of the
official results repository:

```bash
decision-bench stage-result results/my-model ../decision-bench-results \
  --model-id org/model --model-revision COMMIT_SHA \
  --artifact-uri hf://buckets/ORG/BUCKET/path/to/run \
  --dataset-revision b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443
```

Add `--create-pr` to commit the record, push a branch, and open a pull request using the GitHub CLI.
Results CI validates schema, directory identity, revisions, hashes, duplicate records, and metric
arithmetic. Pull requests receive an automatic score/coverage comparison.
