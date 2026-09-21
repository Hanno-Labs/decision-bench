# Results Ecosystem

DecisionBench follows the same separation that makes MTEB auditable:

```text
decision-bench              runtime, task metadata, adapters, docs, leaderboard code
decision-bench-results      reviewed compact result records and validation CI
Hugging Face results data   generated Parquet mirror for fast queries
HF Space leaderboard        interactive browser over reviewed records
HF bucket artifacts         complete row-level inputs, outputs, scores, errors, and hashes
```

The Git result record is reviewable and small. It cannot replace the durable raw artifact, and the
leaderboard is a generated view rather than a source of truth.
