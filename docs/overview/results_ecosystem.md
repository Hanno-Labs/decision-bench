# Results Ecosystem

DecisionBench follows the same separation that makes MTEB auditable:

```text
decision-bench              runtime, task metadata, adapters, docs, leaderboard code
decision-bench-results      reviewed compact result records and validation CI
Hugging Face results data   generated Parquet mirror for fast queries
HF Space leaderboard        interactive browser over reviewed records
Optional artifact storage  complete row-level inputs, outputs, scores, errors, and hashes
```

The Git result record is the reviewed source of truth for the leaderboard. Complete raw artifacts
are retained for official runs and may be attached to contributor submissions when useful, but they
are not required. The leaderboard is a generated view of the reviewed records.
