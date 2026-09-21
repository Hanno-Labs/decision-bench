# Get Started

The checked-in task specification pins the canonical dataset to an immutable revision. Inspect it
before running inference:

```bash
decision-bench inspect task_specs/decisionbench-dev.toml --limit 1
```

A completed run contains:

```text
results/model-name/
├── raw.jsonl
├── summary.json
└── manifest.json
```

`raw.jsonl` is the auditable source of truth. `summary.json` is a derived aggregate, and
`manifest.json` binds both with content hashes.
