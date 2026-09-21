# Select Tasks and Views

DecisionBench is one frozen evaluation collection. Rows carry `task_name`, `family`, `domain`,
`primitive`, and source metadata, which define views rather than competing benchmark versions.

Use the Python API to select a view without changing row identity:

```python
from pathlib import Path
from decision_bench import get_benchmark

benchmark = get_benchmark(Path("task_specs/decisionbench-dev.toml"))
legal = benchmark.select(domain="legal")
routing = benchmark.select(family="routing_triage")
choice = benchmark.select(primitive="candidate_selection")
```

The primary full-suite leaderboard uses all 23,900 frozen rows. Task, family, domain, primitive,
candidate-count, and reasoning views remain filterable. Reasoning and ordinary task aggregates are
reported separately when a result contract distinguishes them.
