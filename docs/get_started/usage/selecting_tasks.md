# Select Tasks and Views

DecisionBench follows MTEB's hierarchy: tasks own pinned datasets, while benchmarks are named task
collections. Task metadata carries `family`, `domain`, and `primitive` for narrower views.

Use the Python API to select a view without changing row identity:

```python
from decision_bench import get_benchmark, get_tasks

benchmark = get_benchmark("DecisionBench")
legal = benchmark.select(domain="legal")
routing = benchmark.select(family="routing_triage")
choice = benchmark.select(primitive="candidate_selection")
finance_tasks = get_tasks(domains=["finance"])
```

The primary full-suite leaderboard uses all 23,900 frozen rows. Task, family, domain, primitive,
candidate-count, and reasoning views remain filterable. Reasoning and ordinary task aggregates are
reported separately when a result contract distinguishes them.
