# Add a Benchmark

A benchmark is a named, curated collection of tasks intended to measure performance in a domain or
for a specific purpose. For example, a `DecisionBench Legal` benchmark could collect contract,
legal-retrieval, and citation-verification tasks.

## Before opening a pull request

1. State the benchmark's intended use and what a model score should mean.
2. Select existing tasks from the [task catalog](../overview/tasks.md). If a required task is
   missing, follow [Add a Task](adding_a_task.md) first.
3. Explain the inclusion and exclusion criteria. Prefer established, non-duplicative tasks with
   clear provenance and enough examples to support stable comparisons.
4. Choose a name and version. Once published, a benchmark release is immutable; later task changes
   require a new version.

## Implement the benchmark

Add a benchmark specification under `task_specs/` that records:

- the benchmark name, version, description, and languages; and
- the registered task names included in the collection.

Benchmarks do not own, copy, or republish task datasets. Each selected task already pins its own
dataset repository, revision, split, metadata, and transform.

Load the specification before submission and verify the expected task and row counts:

```python
from decision_bench import get_benchmark

benchmark = get_benchmark("DecisionBench Legal")
print(benchmark.spec.name, len(benchmark.examples))
print([task.metadata.name for task in benchmark.tasks])
```

Add tests for the specification, task membership, row count, and duplicate row IDs. Dataset revision
checks remain with each task.

## Publish it on the leaderboard

Benchmark acceptance and leaderboard publication are separate steps:

1. Open a DecisionBench pull request containing the task specification, documentation, and tests.
2. Evaluate the agreed reference models through their declared native contracts.
3. Submit those reviewed records to
   [`Hanno-Labs/decision-bench-results`](https://github.com/Hanno-Labs/decision-bench-results).
4. Add the benchmark to the leaderboard menu after its reference results merge.

The benchmark remains available through the Python API even if it is too specialized for the
headline leaderboard.
