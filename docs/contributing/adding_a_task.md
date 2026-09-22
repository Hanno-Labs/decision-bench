# Add a Task

A task is one dataset-backed evaluation unit using an existing DecisionBench
output primitive. Task discovery, transformation, import, generation, and
artifact validation live in
[`Hanno-Labs/decision-bench-data-gen`](https://github.com/Hanno-Labs/decision-bench-data-gen),
not in this runtime repository.

Every task must define a primitive, family, domain, provenance, commercial-compatible license,
candidate semantics, gold candidate, and gold probability distribution. Generation and evaluation
splits must be frozen before the task enters a benchmark release. Existing releases are never
silently rewritten.

## Workflow

1. Check the [task catalog](../overview/tasks.md) and open issues for overlap.
2. Choose an existing primitive: `binary_classification`, `candidate_selection`,
   or `ordinal_scoring`.
3. Add a pinned public source, reproducible importer or transform, task metadata,
   and tests to `decision-bench-data-gen`. You do **not** need to create a separate
   Hugging Face repository.
4. Normalize the source into `DecisionExample` rows and run:

   ```bash
   decision-bench-data-gen validate-task task.toml datasets/my-task/eval.parquet
   ```

5. Open one data-generation pull request with the validator report, license and
   provenance evidence, split and leakage notes, and available reference baselines.
6. After review, Hanno Labs publishes the normalized rows into the canonical
   DecisionBench dataset.

The complete implementation checklist and metadata format are in the
[data-generation contributor guide](https://github.com/Hanno-Labs/decision-bench-data-gen/blob/main/CONTRIBUTING.md).

## Task versus benchmark

Merging a task does not automatically alter a frozen benchmark or the headline
leaderboard. Benchmark inclusion is a separate review that pins the new dataset
revision and reference results. See [Add a Benchmark](adding_a_benchmark.md).

A new primitive changes the model-output contract rather than adding a task. Open
an issue here before implementing one.
