# Add a Task

A task is one independently versioned, dataset-backed evaluation unit using an existing
DecisionBench output primitive. Like MTEB, each task owns its dataset reference, immutable revision,
metadata, and any transform needed to produce the common evaluation schema. A task does not have to
be copied into the Hanno Labs dataset.

Every task must define a primitive, family, domain, provenance, commercial-compatible license,
candidate semantics, gold candidate, and gold probability distribution. Generation and evaluation
splits must be frozen before the task enters a benchmark release. Existing releases are never
silently rewritten.

## Workflow

1. Check the [task catalog](../overview/tasks.md) and open issues for overlap.
2. Choose an existing primitive: `binary_classification`, `candidate_selection`,
   or `ordinal_scoring`.
3. Publish the task's data in a public Hugging Face dataset repository and pin a commit SHA. The
   source may already use the DecisionBench schema or may need a deterministic transform.
4. Add a registered `DecisionTask` with its pinned dataset and metadata:

   ```python
   from decision_bench import DatasetSpec, DecisionTask, TaskMetadata, register_task
   from decision_bench.schemas import Primitive

   @register_task
   class MyTask(DecisionTask):
       metadata = TaskMetadata(
           name="MyTask",
           description="What the task measures.",
           dataset=DatasetSpec(path="org/my-task", revision="<commit-sha>", split="test"),
           license="Apache-2.0",
           languages=("eng-Latn",),
           primitive=Primitive.CANDIDATE_SELECTION,
           family="routing_triage",
           domain="biopharma",
       )
   ```

   Add the definition to `src/decision_bench/task_spec.py` so importing DecisionBench registers it.
   The registry is the runtime task catalog; the benchmark specification below only names entries
   from that catalog.

5. Override `dataset_transform()` only when source rows need conversion to `DecisionExample`.
6. Open a DecisionBench pull request with task tests, license and provenance evidence, split and
   leakage notes, and available reference baselines.

The existing DecisionBench 1.0 tasks happen to share one consolidated dataset release. That is a
property of that release, not a requirement for new contributions.

## Task versus benchmark

Merging a task does not automatically alter a frozen benchmark or the headline leaderboard.
Benchmark inclusion is a separate review that composes registered task names and reference results.
See [Add a Benchmark](adding_a_benchmark.md).
