from __future__ import annotations

from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from decision_bench.benchmark import Benchmark, BenchmarkSpec, get_benchmark
from decision_bench.data import DatasetSpec
from decision_bench.schemas import Primitive
from decision_bench.task_spec import DecisionTask, TaskMetadata

DATASET = DatasetSpec(backend="local", path="unused.jsonl", split="eval")


class ContractsTask(DecisionTask):
    metadata = TaskMetadata(
        name="ContractsFixture",
        description="Contract classification fixture.",
        dataset=DATASET,
        license="Apache-2.0",
        languages=("eng-Latn",),
        primitive=Primitive.CANDIDATE_SELECTION,
        family="document_workflows",
        domain="legal",
        source_task_name="contracts",
    )


class RoutingTask(DecisionTask):
    metadata = TaskMetadata(
        name="RoutingFixture",
        description="Routing fixture.",
        dataset=DATASET,
        license="Apache-2.0",
        languages=("eng-Latn",),
        primitive=Primitive.CANDIDATE_SELECTION,
        family="routing_triage",
        domain="support",
        source_task_name="routing",
    )


def _row(
    *,
    row_id: str,
    task_name: str,
    family: str,
    domain: str,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "task_name": task_name,
        "primitive": "candidate_selection",
        "family": family,
        "domain": domain,
        "instruction": "Choose one.",
        "state": {"text": "Example"},
        "candidates": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        "gold_candidate_id": "a",
        "gold_probabilities": [1.0, 0.0],
        "source": {"dataset": "fixture"},
    }


def test_benchmark_loads_shared_dataset_once_and_selects_tasks(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0

    def fake_load_rows(
        dataset: DatasetSpec,
        *,
        project_root: Path | None = None,
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        assert dataset == DATASET
        assert project_root == Path("/project")
        return [
            _row(
                row_id="row-1",
                task_name="contracts",
                family="document_workflows",
                domain="legal",
            ),
            _row(
                row_id="row-2",
                task_name="routing",
                family="routing_triage",
                domain="support",
            ),
        ]

    monkeypatch.setattr("decision_bench.benchmark.load_rows", fake_load_rows)
    benchmark = Benchmark(
        spec=BenchmarkSpec(
            name="Fixture",
            version="1.0",
            description="Fixture benchmark.",
            languages=("eng-Latn",),
            tasks=("ContractsFixture", "RoutingFixture"),
        ),
        tasks=(ContractsTask(), RoutingTask()),
        project_root=Path("/project"),
    )

    assert [row.row_id for row in benchmark.examples] == ["row-1", "row-2"]
    assert calls == 1
    assert [row.row_id for row in benchmark.select(domain="legal").examples] == ["row-1"]
    assert benchmark.select(family="missing").examples == ()


def test_builtin_benchmark_is_composed_from_pinned_tasks() -> None:
    benchmark = get_benchmark("DecisionBench")

    assert benchmark.spec.name == "DecisionBench (eng, v1)"
    assert len(benchmark.tasks) == 43
    assert len(benchmark.datasets) == 1
    assert benchmark.datasets[0].revision == "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443"


def test_toml_benchmark_spec_composes_registered_tasks() -> None:
    benchmark = get_benchmark(Path("task_specs/decisionbench-dev.toml"))

    assert benchmark.spec.name == "DecisionBench (eng, v1)"
    assert len(benchmark.tasks) == 43
    assert set(benchmark.spec.tasks) == {task.metadata.name for task in benchmark.tasks}
