"""MTEB-style benchmark composition over independently versioned tasks."""

from __future__ import annotations

import tomllib
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from decision_bench.data import DatasetSpec, load_rows
from decision_bench.schemas import DecisionExample, Primitive
from decision_bench.task_spec import BUILTIN_TASK_NAMES, DecisionTask, get_tasks


class BenchmarkSpec(BaseModel):
    """Describe a named, immutable collection of registered tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    languages: tuple[str, ...] = Field(min_length=1)
    tasks: tuple[str, ...] = Field(min_length=1)
    reference: str | None = None
    citation: str | None = None

    @field_validator("tasks")
    @classmethod
    def task_names_are_unique(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        if len(names) != len(set(names)):
            raise ValueError("benchmark task names must be unique")
        return names


def load_benchmark_spec(path: Path) -> BenchmarkSpec:
    """Load benchmark metadata and task membership from TOML."""

    with path.open("rb") as handle:
        return BenchmarkSpec.model_validate(tomllib.load(handle))


class Benchmark:
    """A named collection of tasks whose datasets remain independently pinned."""

    def __init__(
        self,
        *,
        spec: BenchmarkSpec,
        tasks: tuple[DecisionTask, ...],
        project_root: Path,
        spec_path: Path | None = None,
    ) -> None:
        self.spec = spec
        self.tasks = tasks
        self.project_root = project_root
        self.spec_path = spec_path

    @cached_property
    def examples(self) -> tuple[DecisionExample, ...]:
        """Load each immutable dataset once, then materialize its member tasks."""

        grouped: dict[
            tuple[str, str, str | None, str | None, str],
            list[DecisionTask],
        ] = defaultdict(list)
        dataset_by_key: dict[
            tuple[str, str, str | None, str | None, str],
            DatasetSpec,
        ] = {}
        for task in self.tasks:
            dataset = task.metadata.dataset
            grouped[dataset.cache_key].append(task)
            dataset_by_key[dataset.cache_key] = dataset

        examples: list[DecisionExample] = []
        for key, member_tasks in grouped.items():
            rows = tuple(
                dict(row)
                for row in load_rows(dataset_by_key[key], project_root=self.project_root)
            )
            for task in member_tasks:
                examples.extend(task.load_data(rows=rows))

        row_ids = [example.row_id for example in examples]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError(f"benchmark {self.spec.name!r} contains duplicate row IDs")
        return tuple(examples)

    @property
    def datasets(self) -> tuple[DatasetSpec, ...]:
        """Return the benchmark's unique immutable datasets in task order."""

        unique: dict[tuple[str, str, str | None, str | None, str], DatasetSpec] = {}
        for task in self.tasks:
            dataset = task.metadata.dataset
            unique.setdefault(dataset.cache_key, dataset)
        return tuple(unique.values())

    def select(
        self,
        *,
        task_name: str | None = None,
        family: str | None = None,
        domain: str | None = None,
        primitive: Primitive | str | None = None,
    ) -> Self:
        """Return a task-filtered benchmark without changing task identity."""

        normalized_primitive = Primitive(primitive) if primitive is not None else None
        selected = tuple(
            task
            for task in self.tasks
            if (task_name is None or task.metadata.name == task_name)
            and (family is None or task.metadata.family == family)
            and (domain is None or task.metadata.domain == domain)
            and (
                normalized_primitive is None
                or task.metadata.primitive is normalized_primitive
            )
        )
        return type(self)(
            spec=self.spec,
            tasks=selected,
            project_root=self.project_root,
            spec_path=self.spec_path,
        )


_BUILTIN_SPEC = BenchmarkSpec(
    name="DecisionBench (eng, v1)",
    version="1.0",
    description="Decision model evaluation across applied and reasoning tasks.",
    languages=("eng-Latn",),
    tasks=BUILTIN_TASK_NAMES,
    reference="https://huggingface.co/datasets/Hanno-Labs/decision-bench",
)

_BENCHMARK_REGISTRY: dict[str, BenchmarkSpec] = {
    "DecisionBench": _BUILTIN_SPEC,
    _BUILTIN_SPEC.name: _BUILTIN_SPEC,
}


def get_benchmark(
    benchmark: str | Path | BenchmarkSpec = "DecisionBench",
    *,
    project_root: Path | None = None,
) -> Benchmark:
    """Resolve a registered name, TOML specification, or BenchmarkSpec."""

    spec_path: Path | None = None
    if isinstance(benchmark, BenchmarkSpec):
        spec = benchmark
    elif isinstance(benchmark, Path) or Path(benchmark).suffix == ".toml":
        spec_path = Path(benchmark)
        spec = load_benchmark_spec(spec_path)
    else:
        try:
            spec = _BENCHMARK_REGISTRY[benchmark]
        except KeyError as error:
            available = ", ".join(sorted(_BENCHMARK_REGISTRY))
            raise KeyError(
                f"unknown benchmark {benchmark!r}; available benchmarks: {available}"
            ) from error

    root = project_root if project_root is not None else Path.cwd()
    return Benchmark(
        spec=spec,
        tasks=tuple(get_tasks(spec.tasks)),
        project_root=root,
        spec_path=spec_path,
    )


def get_benchmarks() -> list[BenchmarkSpec]:
    """Return each registered benchmark specification exactly once."""

    return list({spec.name: spec for spec in _BENCHMARK_REGISTRY.values()}.values())
