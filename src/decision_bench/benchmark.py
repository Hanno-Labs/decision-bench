"""Benchmark loading and filterable views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from decision_bench.data import load_examples
from decision_bench.schemas import DecisionExample, Primitive
from decision_bench.task_spec import TaskSpec, load_task_spec


@dataclass(frozen=True)
class Benchmark:
    """One immutable task specification and its validated examples."""

    spec_path: Path
    spec: TaskSpec
    examples: tuple[DecisionExample, ...]

    def select(
        self,
        *,
        task_name: str | None = None,
        family: str | None = None,
        domain: str | None = None,
        primitive: Primitive | str | None = None,
    ) -> Benchmark:
        """Return a view without changing row identity or benchmark metadata."""

        normalized_primitive = Primitive(primitive) if primitive is not None else None
        selected = tuple(
            example
            for example in self.examples
            if (task_name is None or example.task_name == task_name)
            and (family is None or example.family == family)
            and (domain is None or example.domain == domain)
            and (normalized_primitive is None or example.primitive is normalized_primitive)
        )
        return Benchmark(spec_path=self.spec_path, spec=self.spec, examples=selected)


def get_benchmark(spec_path: Path, *, project_root: Path | None = None) -> Benchmark:
    """Load a pinned task specification and validate every dataset row."""

    spec = load_task_spec(spec_path)
    root = project_root if project_root is not None else Path.cwd()
    examples = tuple(load_examples(spec.dataset, project_root=root))
    return Benchmark(spec_path=spec_path, spec=spec, examples=examples)
