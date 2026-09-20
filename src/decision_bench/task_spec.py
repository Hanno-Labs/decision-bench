"""Task metadata loading."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from decision_bench.data import DatasetSpec
from decision_bench.schemas import Primitive


class TaskSpec(BaseModel):
    """Filterable benchmark metadata plus its external dataset reference."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    license: str = Field(min_length=1)
    languages: list[str] = Field(min_length=1)
    primitives: list[Primitive] = Field(min_length=1)
    families: list[str] = Field(min_length=1)
    domains: list[str] = Field(min_length=1)
    dataset: DatasetSpec


def load_task_spec(path: Path) -> TaskSpec:
    """Read and validate one TOML task specification."""

    with path.open("rb") as handle:
        return TaskSpec.model_validate(tomllib.load(handle))
