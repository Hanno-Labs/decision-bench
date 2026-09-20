"""Dataset loading with interchangeable local and Hugging Face backends."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset, load_dataset
from pydantic import BaseModel, ConfigDict, model_validator

from decision_bench.schemas import DecisionExample


class DatasetSpec(BaseModel):
    """Where benchmark rows live without coupling tasks to one storage backend."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["local", "huggingface"]
    split: str = "test"
    path: str | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None
    hf_config: str | None = None

    @model_validator(mode="after")
    def validate_backend(self) -> DatasetSpec:
        if self.backend == "local" and self.path is None:
            raise ValueError("local datasets require path")
        if self.backend == "huggingface" and (self.hf_repo is None or self.hf_revision is None):
            raise ValueError("Hugging Face datasets require hf_repo and immutable hf_revision")
        return self


def resolve_local_path(path: str, *, project_root: Path | None = None) -> Path:
    """Resolve a local dataset path through an optional external data root."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    configured_root = os.environ.get("DECISION_BENCH_DATA_DIR")
    if configured_root is not None:
        return Path(configured_root).expanduser() / candidate
    root = project_root if project_root is not None else Path.cwd()
    return root / candidate


def load_rows(spec: DatasetSpec, *, project_root: Path | None = None) -> Dataset:
    """Load raw rows from a local artifact or an immutable HF revision."""

    if spec.backend == "local":
        assert spec.path is not None
        path = resolve_local_path(spec.path, project_root=project_root)
        if not path.exists():
            raise FileNotFoundError(f"DecisionBench dataset not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported local dataset format: {suffix}")

    assert spec.hf_repo is not None
    assert spec.hf_revision is not None
    return load_dataset(
        spec.hf_repo,
        spec.hf_config,
        split=spec.split,
        revision=spec.hf_revision,
    )


def load_examples(
    spec: DatasetSpec,
    *,
    project_root: Path | None = None,
) -> Iterator[DecisionExample]:
    """Validate raw dataset rows against the benchmark contract."""

    rows = load_rows(spec, project_root=project_root)
    for row in rows:
        materialized = dict(_mapping(row))
        if "state_json" in materialized:
            materialized = _decode_storage_row(materialized)
        yield DecisionExample.model_validate(materialized)


def _mapping(row: Mapping[str, Any] | object) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError(f"dataset row is not a mapping: {type(row)!r}")
    return row


def _decode_storage_row(row: dict[str, Any]) -> dict[str, Any]:
    """Decode the flat Parquet storage contract into a benchmark example."""

    return {
        "row_id": row["row_id"],
        "task_name": row["task_name"],
        "primitive": row["primitive"],
        "family": row["family"],
        "domain": row["domain"],
        "instruction": row["instruction"],
        "state": json.loads(row["state_json"]),
        "candidates": json.loads(row["candidates_json"]),
        "gold_candidate_id": row["gold_candidate_id"],
        "gold_probabilities": row["gold_probabilities"],
        "source": json.loads(row["source_json"]),
    }
