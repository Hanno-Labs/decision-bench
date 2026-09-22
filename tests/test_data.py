from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from decision_bench.data import DatasetSpec, load_examples, resolve_local_path


def test_local_dataset_path_loads_without_hf_repository(tmp_path: Path) -> None:
    dataset_path = tmp_path / "local" / "eval.jsonl"
    dataset_path.parent.mkdir()
    dataset_path.write_text(
        json.dumps(
            {
                "row_id": "row-1",
                "task_name": "verification",
                "primitive": "binary_classification",
                "family": "retrieval_verification",
                "domain": "web",
                "instruction": "Is the claim supported?",
                "state": {"claim": "The document is current.", "document": "Updated today."},
                "candidates": [
                    {"id": "no", "label": "No"},
                    {"id": "yes", "label": "Yes"},
                ],
                "gold_candidate_id": "yes",
                "gold_probabilities": [0.0, 1.0],
                "source": {"dataset": "fixture"},
            }
        )
        + "\n"
    )
    spec = DatasetSpec(backend="local", path="local/eval.jsonl", split="eval")
    rows = list(load_examples(spec, project_root=tmp_path))
    assert len(rows) == 1
    assert rows[0].gold_candidate_id == "yes"


def test_configured_data_root_overrides_project_root(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    data_root = tmp_path / "data-root"
    monkeypatch.setenv("DECISION_BENCH_DATA_DIR", str(data_root))
    assert resolve_local_path("suite/eval.parquet", project_root=tmp_path / "wrong") == (
        data_root / "suite/eval.parquet"
    )


def test_hugging_face_dataset_uses_mteb_names_and_accepts_legacy_aliases() -> None:
    spec = DatasetSpec.model_validate(
        {
            "backend": "huggingface",
            "hf_repo": "org/task",
            "hf_revision": "abc123",
            "hf_config": "default",
            "split": "test",
        }
    )

    assert spec.path == "org/task"
    assert spec.revision == "abc123"
    assert spec.config_name == "default"
    assert "hf_repo" not in spec.model_dump()


def test_hugging_face_dataset_requires_immutable_revision() -> None:
    with pytest.raises(ValidationError, match="pinned revision"):
        DatasetSpec(path="org/task")
