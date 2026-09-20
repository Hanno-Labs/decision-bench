from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_bench.evaluate import (
    expected_calibration_error,
    refresh_summary,
    select_smoke_examples,
    summarize_raw,
)
from decision_bench.schemas import Candidate, DecisionExample, Primitive


def _example(row_id: str, family: str, primitive: Primitive, candidates: int) -> DecisionExample:
    candidate_rows = [
        Candidate(
            id=str(index),
            label=str(index),
            ordinal_value=float(index) if primitive is Primitive.ORDINAL_SCORING else None,
        )
        for index in range(candidates)
    ]
    return DecisionExample(
        row_id=row_id,
        task_name="task",
        primitive=primitive,
        family=family,
        domain="test",
        instruction="Choose.",
        state="state",
        candidates=candidate_rows,
        gold_candidate_id="0",
        gold_probabilities=[1.0, *([0.0] * (candidates - 1))],
    )


def test_smoke_selection_covers_family_primitive_and_large_choice() -> None:
    examples = [
        _example("a", "one", Primitive.BINARY_CLASSIFICATION, 2),
        _example("b", "two", Primitive.CANDIDATE_SELECTION, 255),
        _example("c", "three", Primitive.ORDINAL_SCORING, 4),
    ]
    selected = select_smoke_examples(examples)
    assert {example.row_id for example in selected} == {"a", "b", "c"}


def test_summary_uses_latest_attempt_per_row(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    records = [
        {"status": "error", "row_id": "a"},
        {
            "status": "ok",
            "row_id": "a",
            "primitive": "binary_classification",
            "family": "verification",
            "domain": "web",
            "scored": {"correct": True, "probabilities": [0.1, 0.9]},
            "negative_log_likelihood": 0.1,
            "latency_seconds": 1.0,
            "input_contract": {
                "policy_version": "balanced-text-cap-v1",
                "truncated": True,
            },
        },
    ]
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    summary = summarize_raw(raw_path)
    assert summary["successful_rows"] == 1
    assert summary["error_rows"] == 0
    assert summary["metrics"]["overall"]["expected_calibration_error"] == pytest.approx(0.1)
    assert summary["model_input_truncation"] == {
        "reported_rows": 1,
        "truncated_rows": 1,
        "policy_versions": ["balanced-text-cap-v1"],
    }


def test_expected_calibration_error_uses_top_label_confidence() -> None:
    records = [
        {"scored": {"correct": True, "probabilities": [0.1, 0.9]}},
        {"scored": {"correct": False, "probabilities": [0.6, 0.4]}},
    ]
    assert expected_calibration_error(records) == pytest.approx(0.35)


def test_refresh_summary_preserves_run_metadata(tmp_path: Path) -> None:
    record = {
        "status": "ok",
        "row_id": "a",
        "primitive": "binary_classification",
        "family": "verification",
        "domain": "web",
        "scored": {"correct": True, "probabilities": [0.1, 0.9]},
        "negative_log_likelihood": 0.1,
        "latency_seconds": 1.0,
    }
    (tmp_path / "raw.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "summary.json").write_text(json.dumps({"model": "model-id"}))
    summary = refresh_summary(tmp_path)
    assert summary["model"] == "model-id"
    assert summary["ece_definition"].startswith("15-bin")
    assert summary["metrics"]["overall"]["expected_calibration_error"] == pytest.approx(0.1)
