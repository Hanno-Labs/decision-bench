import hashlib
import json
from pathlib import Path

from decision_bench.results import ModelMetadata, ResultCache


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_and_load_result(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    raw = run / "raw.jsonl"
    raw.write_text(
        '{"row_id":"row-1","status":"ok"}\n'
        '{"row_id":"row-2","status":"error","error_type":"NetworkError"}\n'
    )
    summary = run / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "requested_rows": 2,
                "successful_rows": 1,
                "error_rows": 1,
                "task_spec_sha256": "a" * 64,
                "metrics": {
                    "overall": {
                        "rows": 1,
                        "accuracy": 1.0,
                        "mean_negative_log_likelihood": 0.1,
                        "expected_calibration_error": 0.2,
                        "mean_latency_seconds": 0.3,
                        "ece_bins": 15,
                    }
                },
            },
            sort_keys=True,
        )
    )
    manifest = run / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"files": {"raw.jsonl": _sha256(raw), "summary.json": _sha256(summary)}},
            sort_keys=True,
        )
    )
    cache = ResultCache(tmp_path / "results-repo")
    result_path = cache.stage_result(
        run,
        model=ModelMetadata(
            name="org/model",
            revision="revision",
            model_type="decision-model",
            adapter="test",
            probability_source="test",
        ),
        artifact_uri="hf://buckets/org/bucket/run",
        dataset_revision="dataset-revision",
    )

    assert result_path.is_file()
    [loaded] = cache.load_results()
    assert loaded.coverage == 0.5
    assert loaded.model.model_type == "decision-model"
    assert loaded.primary_accuracy == 0.5
    assert loaded.unsupported_rows == 0
    assert loaded.error_rows == 1
    assert loaded.artifact is not None
    assert loaded.artifact.uri == "hf://buckets/org/bucket/run"


def test_stage_result_without_published_artifact(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    raw = run / "raw.jsonl"
    raw.write_text('{"row_id":"row-1","status":"ok"}\n')
    summary = run / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "requested_rows": 1,
                "successful_rows": 1,
                "task_spec_sha256": "a" * 64,
                "metrics": {
                    "overall": {
                        "rows": 1,
                        "accuracy": 1.0,
                        "mean_negative_log_likelihood": 0.1,
                        "expected_calibration_error": 0.2,
                        "mean_latency_seconds": 0.3,
                    }
                },
            },
            sort_keys=True,
        )
    )
    manifest = run / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"files": {"raw.jsonl": _sha256(raw), "summary.json": _sha256(summary)}},
            sort_keys=True,
        )
    )
    cache = ResultCache(tmp_path / "results-repo")
    result_path = cache.stage_result(
        run,
        model=ModelMetadata(
            name="org/model",
            revision="revision",
            model_type="language-model",
            adapter="test",
            probability_source="test",
        ),
        dataset_revision="dataset-revision",
    )

    result_payload = json.loads(result_path.read_text())
    assert "artifact" not in result_payload
    [loaded] = cache.load_results()
    assert loaded.artifact is None


def test_stage_result_computes_non_reasoning_suite_ece_from_raw_rows(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    raw = run / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row_id": "row-1",
                        "status": "ok",
                        "family": "technical",
                        "latency_seconds": 0.1,
                        "negative_log_likelihood": 0.1,
                        "scored": {"correct": True, "probabilities": [0.9, 0.1]},
                    }
                ),
                json.dumps(
                    {
                        "row_id": "row-2",
                        "status": "ok",
                        "family": "legal",
                        "latency_seconds": 0.2,
                        "negative_log_likelihood": 0.2,
                        "scored": {"correct": False, "probabilities": [0.6, 0.4]},
                    }
                ),
                json.dumps(
                    {
                        "row_id": "row-3",
                        "status": "ok",
                        "family": "reasoning",
                        "latency_seconds": 0.3,
                        "negative_log_likelihood": 0.3,
                        "scored": {"correct": True, "probabilities": [0.8, 0.2]},
                    }
                ),
            ]
        )
        + "\n"
    )
    summary = run / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "requested_rows": 3,
                "successful_rows": 3,
                "task_spec_sha256": "a" * 64,
                "metrics": {
                    "overall": {
                        "rows": 3,
                        "accuracy": 2 / 3,
                        "mean_negative_log_likelihood": 0.2,
                        "expected_calibration_error": 0.2,
                        "mean_latency_seconds": 0.2,
                    }
                },
            },
            sort_keys=True,
        )
    )
    manifest = run / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"files": {"raw.jsonl": _sha256(raw), "summary.json": _sha256(summary)}},
            sort_keys=True,
        )
    )

    cache = ResultCache(tmp_path / "results-repo")
    cache.stage_result(
        run,
        model=ModelMetadata(
            name="org/model",
            revision="revision",
            model_type="classifier",
            adapter="test",
            probability_source="test",
        ),
        dataset_revision="dataset-revision",
    )

    [loaded] = cache.load_results()
    suite = loaded.views["suite:DecisionBench(eng, v1)"]
    assert suite.rows == 2
    assert suite.accuracy == 0.5
    assert suite.expected_calibration_error == 0.35
