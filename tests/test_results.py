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
            adapter="test",
            probability_source="test",
        ),
        artifact_uri="hf://buckets/org/bucket/run",
        dataset_revision="dataset-revision",
    )

    assert result_path.is_file()
    [loaded] = cache.load_results()
    assert loaded.coverage == 0.5
    assert loaded.primary_accuracy == 0.5
    assert loaded.unsupported_rows == 0
    assert loaded.error_rows == 1
