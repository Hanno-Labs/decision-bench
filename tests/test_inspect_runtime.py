"""Exercise real Inspect execution without downloading weights or calling paid APIs."""

import json
import math
from pathlib import Path
from typing import ClassVar

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log

from decision_bench.evaluate import _run_evaluation, _run_hf_batches, _successful_record
from decision_bench.inspect_runtime import (
    UnsupportedCandidateCount,
    candidate_target,
    decision_sample,
    run_inspect,
)
from decision_bench.models.hf import HFDecisionResponse
from decision_bench.models.openrouter_top_logprobs import OpenRouterTopLogprobsDecisionModel
from decision_bench.results import ModelMetadata, ResultCache
from decision_bench.schemas import Candidate, DecisionExample, DecisionPrediction, Primitive


def example(row_id: str, count: int = 2, *, ordinal: bool = False) -> DecisionExample:
    return DecisionExample(
        row_id=row_id,
        task_name="native-test",
        primitive=(
            Primitive.ORDINAL_SCORING
            if ordinal
            else Primitive.BINARY_CLASSIFICATION
            if count == 2
            else Primitive.CANDIDATE_SELECTION
        ),
        family="test",
        domain="multilingual",
        instruction="Choose a candidate",
        state={"text": "候補を選択"},
        candidates=[
            Candidate(id=f"id-{index}", label=str(index), ordinal_value=index if ordinal else None)
            for index in range(count)
        ],
        gold_candidate_id="id-0",
        gold_probabilities=[0.7, 0.3] + [0.0] * (count - 2),
        source={"hidden_label": "id-0"},
    )


class NativeAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.batches: list[list[str]] = []
        self.failures: set[str] = set()
        self.unsupported: set[str] = set()
        self.correct: set[str] = set()

    def validate_example(self, row: DecisionExample) -> None:
        if row.row_id in self.unsupported:
            raise UnsupportedCandidateCount("candidate set unsupported")

    def prompt_characters(self, row: DecisionExample) -> int:
        return len(row.instruction)

    def predict(self, row: DecisionExample) -> HFDecisionResponse:
        self.calls.append(row.row_id)
        if row.row_id in self.failures:
            raise ConnectionError("fixture transport failure")
        probabilities = [0.8, 0.2] if row.row_id in self.correct else [0.4, 0.6]
        probabilities += [0.0] * (len(row.candidates) - 2)
        return HFDecisionResponse(
            prediction=DecisionPrediction(probabilities=probabilities),
            request={"candidate_ids": [item.id for item in row.candidates]},
            response={"probabilities": probabilities},
            latency_seconds=0.125,
        )

    def predict_batch(self, rows: list[DecisionExample]) -> list[HFDecisionResponse]:
        self.batches.append([row.row_id for row in rows])
        return [self.predict(row) for row in rows]


def logs(run: Path) -> list[EvalLog]:
    return [read_eval_log(str(path)) for path in sorted((run / "inspect").glob("*.eval"))]


def metrics(log: EvalLog) -> dict[str, float]:
    assert log.results is not None
    return {
        key: value.value for score in log.results.scores for key, value in score.metrics.items()
    }


@pytest.mark.parametrize("gold,other", [("A-B", "ab"), ("the", "THE"), ("one!", "one")])
def test_exact_match_preserves_distinct_candidate_ids(
    tmp_path: Path, gold: str, other: str
) -> None:
    row = example("normalization").model_copy(
        update={
            "candidates": [Candidate(id=gold, label="Gold"), Candidate(id=other, label="Other")],
            "gold_candidate_id": gold,
        }
    )
    summary = _run_evaluation(
        [row], tmp_path, decision_model=NativeAdapter(), concurrency=1, metadata={}
    )
    [log] = logs(tmp_path)
    assert log.samples is not None
    [sample] = log.samples
    assert sample.scores is not None
    assert sample.scores["exact"].value == "I"
    assert sample.output.completion == candidate_target(other)
    assert summary["benchmark_accuracy_counting_unsupported_as_incorrect"] == 0.0


@pytest.mark.parametrize("count,ordinal", [(2, False), (4, True), (255, False)])
def test_inspect_native_probability_and_metric_parity(
    tmp_path: Path,
    count: int,
    ordinal: bool,
) -> None:
    row = example("row", count, ordinal=ordinal)
    adapter = NativeAdapter()
    summary = _run_evaluation(
        [row],
        tmp_path,
        decision_model=adapter,
        concurrency=1,
        metadata={"model": "fixture", "task_spec_sha256": "a" * 64},
    )
    [record] = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text().splitlines()]
    assert record["scored"]["probabilities"] == [0.4, 0.6] + [0.0] * (count - 2)
    assert record["scored"]["selected_candidate_id"] == "id-1"
    assert record["negative_log_likelihood"] == pytest.approx(
        -0.7 * math.log(0.4) - 0.3 * math.log(0.6)
    )
    if ordinal:
        assert record["scored"]["expected_ordinal_score"] == pytest.approx(0.6)
    [log] = logs(tmp_path)
    assert log.status == "success"
    assert log.eval.model == "decision-bench/fixture"
    assert log.samples is not None
    [sample] = log.samples
    payload = json.loads(str(sample.input))
    assert "gold_candidate_id" not in payload
    assert "gold_probabilities" not in payload
    assert "source" not in payload
    assert sample.output.completion == candidate_target("id-1")
    assert sample.output.metadata is not None
    assert (
        sample.output.metadata["raw_record"]["scored"]["probabilities"]
        == record["scored"]["probabilities"]
    )
    assert sample.scores is not None and set(sample.scores) == {"exact"}
    assert log.plan is not None
    assert [step.solver for step in log.plan.steps] == ["generate"]
    assert any(isinstance(event, ModelEvent) and event.call is not None for event in sample.events)
    values = metrics(log)
    assert values["accuracy"] == summary["benchmark_accuracy_counting_unsupported_as_incorrect"]
    assert values["negative_log_likelihood"] == pytest.approx(record["negative_log_likelihood"])
    assert values["expected_calibration_error"] == pytest.approx(0.6)
    assert summary["inspect_provenance_complete"]
    ResultCache(tmp_path / "cache").stage_result(
        tmp_path,
        model=ModelMetadata(
            name="fixture",
            revision="pinned",
            model_type="decision-model",
            adapter="test",
            probability_source="native",
        ),
        dataset_revision="pinned-dataset",
    )
    path = next((tmp_path / "inspect").glob("*.eval"))
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        ResultCache(tmp_path / "cache").stage_result(
            tmp_path,
            model=ModelMetadata(
                name="fixture",
                revision="pinned",
                model_type="decision-model",
                adapter="test",
                probability_source="native",
            ),
            dataset_revision="pinned-dataset",
        )


def test_inspect_accounts_for_errors_and_resume(tmp_path: Path) -> None:
    adapter = NativeAdapter()
    adapter.failures = {"retry"}
    adapter.unsupported = {"unsupported"}
    adapter.correct = {"ok"}
    rows = [example("ok"), example("retry"), example("unsupported", 255)]
    first = _run_evaluation(rows, tmp_path, decision_model=adapter, concurrency=2, metadata={})
    assert first["successful_rows"] == 1
    assert first["error_rows"] == 2
    assert first["coverage"] == pytest.approx(1 / 3)
    assert first["benchmark_accuracy_counting_unsupported_as_incorrect"] == pytest.approx(1 / 3)
    [first_log] = logs(tmp_path)
    assert metrics(first_log)["coverage"] == pytest.approx(1 / 3)
    assert metrics(first_log)["accuracy"] == pytest.approx(1 / 3)
    assert len(first_log.samples or []) == 3
    assert "unsupported" not in adapter.calls
    adapter.failures.clear()
    second = _run_evaluation(rows, tmp_path, decision_model=adapter, concurrency=2, metadata={})
    assert adapter.calls.count("ok") == 1
    assert adapter.calls.count("retry") == 2
    assert second["successful_rows"] == 2
    assert second["error_rows"] == 1
    assert second["inspect_rows"] == 3
    assert second["benchmark_accuracy_counting_unsupported_as_incorrect"] == pytest.approx(1 / 3)
    assert len(logs(tmp_path)) == 2
    assert sorted(len(log.samples or []) for log in logs(tmp_path)) == [2, 3]


def test_inspect_preserves_native_batch_plan_at_low_concurrency(tmp_path: Path) -> None:
    adapter = NativeAdapter()
    rows = [example("a"), example("b"), example("c", 4), example("d", 255)]
    tmp_path.mkdir(exist_ok=True)
    run_inspect(
        rows,
        tmp_path,
        adapter=adapter,
        record_result=_successful_record,
        concurrency=1,
        metadata={},
        batches=[rows[:2], rows[2:3], rows[3:]],
    )
    assert adapter.batches == [["a", "b"], ["c"], ["d"]]
    assert len(logs(tmp_path)[0].samples or []) == 4


def test_hf_batch_failures_and_unsupported_rows_are_logged(tmp_path: Path) -> None:
    adapter = NativeAdapter()
    adapter.failures = {"batch-fails"}
    adapter.unsupported = {"unsupported"}
    rows = [example("batch-fails"), example("peer"), example("unsupported", 255)]
    summary = _run_hf_batches(
        rows,
        tmp_path,
        decision_model=adapter,
        batch_size=2,
        max_prompt_characters_per_batch=1000,
        metadata={},
    )
    assert adapter.batches == [["batch-fails", "peer"]]
    assert summary["successful_rows"] == 0
    assert summary["error_rows"] == 3
    assert summary["coverage"] == 0
    assert summary["benchmark_accuracy_counting_unsupported_as_incorrect"] == 0
    assert len(logs(tmp_path)[0].samples or []) == 3


def test_sample_preserves_candidate_order_and_excludes_labels() -> None:
    row = example("sample", 255)
    payload = json.loads(str(decision_sample(row).input))
    assert [item["id"] for item in payload["candidates"]] == [item.id for item in row.candidates]
    assert set(payload).isdisjoint({"gold_candidate_id", "gold_probabilities", "source"})


def test_legacy_rows_are_not_relabelled_inspect(tmp_path: Path) -> None:
    adapter = NativeAdapter()
    row = example("legacy")
    record = _successful_record(row, adapter.predict(row))
    (tmp_path / "raw.jsonl").write_text(json.dumps(record) + "\n")
    summary = _run_evaluation(
        [row, example("new")], tmp_path, decision_model=adapter, concurrency=1, metadata={}
    )
    assert summary["legacy_rows"] == 1
    assert summary["inspect_rows"] == 1
    assert not summary["inspect_provenance_complete"]
    assert len(logs(tmp_path)[0].samples or []) == 1


def test_duplicate_rows_rejected_before_inspect(tmp_path: Path) -> None:
    row = example("duplicate")
    with pytest.raises(ValueError, match="unique"):
        _run_evaluation(
            [row, row], tmp_path, decision_model=NativeAdapter(), concurrency=1, metadata={}
        )


def test_resume_recovers_raw_row_missing_from_inspect_log(tmp_path: Path) -> None:
    adapter = NativeAdapter()
    row = example("interrupted")
    record = _successful_record(row, adapter.predict(row))
    record["evaluation_framework"] = "inspect-ai"
    (tmp_path / "raw.jsonl").write_text(json.dumps(record) + "\n")
    summary = _run_evaluation([row], tmp_path, decision_model=adapter, concurrency=1, metadata={})
    assert adapter.calls == ["interrupted", "interrupted"]
    assert summary["inspect_provenance_complete"]
    assert len(logs(tmp_path)[0].samples or []) == 1


def test_resume_rejects_changed_inputs_and_model_metadata(tmp_path: Path) -> None:
    adapter = NativeAdapter()
    row = example("pinned")
    _run_evaluation(
        [row], tmp_path, decision_model=adapter, concurrency=1, metadata={"model": "one"}
    )
    with pytest.raises(ValueError, match="metadata changed"):
        _run_evaluation(
            [row], tmp_path, decision_model=adapter, concurrency=1, metadata={"model": "two"}
        )
    changed = row.model_copy(update={"state": "different"})
    with pytest.raises(ValueError, match="different benchmark input"):
        _run_evaluation(
            [changed], tmp_path, decision_model=adapter, concurrency=1, metadata={"model": "one"}
        )


def test_top_logprobs_keeps_unsupported_rows_in_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = NativeAdapter()
    adapter.correct = {"eligible"}
    model = object.__new__(OpenRouterTopLogprobsDecisionModel)
    model.top_logprobs = 2
    monkeypatch.setattr(model, "predict", adapter.predict)
    summary = _run_evaluation(
        [example("eligible"), example("too-many", 255)],
        tmp_path,
        decision_model=model,
        concurrency=1,
        metadata={},
    )
    assert adapter.calls == ["eligible"]
    assert summary["benchmark_accuracy_counting_unsupported_as_incorrect"] == 0.5
    assert len(logs(tmp_path)[0].samples or []) == 2


def test_inspect_checkpoints_include_complete_flushed_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    import decision_bench.evaluate as evaluate

    adapter = NativeAdapter()
    predict = adapter.predict

    def slow_first(row: DecisionExample) -> HFDecisionResponse:
        if row.row_id == "first":
            time.sleep(1.05)
        return predict(row)

    monkeypatch.setattr(adapter, "predict", slow_first)
    checkpoint = tmp_path / "checkpoint"
    prefixes: list[list[str]] = []
    write_checkpoint = evaluate._write_raw_checkpoint

    def capture(raw: Path, destination: Path) -> None:
        write_checkpoint(raw, destination)
        prefixes.append(
            [
                json.loads(line)["row_id"]
                for line in (destination / "raw.jsonl").read_text().splitlines()
            ]
        )

    monkeypatch.setattr(evaluate, "_write_raw_checkpoint", capture)
    run = tmp_path / "run"
    _run_evaluation(
        [example("first"), example("second")],
        run,
        decision_model=adapter,
        concurrency=1,
        metadata={"model": "fixture"},
        checkpoint_dir=checkpoint,
        checkpoint_interval_seconds=1,
    )
    assert prefixes[0] == []
    assert ["first"] in prefixes
    assert prefixes[-1] == ["first", "second"]
    assert (checkpoint / "raw.jsonl").read_bytes() == (run / "raw.jsonl").read_bytes()


@pytest.mark.parametrize("model_type", ["gliner25", "tev1"])
def test_public_hf_validation_rows_remain_inspect_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_type: str
) -> None:
    import decision_bench.evaluate as evaluate

    adapter = NativeAdapter()
    adapter.unsupported.add("unsupported")
    monkeypatch.setattr(
        adapter, "metadata", {"model_repo": "fixture", "model_revision": "pin"}, raising=False
    )
    constructor = "GLiNER25DecideModel" if model_type == "gliner25" else "Tev1HFDecisionModel"
    monkeypatch.setattr(evaluate, constructor, lambda **kwargs: adapter)
    summary = evaluate.run_public_hf_evaluation(
        [example("supported"), example("unsupported", 255)],
        tmp_path,
        model_dir=tmp_path,
        model_type=model_type,
        model_repo="fixture",
        model_revision="pin",
        base_revision=None,
        expected_weights_sha256=None,
        batch_size=1,
        max_prompt_characters_per_batch=1000,
        attn_implementation="eager",
    )
    assert summary["inspect_rows"] == summary["requested_rows"] == 2
    assert summary["ineligible_rows"] == 1
    assert summary["inspect_provenance_complete"]
    assert adapter.calls == ["supported"]
    [log] = logs(tmp_path)
    assert log.samples is not None
    assert {sample.id for sample in log.samples} == {"supported", "unsupported"}


def test_system_one_http_validation_rows_remain_inspect_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decision_bench.evaluate as evaluate

    class EndpointAdapter(NativeAdapter):
        metadata: ClassVar[dict[str, str]] = {
            "model": "fixture",
            "eligibility_definition": "fixture input limits",
        }

        def __enter__(self) -> "EndpointAdapter":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    adapter = EndpointAdapter()
    adapter.unsupported.add("unsupported")
    monkeypatch.setattr(evaluate, "SystemOneHTTPDecisionModel", lambda **kwargs: adapter)
    summary = evaluate.run_system_one_http_evaluation(
        [example("supported"), example("unsupported", 255)],
        tmp_path,
        base_url="http://fixture",
        model="fixture",
        model_repo="fixture",
        model_revision="pin",
        serving_bundle_sha256="a" * 64,
        inference_image="fixture@sha256:pin",
        concurrency=1,
    )
    assert summary["inspect_provenance_complete"]
    assert summary["inspect_rows"] == 2
    assert summary["eligible_rows"] == summary["ineligible_rows"] == 1
    assert summary["ineligible_definition"] == "fixture input limits"
    assert adapter.calls == ["supported"]
