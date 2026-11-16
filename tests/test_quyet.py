"""Public-API contract tests; no weights, network or package internals."""

import json
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from decision_bench import cli
from decision_bench.evaluate import _run_hf_batches
from decision_bench.models.hf import HFDecisionResponse
from decision_bench.models.quyet import (
    QuyetDecisionModel,
    QuyetOutputError,
    UnsupportedQuyetInput,
    build_quyet_request,
)
from decision_bench.results import _classify_errors
from decision_bench.schemas import Candidate, DecisionExample, Primitive


def example(count=3, primitive=Primitive.CANDIDATE_SELECTION, row_id="row"):
    return DecisionExample(
        row_id=row_id,
        task_name="contract",
        primitive=primitive,
        family="test",
        domain="test",
        instruction="Select the supplied candidate.",
        state={"text": "café", "ordered": [2, 1]},
        candidates=[
            Candidate(
                id=f"id-{i}",
                label=f"label-{count - i}",
                description=None if i == 0 else f"Description {i}",
                ordinal_value=count - i if primitive == Primitive.ORDINAL_SCORING else None,
            )
            for i in range(count)
        ],
        gold_candidate_id="id-0",
        gold_probabilities=[1.0] + [0.0] * (count - 1),
        source={"never_pass_to_model": True},
    )


class QuestionError(ValueError):
    pass


class PublicStub:
    def __init__(self):
        self.calls = []

    def raw_probabilities(self, requests, *, strict):
        assert strict is True
        self.calls.append(requests)
        if any(r["state"] == "over limit" for r in requests):
            raise QuestionError("prompt exceeds complete token limit; does not truncate")
        return [
            {
                "decision": [1 / len(r["questions"]["decision"]["criteria"])]
                * len(r["questions"]["decision"]["criteria"])
            }
            for r in requests
        ]


def model():
    m = QuyetDecisionModel.__new__(QuyetDecisionModel)
    m.max_options = 256
    m.max_score_options = 10
    m.question_error = QuestionError
    m.model = PublicStub()
    m.input_contract = {"truncation_policy": "reject"}
    return m


@pytest.mark.parametrize(
    "primitive,count",
    [
        (Primitive.BINARY_CLASSIFICATION, 2),
        (Primitive.ORDINAL_SCORING, 7),
        (Primitive.CANDIDATE_SELECTION, 151),
        (Primitive.CANDIDATE_SELECTION, 255),
    ],
)
def test_primitives_preserve_public_input_and_order(primitive, count):
    e = example(count, primitive)
    m = model()
    result = m.predict(e)
    request = build_quyet_request(e)
    assert request["state"] == e.state
    ordinal = primitive is Primitive.ORDINAL_SCORING
    ordered = sorted(e.candidates, key=lambda c: c.ordinal_value) if ordinal else e.candidates
    texts = [c.label if c.description is None else f"{c.label}: {c.description}" for c in ordered]
    criteria = texts if ordinal else dict(zip((c.id for c in ordered), texts, strict=True))
    assert request["questions"]["decision"] == {
        "type": "score" if ordinal else "choice",
        "instructions": e.instruction,
        "criteria": criteria,
    }
    assert "gold" not in json.dumps(request) and "never_pass" not in json.dumps(request)
    assert json.loads(result.request["requests_json"]) == [request]
    assert result.request["candidate_ids"] == [c.id for c in e.candidates]
    assert result.request["response_candidate_ids"] == [c.id for c in ordered]
    assert result.prediction.probabilities == pytest.approx([1 / count] * count)
    assert result.response["raw_probabilities"]["decision"] == [1 / count] * count


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"other": [0.5, 0.5]},
        {"decision": [1.0]},
        {"decision": [float("nan"), 0.0]},
        {"decision": [float("inf"), 0.0]},
        {"decision": [-0.1, 1.1]},
        {"decision": [0.1, 0.2]},
        {"decision": [True, False]},
        {"decision": [0.5, 0.5], "extra": []},
    ],
)
def test_malformed_distribution_keeps_raw_response(bad):
    m = model()
    m.model = SimpleNamespace(raw_probabilities=lambda requests, strict: [bad])
    with pytest.raises(QuyetOutputError) as caught:
        m.predict(example(2, Primitive.BINARY_CLASSIFICATION))
    assert caught.value.response["raw_probabilities"] is bad
    assert caught.value.request["strict"]


def test_wrong_response_row_count():
    m = model()
    m.model = SimpleNamespace(raw_probabilities=lambda requests, strict: [])
    with pytest.raises(QuyetOutputError, match="row count"):
        m.predict(example())


@pytest.mark.parametrize(
    "change,message",
    [
        ({"state": None}, "state"),
        ({"instruction": " "}, "instruction"),
    ],
)
def test_invalid_public_fields(change, message):
    with pytest.raises(UnsupportedQuyetInput, match=message):
        model().predict(example().model_copy(update=change))


def test_duplicate_labels_are_supported_with_id_keys():
    e = example()
    e.candidates[1].label = e.candidates[0].label
    result = model().predict(e)
    criteria = build_quyet_request(e)["questions"]["decision"]["criteria"]
    assert list(criteria) == [c.id for c in e.candidates]
    assert len(criteria) == 3
    assert result.request["candidate_labels"] == [c.label for c in e.candidates]


def test_candidate_text_convention():
    e = example()
    e.candidates[1].description = e.candidates[1].label
    criteria = build_quyet_request(e)["questions"]["decision"]["criteria"]
    assert list(criteria.values()) == ["label-3", "label-2", "label-1: Description 2"]


def test_ordinal_round_trip_shuffled_display_order():
    e = example(4, Primitive.ORDINAL_SCORING)
    e.candidates = [e.candidates[i] for i in (1, 3, 0, 2)]
    m = model()
    raw = {"decision": [0.1, 0.2, 0.3, 0.4]}
    m.model = SimpleNamespace(raw_probabilities=lambda requests, strict: [raw])
    result = m.predict(e)
    assert result.request["response_candidate_ids"] == ["id-3", "id-2", "id-1", "id-0"]
    assert result.prediction.probabilities == pytest.approx([0.3, 0.1, 0.4, 0.2])
    assert result.response["raw_probabilities"] is raw


def test_score_boundary():
    m = model()
    m.predict(example(10, Primitive.ORDINAL_SCORING))
    with pytest.raises(UnsupportedQuyetInput, match="score supports 2 to 10"):
        m.predict(example(11, Primitive.ORDINAL_SCORING))


def test_option_boundary():
    m = model()
    m.max_options = 10
    m.predict(example(10))
    with pytest.raises(UnsupportedQuyetInput, match="2 to 10"):
        m.predict(example(11))
    m.max_options = 256
    m.max_score_options = 10
    m.predict(example(255))
    # Harness itself caps at 255; exercise the adapter boundary without bypassing that in runs.
    e = example(255)
    e = e.model_copy(update={"candidates": [*e.candidates, Candidate(id="last", label="last")]})
    m.validate_example(e)
    e = e.model_copy(update={"candidates": [*e.candidates, Candidate(id="over", label="over")]})
    with pytest.raises(UnsupportedQuyetInput, match="256"):
        m.validate_example(e)


def test_context_rejection_does_not_discard_batch_peers():
    m = model()
    rows = [
        example(row_id="first"),
        example(row_id="long").model_copy(update={"state": "over limit"}),
        example(row_id="last"),
    ]
    result = m.predict_batch(rows)
    assert isinstance(result[0], HFDecisionResponse)
    assert isinstance(result[1], UnsupportedQuyetInput)
    assert isinstance(result[2], HFDecisionResponse)
    assert "over limit" in result[1].request["requests_json"]


def test_pinned_public_loader(monkeypatch, tmp_path):
    cfg = tmp_path / "quyet_config.json"
    cfg.write_text(
        json.dumps(
            {
                "limits": {"max_options": {"choice": 256}, "max_input_tokens": 32768},
                "temperatures": {"choice": 1.003},
            }
        )
    )
    calls = []
    fake = SimpleNamespace(
        __version__="1.2.0",
        QuestionError=QuestionError,
        load=lambda *a, **k: calls.append((a, k)) or PublicStub(),
    )
    monkeypatch.setitem(sys.modules, "quyet", fake)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(cfg))
    m = QuyetDecisionModel(model_repo="org/model", model_revision="a" * 40, device="cpu")
    assert calls == [
        (("org/model",), {"revision": "a" * 40, "device": "cpu", "prefix_cache": False})
    ]
    assert m.metadata["tokenizer_revision"] == "a" * 40
    assert m.input_contract["max_input_tokens"] == 32768
    with pytest.raises(ValueError, match="pinned"):
        QuyetDecisionModel(model_repo="org/model", model_revision="main")


def test_inspect_records_unsupported_and_resumes(tmp_path):
    m = model()
    rows = [example(row_id="ok"), example(row_id="long").model_copy(update={"state": "over limit"})]
    kwargs = dict(
        decision_model=m,
        batch_size=2,
        max_prompt_characters_per_batch=100000,
        metadata={"model_repo": "org/model", "adapter": "quyet-public-api-v1"},
    )
    summary = _run_hf_batches(rows, tmp_path / "run", **kwargs)
    raw = [json.loads(x) for x in (tmp_path / "run/raw.jsonl").read_text().splitlines()]
    assert summary["successful_rows"] == 1 and summary["error_rows"] == 1
    error = next(r for r in raw if r["status"] == "error")
    assert error["error_type"] == "UnsupportedQuyetInput" and error["request"]["strict"]
    assert _classify_errors(tmp_path / "run/raw.jsonl") == {"unsupported_rows": 1, "error_rows": 0}
    calls = len(m.model.calls)
    _run_hf_batches(rows, tmp_path / "run", **kwargs)
    assert len(m.model.calls) == calls + 1
    assert m.model.calls[-1][0]["state"] == "over limit"  # Existing harness retries errors only.


def test_cli_exposes_standard_fields_only():
    result = CliRunner().invoke(cli.app, ["run-quyet", "--help"])
    assert result.exit_code == 0
    assert "model-revision" in result.stdout
    assert "compact-fields" not in result.stdout


def test_runtime_error_is_not_unsupported():
    m = model()

    def fail(requests, strict):
        raise RuntimeError("device failure")

    m.model = SimpleNamespace(raw_probabilities=fail)
    with pytest.raises(QuyetOutputError) as caught:
        m.predict(example())
    assert caught.value.response["raw_probabilities"]["error_type"] == "RuntimeError"
