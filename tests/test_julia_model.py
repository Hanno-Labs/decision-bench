"""The Julia-1 adapter preserves candidate identity and its native option readout."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from decision_bench.models import julia_hf
from decision_bench.models.julia_hf import (
    HEAD_LENGTH,
    MAX_CANDIDATES,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    WEIGHTS_SHA256,
    JuliaHFDecisionModel,
    UnsupportedJuliaInput,
)
from decision_bench.schemas import Candidate, DecisionExample, Primitive


def _example(
    count: int,
    primitive: Primitive,
    *,
    ids: list[str] | None = None,
    labels: list[str] | None = None,
) -> DecisionExample:
    candidate_ids = ids or [f"candidate-{index}" for index in range(count)]
    candidate_labels = labels or [f"Option {index}" for index in range(count)]
    candidates = [
        Candidate(
            id=candidate_id,
            label=candidate_labels[index],
            ordinal_value=float(index) if primitive is Primitive.ORDINAL_SCORING else None,
        )
        for index, candidate_id in enumerate(candidate_ids)
    ]
    return DecisionExample(
        row_id=f"julia-{count}-{primitive.value}",
        task_name="routing",
        primitive=primitive,
        family="routing_triage",
        domain="financial",
        instruction="Choose the option named in the state.",
        state={"best_option": candidate_labels[0]},
        candidates=candidates,
        gold_candidate_id=candidates[0].id,
        gold_probabilities=[1.0] + [0.0] * (count - 1),
    )


class _FakeEngine:
    def __init__(
        self,
        logits: list[list[float]] | None = None,
        *,
        token_count: int = 123,
        encoding_error: str | None = None,
    ) -> None:
        self.logits_by_row = logits
        self.token_count = token_count
        self.encoding_error = encoding_error
        self.rows: list[dict[str, Any]] = []

    def logits(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        self.rows = rows
        if self.logits_by_row is not None:
            return self.logits_by_row
        return [[float(index) for index in range(len(row["options"]))] for row in rows]

    def encoding_info(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.encoding_error is not None:
            raise ValueError(self.encoding_error)
        return [
            {
                "tokens": self.token_count,
                "optionTokens": [3] * len(row["options"]),
            }
            for row in rows
        ]


def _model(engine: _FakeEngine) -> JuliaHFDecisionModel:
    model = JuliaHFDecisionModel.__new__(JuliaHFDecisionModel)
    model._engine = engine
    model.max_length = MAX_LENGTH
    model.head_length = HEAD_LENGTH
    return model


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


@pytest.mark.parametrize(
    ("count", "primitive", "ids", "labels"),
    [
        (2, Primitive.BINARY_CLASSIFICATION, ["false", "true"], ["No", "Yes"]),
        (4, Primitive.CANDIDATE_SELECTION, None, None),
        (8, Primitive.ORDINAL_SCORING, None, None),
    ],
)
def test_julia_preserves_candidate_order_and_native_softmax(
    count: int, primitive: Primitive, ids: list[str] | None, labels: list[str] | None
) -> None:
    example = _example(count, primitive, ids=ids, labels=labels)
    engine = _FakeEngine()
    model = _model(engine)

    response = model.predict_batch([example])[0]

    native_ids = (
        ["false", "true"]
        if primitive is Primitive.BINARY_CLASSIFICATION
        else [candidate.id for candidate in example.candidates]
    )
    native_logits = [float(index) for index in range(len(native_ids))]
    native_probabilities = _softmax(native_logits)
    expected = {
        candidate_id: value
        for candidate_id, value in zip(native_ids, native_probabilities, strict=True)
    }
    assert response.prediction.probabilities == pytest.approx(
        [expected[candidate.id] for candidate in example.candidates]
    )
    assert response.response["native_candidate_ids"] == native_ids
    assert response.response["native_logits"] == pytest.approx(native_logits)
    assert response.response["native_probabilities"] == pytest.approx(native_probabilities)
    assert response.response["native_type"] == julia_hf.NATIVE_TYPES[primitive]
    assert sum(response.prediction.probabilities) == pytest.approx(1.0)
    assert response.input_contract is not None
    assert response.input_contract["original_input_tokens"] == 123
    assert response.input_contract["max_input_tokens"] == MAX_LENGTH
    assert response.input_contract["truncated"] is False
    assert response.input_contract["policy_version"] == julia_hf.POLICY_VERSION
    assert response.input_contract["model_facing_example"] == example.model_dump(mode="json")


def test_julia_orders_binary_options_false_then_true() -> None:
    example = _example(
        2,
        Primitive.BINARY_CLASSIFICATION,
        ids=["true", "false"],
        labels=["Yes", "No"],
    )
    engine = _FakeEngine(logits=[[5.0, -5.0]])
    model = _model(engine)

    response = model.predict_batch([example])[0]

    assert response.response["native_candidate_ids"] == ["false", "true"]
    assert response.request["options"] == ["No", "Yes"]
    assert response.prediction.probabilities[0] == pytest.approx(0.0, abs=1e-4)
    assert response.prediction.probabilities[1] == pytest.approx(1.0, abs=1e-4)


def test_julia_renders_binary_candidate_labels_literally() -> None:
    example = _example(
        2,
        Primitive.BINARY_CLASSIFICATION,
        ids=["no", "yes"],
        labels=["No", "Yes"],
    )
    engine = _FakeEngine()
    model = _model(engine)

    response = model.predict_batch([example])[0]

    assert response.response["native_type"] == "noul"
    assert response.request["options"] == ["No", "Yes"]
    assert engine.rows[0]["type"] == "noul"


def test_julia_rejects_too_many_candidates() -> None:
    example = _example(MAX_CANDIDATES + 1, Primitive.CANDIDATE_SELECTION)
    model = _model(_FakeEngine())

    with pytest.raises(UnsupportedJuliaInput, match="outside"):
        model.validate_example(example)


def test_julia_reports_encoding_rejections_as_unsupported() -> None:
    example = _example(4, Primitive.CANDIDATE_SELECTION)
    model = _model(_FakeEngine(encoding_error="Option exceeds 48-token model contract"))

    with pytest.raises(UnsupportedJuliaInput, match="48-token"):
        model.validate_example(example)


def test_julia_rejects_mismatched_logit_count() -> None:
    example = _example(4, Primitive.CANDIDATE_SELECTION)
    model = _model(_FakeEngine(logits=[[1.0, 2.0, 3.0]]))

    with pytest.raises(RuntimeError, match="logit count"):
        model.predict_batch([example])


def test_julia_rejects_nonfinite_logits() -> None:
    example = _example(4, Primitive.CANDIDATE_SELECTION)
    model = _model(_FakeEngine(logits=[[1.0, float("nan"), 3.0, 4.0]]))

    with pytest.raises(RuntimeError, match="non-finite"):
        model.predict_batch([example])


def test_julia_rejects_incorrect_answer_count() -> None:
    examples = [
        _example(4, Primitive.CANDIDATE_SELECTION),
        _example(4, Primitive.CANDIDATE_SELECTION),
    ]
    model = _model(_FakeEngine(logits=[[1.0, 2.0, 3.0, 4.0]]))

    with pytest.raises(RuntimeError, match="answer count"):
        model.predict_batch(examples)


def _install_stub_runtime(monkeypatch: pytest.MonkeyPatch, recorded: dict[str, Any]) -> None:
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    julia = ModuleType("julia")

    def load_model(checkpoint: str, **kwargs: Any) -> Any:
        recorded["checkpoint"] = checkpoint
        recorded["kwargs"] = kwargs
        return SimpleNamespace()

    julia.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "julia", julia)


def test_julia_loader_pins_repository_revision_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weights = tmp_path / julia_hf.WEIGHTS_FILE
    weights.write_bytes(b"pinned")
    monkeypatch.setattr(julia_hf, "_sha256_file", lambda path: WEIGHTS_SHA256)
    recorded: dict[str, Any] = {}
    _install_stub_runtime(monkeypatch, recorded)

    model = JuliaHFDecisionModel(
        model_dir=tmp_path,
        model_repo=MODEL_ID,
        model_revision=MODEL_REVISION,
        expected_weights_sha256=WEIGHTS_SHA256,
    )

    assert recorded["checkpoint"] == str(tmp_path)
    assert recorded["kwargs"]["strict_encoding"] is True
    assert recorded["kwargs"]["max_length"] == MAX_LENGTH
    assert recorded["kwargs"]["head_length"] == HEAD_LENGTH
    assert recorded["kwargs"]["device"] == "cpu"
    assert model.device == "cpu"
    assert model.metadata["model_revision"] == MODEL_REVISION
    assert model.metadata["strict_encoding"] is True


def test_julia_loader_rejects_unpinned_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weights = tmp_path / julia_hf.WEIGHTS_FILE
    weights.write_bytes(b"pinned")
    recorded: dict[str, Any] = {}
    _install_stub_runtime(monkeypatch, recorded)

    with pytest.raises(ValueError, match="pinned model repository and revision"):
        JuliaHFDecisionModel(
            model_dir=tmp_path,
            model_repo="SupersonicLabs/Other",
            model_revision=MODEL_REVISION,
            expected_weights_sha256=WEIGHTS_SHA256,
        )
    with pytest.raises(ValueError, match="published checkpoint SHA-256"):
        JuliaHFDecisionModel(
            model_dir=tmp_path,
            model_repo=MODEL_ID,
            model_revision=MODEL_REVISION,
            expected_weights_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="does not match the pinned release"):
        JuliaHFDecisionModel(
            model_dir=tmp_path,
            model_repo=MODEL_ID,
            model_revision=MODEL_REVISION,
            expected_weights_sha256=WEIGHTS_SHA256,
        )
    assert "checkpoint" not in recorded
