"""The GLiNER adapter preserves candidate identity and rejects invalid rows."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from decision_bench.models.gliner25 import GLiNER25DecideModel
from decision_bench.schemas import Candidate, DecisionExample, Primitive


def _example(count: int, primitive: Primitive) -> DecisionExample:
    candidates = [
        Candidate(
            id=f"candidate-{index}",
            label=f"Option {index}",
            ordinal_value=float(index) if primitive is Primitive.ORDINAL_SCORING else None,
        )
        for index in range(count)
    ]
    return DecisionExample(
        row_id=f"gliner-{count}",
        task_name="routing",
        primitive=primitive,
        family="routing_triage",
        domain="financial",
        instruction="Choose the option named in the state.",
        state={"best_option": "Option 0"},
        candidates=candidates,
        gold_candidate_id=candidates[0].id,
        gold_probabilities=[1.0] + [0.0] * (count - 1),
    )


class _CompiledSchema:
    def build(self) -> dict[str, str]:
        return {"schema": "stub"}


class _FakeClassifier:
    def __init__(self, *, token_count: int, missing_label: bool = False) -> None:
        self.token_count = token_count
        self.missing_label = missing_label
        self.labels: list[str] = []
        self.scorer = SimpleNamespace(processor=self)

    def compile_schema(self, schema: Any) -> _CompiledSchema:
        return _CompiledSchema()

    def collate_fn_inference(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, self.token_count)))

    def score(self, text: str, schema: Any, *, config: Any) -> Any:
        labels = self.labels
        if self.missing_label:
            labels = labels[:-1]
        probabilities = {
            label: float(index + 1) / (len(self.labels) * (len(self.labels) + 1) / 2)
            for index, label in enumerate(self.labels)
        }
        logits = {label: float(index) for index, label in enumerate(self.labels)}
        return SimpleNamespace(
            tasks={"decision": {label: logits[label] for label in reversed(labels)}},
            logit=lambda task, label: logits[label],
            probability=lambda task, label: probabilities[label],
        )


def _model(
    monkeypatch: pytest.MonkeyPatch, example: DecisionExample, **kwargs: Any
) -> GLiNER25DecideModel:
    package = ModuleType("gliner2")
    classification = ModuleType("gliner2.classification")
    classification.ClassificationConfig = lambda **options: options  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", package)
    monkeypatch.setitem(sys.modules, "gliner2.classification", classification)

    model = GLiNER25DecideModel.__new__(GLiNER25DecideModel)
    model._classifier = _FakeClassifier(**kwargs)
    model._classifier.labels = [candidate.label for candidate in example.candidates]
    model._build_schema = lambda row: object()  # type: ignore[assignment]
    return model


@pytest.mark.parametrize(
    ("count", "primitive"),
    [
        (2, Primitive.BINARY_CLASSIFICATION),
        (4, Primitive.CANDIDATE_SELECTION),
        (8, Primitive.ORDINAL_SCORING),
    ],
)
def test_gliner25_preserves_candidate_order_and_raw_scores(
    monkeypatch: pytest.MonkeyPatch, count: int, primitive: Primitive
) -> None:
    example = _example(count, primitive)
    model = _model(monkeypatch, example, token_count=100)

    response = model.predict_batch([example])[0]

    assert response.prediction.probabilities == pytest.approx(
        [(index + 1) / sum(range(1, count + 1)) for index in range(count)]
    )
    assert response.response["native_candidate_ids"] == [
        candidate.id for candidate in example.candidates
    ]
    assert len(response.response["native_logits"]) == count
    assert response.input_contract is not None
    assert response.input_contract["original_input_tokens"] == 100
    assert response.input_contract["model_facing_example"] == example.model_dump(mode="json")


def test_gliner25_rejects_overlong_input(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _example(2, Primitive.BINARY_CLASSIFICATION)
    model = _model(monkeypatch, example, token_count=513)

    with pytest.raises(ValueError, match="over the 512-token encoder limit"):
        model.validate_example(example)


def test_gliner25_rejects_duplicate_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _example(2, Primitive.BINARY_CLASSIFICATION)
    example.candidates[1].label = example.candidates[0].label
    model = _model(monkeypatch, example, token_count=100)

    with pytest.raises(ValueError, match="duplicate candidate labels"):
        model.validate_example(example)


def test_gliner25_rejects_missing_candidate_score(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _example(4, Primitive.CANDIDATE_SELECTION)
    model = _model(monkeypatch, example, token_count=100, missing_label=True)

    with pytest.raises(RuntimeError, match="incomplete candidate scores"):
        model.predict_batch([example])
