"""The GLiNER adapter preserves candidate identity and rejects invalid rows."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from decision_bench.models.gliner25 import (
    CHECKPOINTS,
    GLiNER25ClassificationModel,
)
from decision_bench.schemas import Candidate, DecisionExample, Primitive

SMALL_V1 = "fastino/gliner2.5-small-v1"
SMALL_V1_REVISION = "3ec6d3dd7e1e93a7cf9b46096fa47aeda61c711c"
DECIDE = "fastino/GLiNER2.5-Decide"
DECIDE_REVISION = "0872ab149bd2f8a50ed5fc7ad8cfc3293e9a3bad"


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
) -> GLiNER25ClassificationModel:
    package = ModuleType("gliner2")
    classification = ModuleType("gliner2.classification")
    classification.ClassificationConfig = lambda **options: options  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", package)
    monkeypatch.setitem(sys.modules, "gliner2.classification", classification)

    model = GLiNER25ClassificationModel.__new__(GLiNER25ClassificationModel)
    model._classifier = _FakeClassifier(**kwargs)
    model._classifier.labels = [candidate.label for candidate in example.candidates]
    model._build_schema = lambda row: object()  # type: ignore[assignment]
    model.checkpoint = CHECKPOINTS[1]
    model.model_repo = SMALL_V1
    model.model_revision = SMALL_V1_REVISION
    model.device = "cpu"
    return model


def _install_loader(monkeypatch: pytest.MonkeyPatch, *, architecture: str) -> None:
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    package = ModuleType("gliner2")
    classification = ModuleType("gliner2.classification")

    class _InnerModel:
        pass

    _InnerModel.architecture = architecture  # type: ignore[attr-defined]

    class _Classifier:
        def __init__(self, path: str) -> None:
            self.path = path
            self.model = _InnerModel()

        @classmethod
        def from_pretrained(cls, path: str) -> _Classifier:
            return cls(path)

        def to(self, device: str | None = None) -> _Classifier:
            return self

        def eval(self) -> _Classifier:
            return self

    classification.Classifier = _Classifier  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", package)
    monkeypatch.setitem(sys.modules, "gliner2.classification", classification)


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
    assert response.input_contract["policy_version"] == "gliner25-boundary-exact-encoder-v1"
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


def test_gliner25_registry_covers_both_architectures() -> None:
    by_key = {(entry.model_id, entry.revision): entry for entry in CHECKPOINTS}

    assert by_key[(DECIDE, DECIDE_REVISION)].architecture == "span"
    assert by_key[(SMALL_V1, SMALL_V1_REVISION)].architecture == "boundary"
    assert by_key[(SMALL_V1, SMALL_V1_REVISION)].tokenizer_revision == SMALL_V1_REVISION


def test_gliner25_loader_accepts_boundary_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_loader(monkeypatch, architecture="boundary")

    model = GLiNER25ClassificationModel(
        model_dir=Path("/tmp/model"),
        model_repo=SMALL_V1,
        model_revision=SMALL_V1_REVISION,
    )

    assert model.device == "cpu"
    assert model.metadata["architecture"] == "boundary"
    assert model.metadata["model_type"] == "gliner25_boundary_classifier"
    assert model.metadata["tokenizer_revision"] == SMALL_V1_REVISION


def test_gliner25_loader_rejects_unpinned_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_loader(monkeypatch, architecture="boundary")

    with pytest.raises(ValueError, match=r"unsupported GLiNER2\.5 checkpoint"):
        GLiNER25ClassificationModel(
            model_dir=Path("/tmp/model"),
            model_repo=SMALL_V1,
            model_revision="0" * 40,
        )


def test_gliner25_loader_rejects_architecture_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_loader(monkeypatch, architecture="span")

    with pytest.raises(ValueError, match="pins architecture 'boundary'"):
        GLiNER25ClassificationModel(
            model_dir=Path("/tmp/model"),
            model_repo=SMALL_V1,
            model_revision=SMALL_V1_REVISION,
        )
