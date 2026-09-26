"""The lev adapter preserves candidate identity and lev's native option distribution."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from decision_bench.models.lev_hf import (
    LEV_MODEL_REPO,
    LEV_MODEL_REVISION,
    LevHFDecisionModel,
    UnsupportedLevInput,
)
from decision_bench.schemas import Candidate, DecisionExample, Primitive


class _FakeEngine:
    """Minimal stand-in for ``lev.DecisionEngine``."""

    def __init__(
        self,
        *,
        choice: dict[str, float] | None = None,
        score: dict[int, float] | None = None,
        noul_probability: float = 0.75,
        rating: dict[int, float] | None = None,
        drop_key: bool = False,
        no_answer: bool = False,
    ) -> None:
        self.choice = choice or {}
        self.score = score or {}
        self.noul_probability = noul_probability
        self.rating = rating if rating is not None else {0: 0.25, 8: 0.75}
        self.drop_key = drop_key
        self.no_answer = no_answer
        self.config = SimpleNamespace(
            prompt_style="chat",
            noul_readout="rating",
            order_average=True,
            max_label_options=None,
        )
        self.mode_b_head = object()
        self.requests: list[tuple[Any, dict[str, Any]]] = []

    def _probability_map(self, probabilities: Any) -> Any:
        if not self.drop_key:
            return probabilities
        trimmed = dict(probabilities)
        trimmed.pop(sorted(trimmed)[0])
        return trimmed

    def _answer(self, question: dict[str, Any]) -> SimpleNamespace:
        if question["type"] == "noul":
            return SimpleNamespace(
                type="noul",
                noul=self.noul_probability,
                confidence=0.5,
                probabilities=self._probability_map(self.rating),
            )
        if question["type"] == "choice":
            probabilities = self._probability_map(self.choice)
            return SimpleNamespace(
                type="choice", choice=None, confidence=0.5, probabilities=probabilities
            )
        probabilities = self._probability_map(self.score)
        return SimpleNamespace(
            type="score", score=None, legend={}, confidence=0.5, probabilities=probabilities
        )

    def system_one(self, state: Any, questions: dict[str, Any]) -> SimpleNamespace:
        self.requests.append((state, questions))
        answers = {} if self.no_answer else {"decision": self._answer(questions["decision"])}
        return SimpleNamespace(
            model="lev",
            answers=answers,
            usage=SimpleNamespace(input_tokens=12, output_tokens=0, cached_input_tokens=0),
            status=None,
            model_dump=lambda mode="json": {"model": "lev", "answers": {}},
        )


def _install_lev(monkeypatch: pytest.MonkeyPatch, engine: _FakeEngine) -> None:
    module = ModuleType("lev")
    module.load = lambda *args, **kwargs: engine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lev", module)


def _model_dir(tmp_path: Path) -> Path:
    (tmp_path / "lev_release.json").write_text(
        json.dumps({"name": "lev", "step": 18750, "base_model": "Qwen/Qwen3.5-4B"})
    )
    return tmp_path


def _model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine: _FakeEngine
) -> LevHFDecisionModel:
    _install_lev(monkeypatch, engine)
    return LevHFDecisionModel(model_dir=_model_dir(tmp_path))


def _example(
    candidates: list[Candidate], primitive: Primitive, *, state: Any = None
) -> DecisionExample:
    return DecisionExample(
        row_id="lev-row",
        task_name="inbox_triage",
        primitive=primitive,
        family="routing_triage",
        domain="financial",
        instruction="Select the candidate described by the state.",
        state={"ticket": "needs an owner"} if state is None else state,
        candidates=candidates,
        gold_candidate_id=candidates[0].id,
        gold_probabilities=[1.0] + [0.0] * (len(candidates) - 1),
    )


def _binary_by_id() -> list[Candidate]:
    return [
        Candidate(id="label-0", label="Reject", description="Keep the request queued."),
        Candidate(id="label-1", label="Accept", description="Send the request downstream."),
    ]


def _binary_by_label() -> list[Candidate]:
    return [
        Candidate(id="candidate-0", label="No", description="The claim is unsupported."),
        Candidate(id="candidate-1", label="Yes", description="The claim is supported."),
    ]


def _choice() -> list[Candidate]:
    return [
        Candidate(id="candidate-0", label="Alpha", description="Routes to billing."),
        Candidate(id="candidate-1", label="Beta", description="Routes to security."),
        Candidate(id="candidate-2", label="Gamma", description="Routes to support."),
        Candidate(id="candidate-3", label="Delta", description="Routes to platform."),
    ]


def _score(count: int = 5) -> list[Candidate]:
    return [
        Candidate(
            id=f"candidate-{index}",
            label=f"Level {index + 1}",
            description=f"Escalation level {index + 1}.",
            ordinal_value=float(index + 1),
        )
        for index in range(count)
    ]


@pytest.mark.parametrize(
    (
        "primitive",
        "candidates",
        "candidate_ids",
        "native_ids",
        "native",
        "aligned",
        "question_type",
    ),
    [
        (
            Primitive.BINARY_CLASSIFICATION,
            _binary_by_id(),
            ["label-0", "label-1"],
            ["label-1", "label-0"],
            [0.75, 0.25],
            [0.25, 0.75],
            "noul",
        ),
        (
            Primitive.BINARY_CLASSIFICATION,
            _binary_by_label(),
            ["candidate-0", "candidate-1"],
            ["candidate-1", "candidate-0"],
            [0.75, 0.25],
            [0.25, 0.75],
            "noul",
        ),
        (
            Primitive.CANDIDATE_SELECTION,
            _choice(),
            ["candidate-0", "candidate-1", "candidate-2", "candidate-3"],
            ["candidate-0", "candidate-1", "candidate-2", "candidate-3"],
            [0.4, 0.3, 0.1, 0.2],
            [0.4, 0.3, 0.1, 0.2],
            "choice",
        ),
        (
            Primitive.ORDINAL_SCORING,
            _score(),
            ["candidate-0", "candidate-1", "candidate-2", "candidate-3", "candidate-4"],
            ["candidate-0", "candidate-1", "candidate-2", "candidate-3", "candidate-4"],
            [0.1, 0.2, 0.3, 0.3, 0.1],
            [0.1, 0.2, 0.3, 0.3, 0.1],
            "score",
        ),
    ],
)
def test_lev_preserves_candidate_order_and_native_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    primitive: Primitive,
    candidates: list[Candidate],
    candidate_ids: list[str],
    native_ids: list[str],
    native: list[float],
    aligned: list[float],
    question_type: str,
) -> None:
    engine = _FakeEngine(
        # Deliberately unordered so the adapter has to restore candidate order.
        choice={"Gamma": 0.1, "Alpha": 0.4, "Delta": 0.2, "Beta": 0.3},
        score={4: 0.1, 0: 0.1, 1: 0.2, 2: 0.3, 3: 0.3},
    )
    model = _model(monkeypatch, tmp_path, engine)
    example = _example(candidates, primitive)

    response = model.predict_batch([example])[0]

    assert response.response["native_candidate_ids"] == native_ids
    assert response.response["native_probabilities"] == pytest.approx(native)
    assert response.prediction.probabilities == pytest.approx(aligned)
    assert sum(response.prediction.probabilities) == pytest.approx(1.0)
    assert response.input_contract is not None
    assert response.input_contract["candidate_ids"] == candidate_ids
    assert response.input_contract["primitive"] == primitive.value
    assert response.input_contract["question_type"] == question_type
    assert response.input_contract["truncated"] is False
    assert response.input_contract["model_facing_example"] == example.model_dump(mode="json")
    assert response.response["raw_response"] == {"model": "lev", "answers": {}}
    question = response.request["questions"]["decision"]
    assert question["type"] == question_type
    assert response.request["state"] == example.state
    assert engine.requests[0][0] == example.state


def test_lev_renders_the_native_question_per_primitive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine(
        choice={"Alpha": 0.25, "Beta": 0.25, "Gamma": 0.25, "Delta": 0.25},
        score={0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
    )
    model = _model(monkeypatch, tmp_path, engine)

    binary = _example(_binary_by_id(), Primitive.BINARY_CLASSIFICATION)
    choice = _example(_choice(), Primitive.CANDIDATE_SELECTION)
    score = _example(_score(), Primitive.ORDINAL_SCORING)
    model.predict_batch([binary, choice, score])

    _, binary_questions = engine.requests[0]
    _, choice_questions = engine.requests[1]
    _, score_questions = engine.requests[2]
    binary_question = binary_questions["decision"]
    assert set(binary_question["criteria"]) == {"true", "false"}
    assert binary_question["criteria"]["true"] == "Accept: Send the request downstream."
    assert binary_question["criteria"]["false"] == "Reject: Keep the request queued."
    assert binary_question["instructions"] == binary.instruction

    choice_question = choice_questions["decision"]
    assert list(choice_question["criteria"]) == ["Alpha", "Beta", "Gamma", "Delta"]
    assert choice_question["criteria"]["Gamma"] == "Routes to support."

    score_question = score_questions["decision"]
    assert score_question["criteria"] == [
        "Level 1: Escalation level 1.",
        "Level 2: Escalation level 2.",
        "Level 3: Escalation level 3.",
        "Level 4: Escalation level 4.",
        "Level 5: Escalation level 5.",
    ]


def test_lev_records_the_rating_distribution_for_binary_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rating = {level: (level + 1) / 45 for level in range(9)}
    engine = _FakeEngine(noul_probability=0.75, rating=rating)
    model = _model(monkeypatch, tmp_path, engine)
    example = _example(_binary_by_id(), Primitive.BINARY_CLASSIFICATION)

    response = model.predict_batch([example])[0]

    assert response.response["native_rating_probabilities"] == {
        str(level): pytest.approx(value) for level, value in rating.items()
    }
    assert response.prediction.probabilities == pytest.approx([0.25, 0.75])


def test_lev_records_the_native_score_levels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine(score={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.3, 4: 0.1})
    model = _model(monkeypatch, tmp_path, engine)
    example = _example(_score(), Primitive.ORDINAL_SCORING)

    response = model.predict_batch([example])[0]

    assert response.response["native_score_levels"] == {
        "0": pytest.approx(0.1),
        "1": pytest.approx(0.2),
        "2": pytest.approx(0.3),
        "3": pytest.approx(0.3),
        "4": pytest.approx(0.1),
    }


def test_lev_prompt_characters_track_the_rendered_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine(choice={"Alpha": 0.25, "Beta": 0.25, "Gamma": 0.25, "Delta": 0.25})
    model = _model(monkeypatch, tmp_path, engine)
    short = _example(_choice(), Primitive.CANDIDATE_SELECTION, state={"ticket": "tiny"})
    long = _example(_choice(), Primitive.CANDIDATE_SELECTION, state={"ticket": "x" * 400})

    assert model.prompt_characters(short) == model.prompt_characters(short)
    assert model.prompt_characters(long) > model.prompt_characters(short)


def test_lev_rejects_score_rows_outside_the_published_levels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine(score={level: 1 / 11 for level in range(11)})
    model = _model(monkeypatch, tmp_path, engine)
    example = _example(_score(11), Primitive.ORDINAL_SCORING)

    with pytest.raises(UnsupportedLevInput, match="requires 2-10 levels"):
        model.validate_example(example)


def test_lev_rejects_binary_rows_without_true_false_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine()
    model = _model(monkeypatch, tmp_path, engine)
    ambiguous = [
        Candidate(id="candidate-0", label="First", description=None),
        Candidate(id="candidate-1", label="Second", description=None),
    ]
    example = _example(ambiguous, Primitive.BINARY_CLASSIFICATION)

    with pytest.raises(UnsupportedLevInput, match="unambiguous false/true"):
        model.validate_example(example)


def test_lev_rejects_choice_rows_with_duplicate_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine()
    model = _model(monkeypatch, tmp_path, engine)
    candidates = _choice()
    candidates[1] = Candidate(id="candidate-1", label="Alpha", description="Routes to legal.")
    example = _example(candidates, Primitive.CANDIDATE_SELECTION)

    with pytest.raises(UnsupportedLevInput, match="unique candidate labels"):
        model.validate_example(example)


@pytest.mark.parametrize(
    ("engine", "primitive", "candidates", "message"),
    [
        (
            _FakeEngine(
                choice={"Alpha": 0.5, "Beta": 0.3, "Gamma": 0.1, "Delta": 0.1}, drop_key=True
            ),
            Primitive.CANDIDATE_SELECTION,
            _choice(),
            "incomplete candidate probabilities",
        ),
        (
            _FakeEngine(score={0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}, drop_key=True),
            Primitive.ORDINAL_SCORING,
            _score(),
            "incomplete score-level probabilities",
        ),
        (
            _FakeEngine(no_answer=True),
            Primitive.BINARY_CLASSIFICATION,
            _binary_by_id(),
            "no answer",
        ),
    ],
)
def test_lev_rejects_incomplete_native_answers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    engine: _FakeEngine,
    primitive: Primitive,
    candidates: list[Candidate],
    message: str,
) -> None:
    model = _model(monkeypatch, tmp_path, engine)
    example = _example(candidates, primitive)

    with pytest.raises(RuntimeError, match=message):
        model.predict_batch([example])


def test_lev_rejects_noul_probability_outside_the_unit_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine(noul_probability=1.5)
    model = _model(monkeypatch, tmp_path, engine)
    example = _example(_binary_by_id(), Primitive.BINARY_CLASSIFICATION)

    with pytest.raises(RuntimeError, match="outside \\[0, 1\\]"):
        model.predict_batch([example])


def test_lev_records_the_pinned_release_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine()
    model = _model(monkeypatch, tmp_path, engine)

    metadata = model.metadata

    assert metadata["model"] == LEV_MODEL_REPO
    assert metadata["model_revision"] == LEV_MODEL_REVISION
    assert metadata["code_revision"] == "cf104b69329302e4eac674a730c71f3511047db8"
    assert metadata["base_model"] == "Qwen/Qwen3.5-4B"
    assert metadata["release_step"] == 18750
    assert metadata["probability_source"] == "lev_calibrated_option_distribution"
    assert metadata["input_truncation_policy"] == "no_published_limit_and_no_truncation"
    assert metadata["adapter_sha256"] is None
    assert metadata["adapter_config_sha256"] is None
    assert metadata["candidate_boundary"]["score_levels"] == {"min": 2, "max": 10}
    assert metadata["candidate_boundary"]["noul_rating_levels"] == 9
    assert metadata["order_averaging"] is True
    assert metadata["noul_readout"] == "rating"


def test_lev_requires_the_pinned_model_and_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_lev(monkeypatch, _FakeEngine())
    model_dir = _model_dir(tmp_path)

    with pytest.raises(ValueError, match="pinned model and revision"):
        LevHFDecisionModel(model_dir=model_dir, model_revision="0" * 40)


def test_lev_requires_the_pinned_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "lev", None)

    with pytest.raises(RuntimeError, match="decision-bench\\[lev\\]"):
        LevHFDecisionModel(model_dir=_model_dir(tmp_path))


def test_lev_requires_a_release_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_lev(monkeypatch, _FakeEngine())

    with pytest.raises(ValueError, match=r"missing lev_release\.json"):
        LevHFDecisionModel(model_dir=tmp_path)


def test_lev_requires_the_candidate_path_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine()
    engine.mode_b_head = None
    _install_lev(monkeypatch, engine)

    with pytest.raises(ValueError, match=r"mode_b_head\.pt"):
        LevHFDecisionModel(model_dir=_model_dir(tmp_path))


def test_lev_verifies_pinned_asset_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_lev(monkeypatch, _FakeEngine())
    model_dir = _model_dir(tmp_path)
    (model_dir / "adapter_model.safetensors").write_bytes(b"weights")

    digest = hashlib.sha256(b"weights").hexdigest()
    model = LevHFDecisionModel(model_dir=model_dir, expected_weights_sha256=digest)
    assert model.metadata["adapter_sha256"] == digest

    with pytest.raises(ValueError, match="pinned SHA-256"):
        LevHFDecisionModel(model_dir=model_dir, expected_weights_sha256="0" * 64)
    with pytest.raises(ValueError, match="pinned SHA-256"):
        LevHFDecisionModel(model_dir=model_dir, expected_mode_b_head_sha256=digest)
