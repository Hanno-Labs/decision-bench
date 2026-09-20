from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from decision_bench.gguf_server import (
    DecisionQuestionRequest,
    GgufDecisionEngine,
    SystemOneRequest,
    _candidates,
    _normalize_probabilities,
    _project_answer,
)


class FakeLlama:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        probabilities = ",".join(str(value) for value in self.probabilities)
        return {
            "choices": [{"message": {"content": '{"probabilities": [' + probabilities + "]}"}}]
        }


def _engine(fake: FakeLlama) -> GgufDecisionEngine:
    engine = object.__new__(GgufDecisionEngine)
    engine.model_path = Path("model.gguf")
    engine.seed = 7
    engine._llama = fake
    return engine


def test_choice_projects_generic_distribution() -> None:
    question = DecisionQuestionRequest(
        type="choice",
        instructions="Choose a team.",
        criteria={"billing": "Payment issue", "technical": "Product issue"},
    )
    answer = _project_answer(question, _candidates(question), [0.2, 0.8])
    assert answer["choice"] == "technical"
    assert answer["probabilities"] == {"billing": 0.2, "technical": 0.8}


def test_noul_projects_true_probability() -> None:
    question = DecisionQuestionRequest(type="noul", instructions="Escalate?")
    answer = _project_answer(question, _candidates(question), [0.75, 0.25])
    assert answer == {"type": "noul", "noul": 0.75}


def test_score_projects_expected_level_and_legend() -> None:
    question = DecisionQuestionRequest(
        type="score",
        instructions="How severe?",
        criteria=["routine", "important", "critical"],
    )
    answer = _project_answer(question, _candidates(question), [0.1, 0.2, 0.7])
    assert answer["score"] == pytest.approx(1.6)
    assert answer["legend"] == {"0": "routine", "1": "important", "2": "critical"}


def test_engine_asks_generic_gguf_for_exact_candidate_count() -> None:
    fake = FakeLlama([0.25, 0.75])
    engine = _engine(fake)
    question = DecisionQuestionRequest(
        type="choice",
        instructions="Choose.",
        criteria={"a": "A", "b": "B"},
    )
    answer = engine.predict({"text": "state"}, question)
    assert answer["choice"] == "b"
    assert len(fake.calls) == 1
    schema = fake.calls[0]["response_format"]["schema"]
    assert schema["properties"]["probabilities"]["minItems"] == 2
    assert fake.calls[0]["seed"] == 7


def test_system_one_request_accepts_multiple_questions() -> None:
    request = SystemOneRequest(
        state="state",
        questions={
            "route": {
                "type": "choice",
                "instructions": "Route.",
                "criteria": {"a": "A", "b": "B"},
            },
            "urgent": {"type": "noul", "instructions": "Urgent?"},
        },
    )
    assert list(request.questions) == ["route", "urgent"]


def test_probability_validation_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        _normalize_probabilities([1.0], 2)
