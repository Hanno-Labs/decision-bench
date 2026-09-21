from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from decision_bench.models.openrouter import reasoning_effort_for_example
from decision_bench.prompt import (
    build_openrouter_request,
    fit_example_to_token_budget,
    render_user_prompt,
)
from decision_bench.schemas import Candidate, DecisionExample, DecisionPrediction, Primitive
from decision_bench.scoring import negative_log_likelihood, score_prediction


def example(primitive: Primitive = Primitive.CANDIDATE_SELECTION) -> DecisionExample:
    return DecisionExample(
        row_id="row-1",
        task_name="routing",
        primitive=primitive,
        family="routing_triage",
        domain="financial",
        instruction="Route the request to the best team.",
        state={"request": "My card was retained by an ATM."},
        candidates=[
            Candidate(id="cash", label="Cash withdrawal support"),
            Candidate(id="card", label="Card support"),
        ],
        gold_candidate_id="card",
        gold_probabilities=[0.0, 1.0],
    )


def test_prompt_uses_ordered_indices_without_gold_label() -> None:
    row = example()
    payload = json.loads(render_user_prompt(row))
    assert [candidate["index"] for candidate in payload["candidates"]] == [0, 1]
    assert "gold_candidate_id" not in payload


def test_response_schema_has_exact_candidate_count() -> None:
    request = build_openrouter_request(example())
    schema = request["response_format"]["json_schema"]["schema"]
    probabilities = schema["properties"]["probabilities"]
    assert probabilities["minItems"] == 2
    assert probabilities["maxItems"] == 2
    assert request["provider"] == {"require_parameters": True}
    assert "temperature" not in request


def test_reasoning_effort_override_only_applies_to_reasoning_family() -> None:
    ordinary = example()
    reasoning = ordinary.model_copy(update={"family": "reasoning"})

    assert (
        reasoning_effort_for_example(
            ordinary,
            default_reasoning_effort="none",
            reasoning_family_effort="max",
        )
        == "none"
    )
    assert (
        reasoning_effort_for_example(
            reasoning,
            default_reasoning_effort="none",
            reasoning_family_effort="max",
        )
        == "max"
    )


def test_text_truncation_preserves_frozen_decision_contract() -> None:
    row = example().model_copy(
        update={
            "instruction": "instruction " * 40,
            "state": {"request": "state " * 80},
            "candidates": [
                candidate.model_copy(update={"description": candidate.label * 40})
                for candidate in example().candidates
            ],
        }
    )
    original = row.model_dump(mode="json")
    fitted, report = fit_example_to_token_budget(
        row,
        max_input_tokens=500,
        count_tokens=lambda value: len(render_user_prompt(value)),
    )
    repeated = fit_example_to_token_budget(
        row,
        max_input_tokens=500,
        count_tokens=lambda value: len(render_user_prompt(value)),
    )

    assert repeated == (fitted, report)
    assert row.model_dump(mode="json") == original
    assert len(render_user_prompt(fitted)) <= 500
    assert [candidate.id for candidate in fitted.candidates] == ["cash", "card"]
    assert fitted.gold_candidate_id == row.gold_candidate_id
    assert fitted.gold_probabilities == row.gold_probabilities
    assert report.truncated is True
    assert report.final_input_tokens <= report.max_input_tokens


def test_score_uses_candidate_order() -> None:
    scored = score_prediction(example(), DecisionPrediction(probabilities=[0.1, 0.9]))
    assert scored.selected_candidate_id == "card"
    assert scored.correct is True


def test_probability_distribution_is_validated() -> None:
    with pytest.raises(ValidationError):
        DecisionPrediction(probabilities=[0.0, 0.0])


def test_probability_distribution_is_normalized_after_generation() -> None:
    prediction = DecisionPrediction(probabilities=[0.2, 0.2])
    assert prediction.probabilities == [0.5, 0.5]


def test_soft_gold_distribution_is_preserved_in_log_loss() -> None:
    row = example().model_copy(
        update={"gold_probabilities": [0.25, 0.75]},
    )
    prediction = DecisionPrediction(probabilities=[0.25, 0.75])
    assert negative_log_likelihood(row, prediction) == pytest.approx(0.5623351446)


def test_ordinal_expected_score() -> None:
    row = DecisionExample(
        row_id="row-2",
        task_name="priority",
        primitive=Primitive.ORDINAL_SCORING,
        family="rubric_scoring_prioritization",
        domain="support",
        instruction="Score priority.",
        state="Production is unavailable.",
        candidates=[
            Candidate(id="high", label="High", ordinal_value=3),
            Candidate(id="low", label="Low", ordinal_value=1),
            Candidate(id="medium", label="Medium", ordinal_value=2),
        ],
        gold_candidate_id="high",
        gold_probabilities=[1.0, 0.0, 0.0],
    )
    scored = score_prediction(row, DecisionPrediction(probabilities=[0.75, 0.0, 0.25]))
    assert scored.expected_ordinal_score == pytest.approx(2.75)
