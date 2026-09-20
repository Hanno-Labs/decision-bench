from __future__ import annotations

import pytest

from decision_bench.models.jev_openrouter import (
    _prediction_from_answer,
    build_jev_request,
    jev_effective_input_token_budget,
)
from decision_bench.schemas import Candidate, DecisionExample, Primitive


def _example(primitive: Primitive, candidates: list[Candidate]) -> DecisionExample:
    return DecisionExample(
        row_id="row",
        task_name="task",
        primitive=primitive,
        family="family",
        domain="domain",
        instruction="Decide.",
        state={"text": "input"},
        candidates=candidates,
        gold_candidate_id=candidates[0].id,
        gold_probabilities=[1.0, *([0.0] * (len(candidates) - 1))],
    )


def test_noul_maps_probability_to_arbitrary_binary_labels() -> None:
    example = _example(
        Primitive.BINARY_CLASSIFICATION,
        [
            Candidate(id="label-0", label="Reject"),
            Candidate(id="label-1", label="Allow"),
        ],
    )
    request, response_order = build_jev_request(example)
    prediction = _prediction_from_answer(
        example,
        {"type": "noul", "noul": 0.8},
        response_order,
    )
    assert request["questions"]["decision"]["type"] == "noul"
    assert request["questions"]["decision"]["criteria"] == {
        "true": "Reject",
        "false": "Allow",
    }
    assert prediction.probabilities == pytest.approx([0.8, 0.2])


def test_choice_preserves_candidate_ids_and_order() -> None:
    example = _example(
        Primitive.CANDIDATE_SELECTION,
        [Candidate(id="b", label="Beta"), Candidate(id="a", label="Alpha")],
    )
    request, response_order = build_jev_request(example)
    prediction = _prediction_from_answer(
        example,
        {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}},
        response_order,
    )
    assert list(request["questions"]["decision"]["criteria"]) == ["b", "a"]
    assert prediction.probabilities == [0.3, 0.7]


def test_score_orders_levels_then_maps_probabilities_back() -> None:
    example = _example(
        Primitive.ORDINAL_SCORING,
        [
            Candidate(id="high", label="High", ordinal_value=2.0),
            Candidate(id="low", label="Low", ordinal_value=0.0),
            Candidate(id="mid", label="Mid", ordinal_value=1.0),
        ],
    )
    request, response_order = build_jev_request(example)
    prediction = _prediction_from_answer(
        example,
        {"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
        response_order,
    )
    assert request["questions"]["decision"]["criteria"] == ["Low", "Mid", "High"]
    assert prediction.probabilities == [0.7, 0.1, 0.2]


def test_jev_effective_input_budget_reserves_private_serialization_headroom() -> None:
    assert jev_effective_input_token_budget(32_000, 2_048) == 29_952
    with pytest.raises(ValueError):
        jev_effective_input_token_budget(32_000, 32_000)
