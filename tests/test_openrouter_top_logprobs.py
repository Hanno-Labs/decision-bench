from __future__ import annotations

import math

import pytest

from decision_bench.models.openrouter_top_logprobs import (
    IncompleteCandidateLogprobsError,
    prediction_from_top_logprobs,
)
from decision_bench.prompt import build_openrouter_top_logprobs_request
from decision_bench.schemas import Candidate, DecisionExample, Primitive


def _example(candidate_count: int) -> DecisionExample:
    return DecisionExample(
        row_id="row",
        task_name="task",
        primitive=Primitive.CANDIDATE_SELECTION,
        family="family",
        domain="domain",
        instruction="Choose.",
        state={"text": "input"},
        candidates=[Candidate(id=str(index), label=str(index)) for index in range(candidate_count)],
        gold_candidate_id="0",
        gold_probabilities=[1.0, *([0.0] * (candidate_count - 1))],
    )


def _response(alternatives: list[tuple[str, float]]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {"content": '{"choice":"A"}'},
                "logprobs": {
                    "content": [
                        {
                            "token": '"A"',
                            "logprob": alternatives[0][1],
                            "top_logprobs": [
                                {"token": token, "logprob": logprob}
                                for token, logprob in alternatives
                            ],
                        }
                    ]
                },
            }
        ]
    }


def test_request_asks_for_one_token_and_twenty_logprobs() -> None:
    request = build_openrouter_top_logprobs_request(_example(4), model="test/model")
    assert request["max_tokens"] == 16
    assert request["logprobs"] is True
    assert request["top_logprobs"] == 20
    assert request["temperature"] == 0
    choice_schema = request["response_format"]["json_schema"]["schema"]["properties"]["choice"]
    assert choice_schema["enum"] == ["A", "B", "C", "D"]


def test_prediction_conditions_native_logprobs_on_candidate_symbols() -> None:
    response = _response(
        [
            ('"A"', math.log(0.4)),
            (" A", math.log(0.1)),
            ("B", math.log(0.3)),
            ("C", math.log(0.2)),
        ]
    )
    prediction = prediction_from_top_logprobs(response, 3)
    assert prediction.probabilities == pytest.approx([0.5, 0.3, 0.2])


def test_prediction_rejects_incomplete_candidate_distribution() -> None:
    response = _response([("A", math.log(0.7)), ("B", math.log(0.3))])
    with pytest.raises(IncompleteCandidateLogprobsError, match="C"):
        prediction_from_top_logprobs(response, 3)
