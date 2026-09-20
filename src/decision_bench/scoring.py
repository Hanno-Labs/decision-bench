"""Pure scoring for distributions aligned to benchmark candidates."""

from __future__ import annotations

import math

from decision_bench.schemas import DecisionExample, DecisionPrediction, ScoredPrediction


def score_prediction(
    example: DecisionExample,
    prediction: DecisionPrediction,
) -> ScoredPrediction:
    """Validate alignment and derive the selected candidate and ordinal score."""

    if len(prediction.probabilities) != len(example.candidates):
        raise ValueError("prediction length does not match candidate count")
    selected_index = max(
        range(len(prediction.probabilities)),
        key=prediction.probabilities.__getitem__,
    )
    selected_id = example.candidates[selected_index].id
    expected_score: float | None = None
    if all(candidate.ordinal_value is not None for candidate in example.candidates):
        expected_score = 0.0
        for probability, candidate in zip(
            prediction.probabilities,
            example.candidates,
            strict=True,
        ):
            ordinal_value = candidate.ordinal_value
            assert ordinal_value is not None
            expected_score += probability * ordinal_value
    return ScoredPrediction(
        row_id=example.row_id,
        probabilities=prediction.probabilities,
        gold_probabilities=example.gold_probabilities,
        selected_candidate_id=selected_id,
        gold_candidate_id=example.gold_candidate_id,
        correct=selected_id == example.gold_candidate_id,
        expected_ordinal_score=expected_score,
    )


def negative_log_likelihood(example: DecisionExample, prediction: DecisionPrediction) -> float:
    """Return multiclass log loss for one row."""

    if len(prediction.probabilities) != len(example.candidates):
        raise ValueError("prediction length does not match candidate count")
    return -sum(
        target * math.log(max(probability, 1e-15))
        for target, probability in zip(
            example.gold_probabilities,
            prediction.probabilities,
            strict=True,
        )
    )
